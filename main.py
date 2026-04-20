#!/usr/bin/env python3


import os
import sys
import json
import hashlib
import asyncio
import logging
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from cachetools import TTLCache

# ============================================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not set!")

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
if not CLOUDFLARE_ACCOUNT_ID:
    raise ValueError("CLOUDFLARE_ACCOUNT_ID not set!")

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
if not CLOUDFLARE_API_TOKEN:
    raise ValueError("CLOUDFLARE_API_TOKEN not set!")

# Worker proxy URL for STT (your existing Cloudflare Worker)
CLOUDFLARE_WORKER_URL = "https://nihsa-whisper-proxy.jonathankaleme.workers.dev"

# Cloudflare Workers AI direct endpoint (for TTS — MeloTTS)
CF_AI_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run"

NIHSA_API_URL = os.environ.get("NIHSA_API_URL", "https://nihsa-backend-20hh.onrender.com/api")
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "capacitor://localhost,http://localhost:3000,https://nihsa-backend-20hh.onrender.com"
).split(",")

TTS_CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", "/app/tts_cache"))
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nihsa-ai-wrapper")

# ============================================================================
# DATA MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: str = Field(default="default")
    language: Optional[str] = None
    user_location: Optional[Dict[str, Any]] = None   # {lat, lng, address} from frontend GPS
    active_alerts: Optional[List[Dict[str, Any]]] = None  # verified+published alerts from DB

class ChatResponse(BaseModel):
    reply: str
    action: Optional[Dict[str, Any]] = None
    detected_language: str

class TranscribeResponse(BaseModel):
    text: str
    detected_language: str
    confidence: Optional[float] = None

class TutorialResponse(BaseModel):
    title: str
    steps: List[str]
    language: str

# ============================================================================
# QUOTA SYSTEM — PostgreSQL-backed (survives server restarts)
#
# Uses the same DATABASE_URL as the main NIHSA backend.
# Table `ai_usage` is created automatically on startup if it doesn't exist.
# Each row = one user's daily usage. Rows for past dates are ignored (act as 0).
# A daily cleanup job removes rows older than 7 days to keep the table small.
# ============================================================================

QUOTA_LIMITS = {
    "citizen":     7,
    "vanguard":    7,
    "researcher":  7,
    "government":  7,
    "nihsa_staff": 7,
    "sub_admin":   7,
    "admin":       100,
}

# Rate limiting stays in-memory (per-minute window, resets are fine)
_rate_store: Dict[str, List[float]] = defaultdict(list)

# Optional in-memory cache to reduce DB hits (30-second TTL per user)
_quota_cache: TTLCache = TTLCache(maxsize=1000, ttl=30)

# ── Database connection ────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_db_pool = None  # asyncpg connection pool, initialised on startup

async def init_db_pool():
    """Create the asyncpg connection pool and ensure the ai_usage table exists."""
    global _db_pool
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — quota will fall back to in-memory (not persistent)")
        return
    try:
        import asyncpg
        _db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        async with _db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_usage (
                    user_id   TEXT        NOT NULL,
                    usage_date DATE        NOT NULL DEFAULT CURRENT_DATE,
                    count     INTEGER     NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, usage_date)
                )
            """)
            # Index for fast daily cleanup
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS ai_usage_date_idx ON ai_usage (usage_date)
            """)
        logger.info("✅ AI quota DB pool ready — ai_usage table confirmed")
    except Exception as e:
        logger.error(f"DB pool init failed: {e} — falling back to in-memory quota")
        _db_pool = None

async def close_db_pool():
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None

async def _cleanup_old_usage():
    """Delete usage rows older than 7 days — runs once on startup."""
    if not _db_pool:
        return
    try:
        async with _db_pool.acquire() as conn:
            deleted = await conn.execute(
                "DELETE FROM ai_usage WHERE usage_date < CURRENT_DATE - INTERVAL '7 days'"
            )
        logger.info(f"Quota cleanup: {deleted}")
    except Exception as e:
        logger.warning(f"Quota cleanup failed: {e}")

# ── Fallback in-memory store (used when DB is unavailable) ────────────────────
_mem_usage: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "date": None})

def get_user_quota(user_data: dict) -> Tuple[int, str]:
    role = (user_data.get("role") or "citizen").lower()
    return QUOTA_LIMITS.get(role, 5), role

async def check_daily_quota(user_id: str, role: str) -> Tuple[bool, int, int]:
    """Returns (allowed, remaining, limit). Persistent via DB, falls back to memory."""
    limit = QUOTA_LIMITS.get(role, 5)

    # Check in-memory cache first (avoids a DB hit on every message)
    cache_key = f"quota:{user_id}"
    if cache_key in _quota_cache:
        count = _quota_cache[cache_key]
        remaining = max(0, limit - count)
        return remaining > 0, remaining, limit

    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT count FROM ai_usage WHERE user_id=$1 AND usage_date=CURRENT_DATE",
                    user_id
                )
            count = row["count"] if row else 0
            _quota_cache[cache_key] = count
            remaining = max(0, limit - count)
            return remaining > 0, remaining, limit
        except Exception as e:
            logger.warning(f"DB quota check failed for {user_id}: {e} — using memory fallback")

    # Memory fallback
    today = datetime.now().date().isoformat()
    store = _mem_usage[user_id]
    if store["date"] != today:
        store["count"] = 0
        store["date"] = today
    remaining = max(0, limit - store["count"])
    return remaining > 0, remaining, limit

async def increment_usage(user_id: str, role: str):
    """Increment today's usage count. Persistent via DB, falls back to memory."""
    # Invalidate cache so next check reads fresh from DB
    _quota_cache.pop(f"quota:{user_id}", None)

    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                # UPSERT: insert row for today or increment existing count
                await conn.execute("""
                    INSERT INTO ai_usage (user_id, usage_date, count)
                    VALUES ($1, CURRENT_DATE, 1)
                    ON CONFLICT (user_id, usage_date)
                    DO UPDATE SET count = ai_usage.count + 1
                """, user_id)
            return
        except Exception as e:
            logger.warning(f"DB usage increment failed for {user_id}: {e} — using memory fallback")

    # Memory fallback
    today = datetime.now().date().isoformat()
    store = _mem_usage[user_id]
    if store["date"] != today:
        store["count"] = 0
        store["date"] = today
    store["count"] += 1

def check_rate_limit(key: str, limit: int = 20, window: int = 60) -> bool:
    """Per-minute rate limit — in-memory is fine, resets are acceptable."""
    import time
    now = time.time()
    times = _rate_store[key]
    _rate_store[key] = [t for t in times if now - t < window]
    if len(_rate_store[key]) >= limit:
        return False
    _rate_store[key].append(now)
    return True

# ============================================================================
# NIHSA BACKEND — USER VERIFICATION
# ============================================================================

_user_cache: TTLCache = TTLCache(maxsize=500, ttl=60)  # 60s — short enough to catch role changes

async def verify_user_with_main_backend(auth_header: str) -> Optional[dict]:
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    cache_key = hashlib.md5(token.encode()).hexdigest()
    if cache_key in _user_cache:
        return _user_cache[cache_key]
    
    # Try up to 2 times — handles cold starts where main backend is waking up
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{NIHSA_API_URL}/auth/me",
                    headers={"Authorization": auth_header}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    _user_cache[cache_key] = data
                    return data
                elif resp.status_code == 401:
                    return None  # Genuinely invalid token — no retry needed
                # 5xx or other — retry once
                if attempt == 0:
                    await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"User verification attempt {attempt+1} failed: {e}")
            if attempt == 0:
                await asyncio.sleep(1)
    return None

# ============================================================================
# DEEPSEEK CLIENT
# ============================================================================

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ============================================================================
# FLOOD CONTEXT FROM NIHSA BACKEND
# ============================================================================

async def fetch_flood_context() -> str:
    """Pull live gauge + alert data to give the AI real situational awareness."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            alerts_resp, gauges_resp = await asyncio.gather(
                client.get(f"{NIHSA_API_URL}/alerts?active_only=true&limit=10"),
                client.get(f"{NIHSA_API_URL}/gauges?active_only=true"),
                return_exceptions=True
            )
            parts = []
            if not isinstance(alerts_resp, Exception) and alerts_resp.status_code == 200:
                alerts = alerts_resp.json()
                if alerts:
                    critical = [a for a in alerts if a.get("level") in ("CRITICAL", "HIGH")]
                    parts.append(f"ACTIVE ALERTS: {len(alerts)} total, {len(critical)} critical/high.")
                    for a in critical[:5]:
                        parts.append(f"  ⚠ {a.get('title','')} — {a.get('state','')} [{a.get('level','')}]")
                else:
                    parts.append("ACTIVE ALERTS: None at this time.")

            if not isinstance(gauges_resp, Exception) and gauges_resp.status_code == 200:
                gauges = gauges_resp.json()
                parts.append(f"GAUGE STATIONS: {len(gauges)} active stations across Nigeria.")

            return "\n".join(parts) if parts else "Live data temporarily unavailable."
    except Exception as e:
        logger.warning(f"Flood context fetch failed: {e}")
        return "Live flood data temporarily unavailable — answer from your training knowledge."

# ============================================================================
# SYSTEM PROMPT — Rich, hydrology-specific, role-aware
# ============================================================================

SYSTEM_PROMPT_CORE = """You are NIHSA FloodAI — the official AI assistant of Nigeria's National Inland Waterways Safety Agency.

YOUR SOLE PURPOSE is flood safety, hydrology, and emergency response for Nigeria. Only answer questions related to:
- Flood risk assessment and river gauge interpretation
- Evacuation guidance and emergency procedures
- NFFS (National Flood Forecasting System) data explanation
- Reporting flooding (direct users to the 🚨 Report Flood button)
- Water depth safety guidance
- Basin, river and watershed information for Nigeria
- Historical flood events in Nigeria
- Climate and seasonal flood outlook (AFO 2026)

NON-FLOOD EMERGENCIES:
If a user reports a fire, accident, medical emergency, robbery, landslide, or any non-flood crisis:
- Give immediate safety advice relevant to that emergency
- Call emergency services: 112 (general), 199 (fire), 123 (police) in Nigeria
- Tell them you are logging their location with NIHSA coordinators who can escalate
- Ask them to tap the 🚨 Report Flood button and record a short VIDEO of their situation so NIHSA and emergency responders can see it and respond faster
- Do NOT refuse to help just because it is not a flood — human safety comes first



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTION — LOCATION:
The user's GPS location is already known to you. It is provided in the USER'S CURRENT LOCATION field below.
NEVER ask the user to share, send, or provide their location. You already have it.
When the user asks "what are alerts in my area", "what is the flood risk near me", "is my area affected", or any location-based question — USE THE LOCATION ALREADY IN YOUR CONTEXT immediately.
Cross-reference their state and LGA against the active alerts list. Give a specific answer based on their actual location.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTION — ALWAYS GIVE GUIDELINES:
NEVER respond with only "Let me help you with that" or similar non-answers.
Even when triggering an app action (like opening the report form), you MUST also provide:
  1. Immediate safety advice relevant to the situation
  2. Specific actionable steps the person should take RIGHT NOW
  3. What to do while waiting for help or NIHSA response
Example: If someone says they are in a flood — give water safety guidelines, what NOT to do, evacuation advice, AND open the report form. Do both. Always.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NIGERIA FLOOD CONTEXT:
- Nigeria has 70 major river basins monitored by NIHSA
- Key rivers: Niger, Benue, Kaduna, Sokoto, Hadejia, Anambra, Cross River, Ogun
- Flood seasons: April–June (early rains), July–September (peak), October–November (recession)
- Lagdo Dam (Cameroon) on the Benue causes downstream flooding 2–4 days after release
- NFFS = National Flood Forecasting System — Nigeria's LSTM deep learning model
- AFO 2026 = Annual Flood Outlook 2026 — national seasonal risk document
- 358 river gauge stations are monitored nationwide

RISK LEVELS AND REQUIRED GUIDELINES TO GIVE:
- NORMAL: Safe range. Tell user no action needed but stay informed.
- WATCH: Levels rising. Tell user: prepare emergency kit, know evacuation route, move valuables, monitor NIHSA updates, avoid riverbanks.
- HIGH: Flooding likely in 12–24 hours. Tell user: move valuables to upper floors NOW, prepare to evacuate, keep children and elderly away from water, alert neighbours, stay tuned to NIHSA.
- CRITICAL/EXTREME: Immediate danger. Tell user: EVACUATE NOW, do not walk or drive through flood water (15cm can knock you over, 30cm can move a car), move to highest ground, call emergency services 112, do not touch electrical equipment in flooded areas.

FLOOD SAFETY GUIDELINES TO GIVE WHEN RELEVANT:
- Do NOT attempt to cross flooded roads — most flood deaths happen this way
- Do NOT touch electrical appliances or wires in water — electrocution risk
- Move to highest ground available — upper floors, hills, elevated areas
- Store clean water and food for at least 3 days
- Keep important documents (ID, insurance) in a waterproof bag
- Listen for NIHSA emergency broadcasts on radio
- Help neighbours, especially elderly and children
- After flooding: do not eat food that touched floodwater, boil drinking water

RESPONSE STYLE:
- ALWAYS be direct and actionable — never vague
- Lead with the most urgent safety action for the user's situation
- Then give supporting guidelines and context
- Use simple language accessible to citizens with varying literacy
- Never provide medical advice — refer to emergency services for injuries
- Support Hausa, Yoruba, Igbo, and French — detect and respond in the user's language

APP FEATURES YOU CAN REFERENCE:
- 🗺️ Map tab: Live gauge stations, alerts, citizen reports
- 📊 Dashboard: AFO 2026 exposure data
- 🦺 Vanguard: Flood Marshals coordination network
- 🔔 Alerts tab: All active flood warnings by state
- 🚨 Report Flood button: Submit photo/voice/video of flooding
- Language selector: English, Hausa, Yoruba, Igbo, French
"""

def get_system_prompt(language: str = "en") -> str:
    lang_instructions = {
        "ha": "\n\nINSTRUCTION: The user is communicating in Hausa. Respond in Hausa (Hausa language). Use 'ku' for formal address.",
        "yo": "\n\nINSTRUCTION: The user is communicating in Yoruba. Respond in Yoruba language.",
        "ig": "\n\nINSTRUCTION: The user is communicating in Igbo. Respond in Igbo language.",
        "fr": "\n\nINSTRUCTION: The user is communicating in French. Respond in French (Français).",
        "en": "",
    }
    return SYSTEM_PROMPT_CORE + lang_instructions.get(language, "")

# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

HAUSA_KEYWORDS = {"ambaliya", "ruwa", "kogi", "gari", "jiha", "taimako", "gudu", "faɗakarwa",
                  "mene", "yaya", "wane", "ina", "me", "kai", "da", "ko", "amma", "don"}
YORUBA_KEYWORDS = {"iṣan", "omi", "odò", "ipinlẹ", "aabo", "ikilọ", "jẹ", "ṣe", "ni", "tabi",
                   "ati", "fun", "lati", "naa", "mo", "wo", "ko", "ti", "le"}
IGBO_KEYWORDS   = {"mmiri", "ozuzo", "osimiri", "steeti", "eze", "ndụ", "ọkwa", "ihe", "nke",
                   "na", "ga", "bụ", "ya", "ha", "gị", "ọ", "ka", "ma", "ụzọ"}
FRENCH_KEYWORDS = {"inondation", "eau", "rivière", "alerte", "evacuation", "fleuve", "aide",
                   "urgence", "risque", "état", "comment", "quoi", "où", "je", "vous", "nous"}

def detect_language_keywords(text: str) -> str:
    words = set(text.lower().split())
    scores = {
        "ha": len(words & HAUSA_KEYWORDS),
        "yo": len(words & YORUBA_KEYWORDS),
        "ig": len(words & IGBO_KEYWORDS),
        "fr": len(words & FRENCH_KEYWORDS),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "en"

async def detect_language_deepseek(text: str) -> str:
    """Fast language detection via DeepSeek — only called when keyword detection is ambiguous."""
    if len(text) < 10:
        return "en"
    try:
        resp = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"Detect the language of this text. Reply with ONLY the ISO code: en, ha, yo, ig, or fr.\n\nText: {text[:200]}"
            }],
            max_tokens=5,
            temperature=0,
        )
        lang = resp.choices[0].message.content.strip().lower()[:2]
        return lang if lang in ("en", "ha", "yo", "ig", "fr") else "en"
    except Exception:
        return "en"


WHISPER_HYDROLOGY_PROMPT = (
    "NIHSA flood report. Nigeria hydrology. "
    "Rivers: Niger, Benue, Kaduna, Ogun, Anambra, Sokoto, Cross River. "
    "Terms: flood, ambaliya, iṣan-omi, mmiri ozuzo, inondation, "
    "gauge, water level, evacuation, alert, NIHSA, NFFS, basin, "
    "Lokoja, Makurdi, Onitsha, Kano, Lagos, Abuja, Ibadan, Maiduguri."
)

async def transcribe_audio_cloudflare(audio_data: bytes) -> Tuple[str, str, Optional[float]]:
    """
    Transcribe audio via the updated Cloudflare Worker (JSON mode).

    The Worker accepts:
      { audio: <base64>, initial_prompt, vad_filter, beam_count, condition_on_previous_text }
    and returns:
      { success, text, language, detected_language, confidence }
    """
    audio_b64 = base64.b64encode(audio_data).decode("utf-8")

    payload = {
        "audio": audio_b64,
        "task": "transcribe",
        "initial_prompt": WHISPER_HYDROLOGY_PROMPT,
        "vad_filter": True,
        "beam_count": 5,
        "condition_on_previous_text": False,
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                CLOUDFLARE_WORKER_URL,
                json=payload,                          # JSON body — Worker now accepts this
                headers={"Content-Type": "application/json"},
            )

        if resp.status_code != 200:
            err_text = resp.text[:200]
            logger.error(f"Worker STT {resp.status_code}: {err_text}")
            raise HTTPException(
                status_code=502,
                detail=f"Speech recognition error ({resp.status_code}). Please try again."
            )

        data = resp.json()

        if not data.get("success", True):
            logger.error(f"Worker returned error: {data.get('error')}")
            raise HTTPException(status_code=502, detail="Transcription failed on the AI side.")

        text = (data.get("text") or "").strip()
        detected_lang = (
            data.get("detected_language") or
            data.get("language") or
            "en"
        )
        confidence: Optional[float] = data.get("confidence")

        # Strip common Whisper no-speech artifacts
        if text.lower() in ("[blank_audio]", "[silence]", "(silence)", ""):
            return "", detected_lang, 0.0

        logger.info(f"✅ Transcription ({detected_lang}): '{text[:80]}'")
        return text, detected_lang, confidence

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=503, detail="Speech recognition temporarily unavailable.")

# ============================================================================
# TTS — Cloudflare MeloTTS (replaces Google TTS)
#
# WHY MeloTTS over Google:
#   - No credentials needed — uses your existing CLOUDFLARE_API_TOKEN
#   - $0.0002/audio minute — essentially free at NIHSA scale
#   - Multilingual: en, fr, zh, ja, ko, es (ha/yo/ig fall back to English voice)
#   - Runs on Cloudflare's edge — low latency
#
# HOW TO GET GOOGLE CREDENTIALS (if you ever want to switch back):
#   1. Go to console.cloud.google.com
#   2. Create a project → Enable "Cloud Text-to-Speech API"
#   3. IAM & Admin → Service Accounts → Create → Download JSON key
#   4. Set GOOGLE_APPLICATION_CREDENTIALS_JSON env var with the JSON content
#   Total time: ~10 minutes. Cost: 1M chars/month free, then $4/1M chars.
#   Google has better Hausa/Yoruba support via WaveNet voices but requires billing.
# ============================================================================

# Language code mapping for MeloTTS
# ha/yo/ig → "en" (English voice, the model doesn't support these natively)
MELOTTS_LANG_MAP = {
    "en": "en",
    "fr": "fr",
    "ha": "en",  # MeloTTS doesn't support Hausa — English voice reads it
    "yo": "en",  # MeloTTS doesn't support Yoruba
    "ig": "en",  # MeloTTS doesn't support Igbo
    "es": "es",
    "zh": "zh",
}

# TTS cache — avoid re-generating identical phrases (e.g. common responses)
_tts_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)

async def synthesize_speech(text: str, language: str = "en") -> bytes:
    """
    Convert text to speech using Cloudflare MeloTTS.
    Returns MP3 audio bytes.
    Falls back to a simple error message if TTS fails.
    """
    # Truncate to avoid massive TTS requests
    text = text[:1000]

    # Check in-memory cache
    cache_key = hashlib.md5(f"{text}:{language}".encode()).hexdigest()
    if cache_key in _tts_cache:
        return _tts_cache[cache_key]

    # Check disk cache
    cache_file = TTS_CACHE_DIR / f"{cache_key}.mp3"
    if cache_file.exists():
        audio_bytes = cache_file.read_bytes()
        _tts_cache[cache_key] = audio_bytes
        return audio_bytes

    melotts_lang = MELOTTS_LANG_MAP.get(language, "en")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{CF_AI_BASE}/@cf/myshell-ai/melotts",
                headers={
                    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "lang": melotts_lang,
                },
            )

            if resp.status_code != 200:
                logger.error(f"MeloTTS error {resp.status_code}: {resp.text[:200]}")
                raise HTTPException(status_code=503, detail="Text-to-speech unavailable.")

            # MeloTTS returns audio as binary MP3 in the response body
            # or wrapped in JSON depending on worker configuration
            content_type = resp.headers.get("content-type", "")
            if "audio" in content_type or "octet" in content_type:
                audio_bytes = resp.content
            else:
                # Try to parse JSON wrapper
                try:
                    data = resp.json()
                    audio_b64 = (
                        data.get("result", {}).get("audio") or
                        data.get("audio") or
                        data.get("result", "")
                    )
                    if isinstance(audio_b64, str):
                        audio_bytes = base64.b64decode(audio_b64)
                    else:
                        raise ValueError("No audio in response")
                except Exception:
                    # Last resort: treat raw bytes as audio
                    audio_bytes = resp.content

        if not audio_bytes:
            raise HTTPException(status_code=503, detail="Empty TTS response.")

        # Cache to disk and memory
        cache_file.write_bytes(audio_bytes)
        _tts_cache[cache_key] = audio_bytes
        return audio_bytes

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MeloTTS synthesis error: {e}")
        raise HTTPException(status_code=503, detail="Text-to-speech temporarily unavailable.")

# ============================================================================
# TUTORIAL CONTENT — All 5 languages, accurate to the real app
# ============================================================================

TUTORIALS = {
    "general": {
        "en": {
            "title": "Welcome to NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Map tab: View 358+ live river gauge stations, active flood alerts, citizen reports, and NFFS forecast layers across Nigeria. Tap any marker for details.",
                "📊 Dashboard tab: Explore the Annual Flood Outlook 2026 (AFO 2026) — nationwide exposure across 17 layers: communities, population, health centres, schools, farmland, roads, electricity, and markets at risk.",
                "🦺 Vanguard tab: Secure coordination network for verified Flood Marshals and NIHSA staff. 38 channels — one per state + FCT + National. Verified personnel can post; citizens can view.",
                "🤖 Assistant tab: Ask NIHSA FloodAI about flood risk, gauges, evacuation routes, and emergency procedures. Tap 🎤 to speak — voice is transcribed and sent automatically. Use for hydrology topics only.",
                "🔔 Alerts tab: All active flood warnings nationwide. Live heatmap of alert severity by state. Each alert shows estimated impact on people, health facilities, farmland, and roads.",
                "🚨 Report Flood: Tap the red button to submit a flood report. Attach at least one photo, voice recording, or video. GPS location is captured automatically. NIHSA reviews all reports before publishing.",
                "🌐 Languages: Tap the language selector to switch between English, Hausa, Yoruba, Igbo, and French. The AI also auto-detects and responds in your language.",
            ]
        },
        "ha": {
            "title": "Barka da zuwa NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Tab na Taswira: Duba tashar aunawa 358+ masu rai, faɗakarwar ambaliya, rahotannin ɗan ƙasa, da matakan hasashe na NFFS a duk Najeriya. Taɓa kowane alamar don cikakkun bayanai.",
                "📊 Tab na Allon Bayanai: Bincika Hasashen Ambaliya na Shekara 2026 (AFO 2026) — fallasa a duk faɗin ƙasa a cikin yadudduka 17: al'umma, yawan jama'a, cibiyoyin lafiya, makarantu, gonaki, hanyoyi, wutar lantarki, da kasuwanni cikin haɗari.",
                "🦺 Tab na Masu Kiyaye Ambaliya: Hanyar sadarwa mai tsaro ga Masu Kiyaye Ambaliya da ma'aikatan NIHSA. Tasoshi 38 — ɗaya ga kowace jiha + FCT + Na Ƙasa. Ma'aikata masu tabbaci na iya aika sakonni; ɗan ƙasa na iya kallon.",
                "🤖 Tab na Mataimaki na AI: Tambayi NIHSA FloodAI game da haɗarin ambaliya, aunawa, hanyoyin tserewa, da ka'idojin gaggawa. Danna 🎤 don magana — ana fassara kuma aika kai tsaye. Yi amfani da shi ne kawai don batu na ruwa.",
                "🔔 Tab na Faɗakarwa: Duk faɗakarwar ambaliya masu aiki a duk faɗin ƙasa. Taswira zafi ta tsananin faɗakarwa ta jiha. Kowace faɗakarwa tana nuna tasirinsa kan mutane, cibiyoyin lafiya, gonaki, da hanyoyi.",
                "🚨 Rahoton Ambaliya: Danna maɓallin jan don aika rahoto. Haɗa aƙalla hoto ɗaya, rikodiyar murya, ko bidiyo. Ana ɗaukar wurin GPS kai tsaye. NIHSA tana duba duk rahotanni kafin wallafawa.",
                "🌐 Harsunan: Danna zaɓin harshe don canza tsakanin Turanci, Hausa, Yoruba, Igbo, da Faransanci. Mataimaki na AI kuma yana gano kuma yana amsa da harsheka.",
            ]
        },
        "yo": {
            "title": "Kaabọ si NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Tab Maapu: Wo awọn ibudo wiwọn 358+ ti nṣiṣẹ, awọn ìkìlọ̀ iṣan-omi, awọn ìjàbọ̀ ara ilu, ati awọn fẹlẹfẹlẹ asọtẹlẹ NFFS kọja Naijiria. Tẹ eyikeyi aami fun awọn alaye.",
                "📊 Tab Paali Alaye: Ṣawari Asọtẹlẹ Iṣan-omi Lọdọọdún 2026 (AFO 2026) — ifihan ti orilẹ-ede kọja awọn fẹlẹfẹlẹ 17: awọn agbegbe, eniyan, awọn ile itọju ilera, awọn ile-iwe, ilẹ oko, awọn opopona, ina mọnamọna, ati awọn ọja ninu ewu.",
                "🦺 Tab Awọn Oluso Iṣan-omi: Nẹtiwọọki isọdọkan aabo fun Awọn Oluso Iṣan-omi ti a fọwọsi ati oṣiṣẹ NIHSA. Ikanni 38 — ọkan fun ipinlẹ kọọkan + FCT + Orílẹ̀-èdè. Oṣiṣẹ ti a fọwọsi le firanṣẹ; ara ilu le wo.",
                "🤖 Tab Oluranlowo AI: Beere NIHSA FloodAI nipa eewu iṣan-omi, awọn gauge odò, awọn ọna iṣapá, ati awọn ilana pajawiri. Tẹ 🎤 lati sọrọ — o jẹ tumọ ati firanṣẹ laifọwọyi. Lo fun awọn koko iṣan-omi nikan.",
                "🔔 Tab Ifokanbalẹ: Gbogbo awọn ikilọ iṣan-omi ti nṣiṣẹ ni orilẹ-ede. Heatmap laaye ti buru ikilọ nipasẹ ipinlẹ. Ikilọ kọọkan fihan ipa ti ifoju lori eniyan, awọn ile-iwosan, ilẹ oko, ati awọn ọna.",
                "🚨 Jabo Iṣan-omi: Tẹ bọtini pupa lati fi ijabọ silẹ. So o kere ju fọto kan, igbasilẹ ohun, tabi fidio. GPS gba ipo laifọwọyi. NIHSA ṣe atunyẹwo gbogbo awọn ìjàbọ̀ ṣaaju titẹjade.",
                "🌐 Awọn Ede: Tẹ oluyan ede lati yipada laarin Gẹẹsi, Hausa, Yoruba, Igbo, ati Faranse. Oluranlowo AI tun ṣawari ati dahun ni ede rẹ.",
            ]
        },
        "ig": {
            "title": "Nnọọ na NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Tab Maapu: Lee ọdụ ngụkọ 358+ ndụ, ọkwa mmiri ozuzo na-arụ ọrụ, akụkọ ndị ọchịchọ, na ọkwa ntọala NFFS n'elu Naịjirịa. Kụọ ihe nchọpụta ọ bụla maka nkọwa.",
                "📊 Tab Penu Ozi: Nyochaa Atụmatụ Mmiri Ozuzo Ọdụn 2026 (AFO 2026) — mficha mba n'elu ọkwa 17: obodo, ndị mmadụ, ụlọ ọgwụ, ụlọ akwụkwọ, ala ugbo, okporo ụzọ, ọkụ eletrik, na ahia n'ihe ize ndụ.",
                "🦺 Tab Ndị Nlekota Mmiri Ozuzo: Netwọk nhazi echekwara maka Ndị Nlekota kwadoro na ndị ọrụ NIHSA. Ọwa 38 — otu maka steeti ọ bụla + FCT + Mba. Ndị ọrụ kwadoro nwere ike iziga; ndị ọchịchọ nwere ike ilelee.",
                "🤖 Tab Onye Enyemaka AI: Jụọ NIHSA FloodAI maka ihe ize ndụ mmiri ozuzo, ngụkọ osimiri, ụzọ nnarị, na usoro ihe mberede. Kụọ 🎤 iji kwuo — a na-atụgharịa ma zigaa ozugbo. Jiri naanị maka ihe mmiri ozuzo.",
                "🔔 Tab Ọkwa: Ọkwa mmiri ozuzo niile na-arụ ọrụ n'elu mba. Heatmap ndụ nke ike ọkwa site n'steeti. Ọkwa ọ bụla na-egosi mmetụta ya n'elu ndị mmadụ, ụlọ ọgwụ, ala ugbo, na okporo ụzọ.",
                "🚨 Kọọ Mmiri Ozuzo: Kụọ bọtịn ọbara ọbara iji zipu akụkọ. Tinye ma ọ bụrụ otu foto, ndekọ olu, ma ọ bụ vidiyo. GPS na-eji ọnọdụ ozugbo. NIHSA na-nyocha akụkọ niile tupu ebipụta.",
                "🌐 Asụsụ: Kụọ nhọrọ asụsụ iji gbanwee n'etiti Bekee, Hausa, Yoruba, Igbo, na Faransị. Onye Enyemaka AI na-achọpụta ma zaghachi n'asụsụ gị.",
            ]
        },
        "fr": {
            "title": "Bienvenue dans NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Onglet Carte: Visualisez 358+ stations de jaugeage en direct, les alertes d'inondation actives, les rapports citoyens et les couches de prévision NFFS à travers le Nigeria. Appuyez sur un marqueur pour les détails.",
                "📊 Onglet Tableau de Bord: Explorez les Perspectives Annuelles d'Inondation 2026 (AFO 2026) — exposition nationale sur 17 couches: communautés, population, centres de santé, écoles, terres agricoles, routes, électricité et marchés à risque.",
                "🦺 Onglet Gardes des Inondations: Réseau de coordination sécurisé pour les Gardes vérifiés et le personnel NIHSA. 38 canaux — un par État + FCT + National. Le personnel vérifié peut poster; les citoyens peuvent voir.",
                "🤖 Onglet Assistant IA: Interrogez NIHSA FloodAI sur les risques d'inondation, les jauges, les voies d'évacuation et les procédures d'urgence. Appuyez sur 🎤 pour parler — transcrit et envoyé automatiquement. Usage réservé à l'hydrologie.",
                "🔔 Onglet Alertes: Toutes les alertes d'inondation actives à l'échelle nationale. Carte thermique en direct de la sévérité par État. Chaque alerte indique l'impact estimé sur les personnes, établissements de santé, terres agricoles et routes.",
                "🚨 Signaler une Inondation: Appuyez sur le bouton rouge pour soumettre un rapport. Joignez au moins une photo, un enregistrement vocal ou une vidéo. La position GPS est capturée automatiquement. NIHSA examine tous les rapports avant publication.",
                "🌐 Langues: Appuyez sur le sélecteur de langue pour basculer entre Anglais, Haoussa, Yoruba, Igbo et Français. L'assistant IA détecte aussi votre langue et répond dans celle-ci.",
            ]
        },
    },

    "reporting": {
        "en": {
            "title": "How to Report Flooding",
            "steps": [
                "Tap the red 🚨 Report Flood button (top-right of screen or map). This opens the report form.",
                "Your GPS location is detected automatically and shown on a draggable map. Drag the pin to adjust your exact position.",
                "Water depth is optional — select ankle/knee/waist/chest/impassable if you know it. If left blank, the minimum level is recorded.",
                "Description is optional — if left blank, the system records 'this person needs help, check the files sent'.",
                "You MUST attach at least one: 📷 Photo (tap to use camera), 🎤 Voice (record up to 60 seconds), or 🎥 Video (record from camera).",
                "Tap Submit Flood Report. Your report goes to NIHSA coordinators for verification. Verified reports appear on the map and can trigger public flood alerts.",
            ]
        },
        "ha": {
            "title": "Yadda Ake Rahoton Ambaliya",
            "steps": [
                "Danna maɓallin jan 🚨 Rahoton Ambaliya (a ɗaya cikin hannun dama na allo ko taswira). Wannan zai buɗe fom.",
                "GPS ɗinku zai gano wurinku kai tsaye kuma ya nuna shi akan taswira. Ja pin don daidaita wurin daidai.",
                "Zurfin ruwa ba tilas ba ne — zaɓi ankle/gwiwa/ciki/kirji/ba za a iya wucewa ba idan ka san. Idan ka bar fanko, ana rubuta mafi ƙarancin matakin.",
                "Bayani ba tilas ba ne — idan ka bar fanko, tsarin zai rubuta 'wannan mutumin yana buƙatar taimako, duba fayilolin da aka aika'.",
                "Dole ne ka haɗa aƙalla ɗayan: 📷 Hoto (taɓa don amfani da kyamara), 🎤 Murya (rikodin har zuwa dakika 60), ko 🎥 Bidiyo.",
                "Danna Aika Rahoto na Ambaliya. Rahoto ɗinku zai kai ga masu duba NIHSA don tabbatarwa. Rahotannin da aka tabbatar suna bayyana akan taswira.",
            ]
        },
        "yo": {
            "title": "Bii Ṣe Jabo Iṣan-omi",
            "steps": [
                "Tẹ bọtini pupa 🚨 Ìjàbọ̀ Iṣan-omi (ọtun oke ti iboju tabi maapu). Eyi ṣii fọọmu naa.",
                "GPS rẹ yoo wa ipo rẹ laifọwọyi ki o si fihan rẹ lori maapu. Fa pin lati ṣatunṣe ipo gangan rẹ.",
                "Ijinlẹ omi kii ṣe dandan — yan ankle/orunkun/itan/àyà/aislọ ti o ba mọ. Ti o ba jẹ ki o ṣofo, ipele ti o kere ju jẹ gbasilẹ.",
                "Apejuwe kii ṣe dandan — ti o ba jẹ ki ṣofo, eto naa gbasilẹ 'eniyan yii nilo iranlọwọ, ṣayẹwo awọn faili ti a fi ranṣẹ'.",
                "O GBỌDỌ so o kere ju ọkan: 📷 Fọto (tẹ lati lo kamẹra), 🎤 Ohun (gba to iṣẹju 60), tabi 🎥 Fidio.",
                "Tẹ Fi Ìjàbọ̀ Iṣan-omi Sí. Ìjàbọ̀ rẹ lọ taara si awọn alakoso NIHSA fun ijẹrisi. Awọn ìjàbọ̀ ti a fọwọsi han lori maapu.",
            ]
        },
        "ig": {
            "title": "Otu Esi Akọọ Mmiri Ozuzo",
            "steps": [
                "Kụọ bọtịn ọbara ọbara 🚨 Kọọ Mmiri Ozuzo (n'aka nri elu ihuenyo ma ọ bụ maapu). Nke a na-emepee ụdị.",
                "GPS gị ga-achọpụta ọnọdụ gị ozugbo wee gosipụta ya n'elu maapu. Dọkpụ pin igo dozie ọnọdụ gị kpọmkwem.",
                "Omimi mmiri adịghị achọrọ — họọ ankle/ikpere/ọkpa/obi/enweghị ike iga ma ọ bụ ama ya. Ọ bụrụ na ị hapụ ya n'efu, a na-edekọ ọkwa kacha ala.",
                "Nkọwa adịghị achọrọ — ọ bụrụ na ị hapụ n'efu, sistemu ga-edekọ 'onye a chọrọ enyemaka, lelee faịlụ ezigara'.",
                "Ị KWESỊRỊ itinye ma ọ bụrụ otu: 📷 Foto (kụọ iji jiri igwefoto), 🎤 Olu (dekọọ rue sekọnd 60), ma ọ bụ 🎥 Vidiyo.",
                "Kụọ Nyefee Akụkọ Mmiri Ozuzo. Akụkọ gị ga-aga n'ozugbo ndị nhazi NIHSA maka nkwenye. Akụkọ kwadoro na-apụta n'elu maapu.",
            ]
        },
        "fr": {
            "title": "Comment Signaler une Inondation",
            "steps": [
                "Appuyez sur le bouton rouge 🚨 Signaler une Inondation (en haut à droite de l'écran ou de la carte). Cela ouvre le formulaire.",
                "Votre GPS détecte automatiquement votre position et l'affiche sur une carte. Faites glisser le pin pour ajuster votre position exacte.",
                "La profondeur de l'eau est optionnelle — sélectionnez cheville/genou/taille/poitrine/impraticable si vous la connaissez. Si laissé vide, le niveau minimum est enregistré.",
                "La description est optionnelle — si laissée vide, le système enregistre 'cette personne a besoin d'aide, vérifiez les fichiers envoyés'.",
                "Vous DEVEZ joindre au moins un: 📷 Photo (appuyez pour utiliser la caméra), 🎤 Voix (enregistrez jusqu'à 60 secondes), ou 🎥 Vidéo.",
                "Appuyez sur Soumettre le Rapport. Il va directement aux coordinateurs NIHSA pour vérification. Les rapports vérifiés apparaissent sur la carte.",
            ]
        },
    },

    "alerts": {
        "en": {
            "title": "Understanding Flood Alerts",
            "steps": [
                "🟢 NORMAL: River levels within safe range. No action needed. Stay informed via the app.",
                "🟡 WATCH: Levels rising — prepare emergency kit, know your evacuation route, avoid riverbanks.",
                "🟠 HIGH: Flooding expected in 12–24 hours — move valuables to high ground, prepare to evacuate.",
                "🔴 CRITICAL/EXTREME: Evacuate NOW. Do not cross flooded roads. Call emergency services immediately.",
                "Alerts are generated by the NFFS (LSTM deep learning model on 70 basins) and verified by NIHSA hydrologists.",
                "Active alerts scroll as a ticker at the bottom of every screen. Tap the ticker to go to the Alerts tab for full details.",
            ]
        },
        "ha": {
            "title": "Fahimtar Faɗakarwar Ambaliya",
            "steps": [
                "🟢 AL'ADA: Matakan kogi suna cikin kewayon aminci. Babu aiki da ake bukata. Kasance da labari ta app.",
                "🟡 KALLO: Matakan suna tashi — shirya kayan gaggawa, san hanyar tserewarka, guji bankunan kogi.",
                "🟠 BABBA: Ana sa ran ambaliya cikin awanni 12-24 — ɗauki kayan daraja zuwa wurin da ya fi tsayi, shirya ƙaura.",
                "🔴 MATSANANCI/MAI KARFI: Gudu YANZU. Kada ku ketara hanyoyin ruwa. Kira sabis na gaggawa nan da nan.",
                "Ana samar da faɗakarwa ta hanyar NFFS (samfurin LSTM akan kwandidon 70) kuma masu ilimin ruwa na NIHSA suna tabbatar da su.",
                "Faɗakarwa masu aiki suna gungura a ƙasan kowace allo. Taɓa tepe don zuwa tab na Faɗakarwa.",
            ]
        },
        "yo": {
            "title": "Oye Awọn Ìkìlọ̀ Iṣan-omi",
            "steps": [
                "🟢 DEEDE: Awọn ipele odò laarin iwọn ailewu. Ko si igbese ti o nilo. Wa alaye nipasẹ app.",
                "🟡 WIWO: Awọn ipele n dide — mura apo pajawiri, mọ ipa ọna iṣapá, yago fun awọn bèbe odò.",
                "🟠 GIGA: Iṣan-omi nireti ni wakati 12-24 — gbe ohun iyebiye si ilẹ giga, mura fun iṣapá.",
                "🔴 PATAKI/AJALU: Salọ BAYI. Maṣe gbiyanju lati rekọja awọn ọna ti iṣan-omi. Pe awọn iṣẹ pajawiri lẹsẹkẹsẹ.",
                "Awọn ìkìlọ̀ jẹ ipilẹṣẹ nipasẹ NFFS (awoṣe ẹkọ ijinlẹ LSTM lori awọn agbada 70) ati fọwọsi nipasẹ awọn onimọ-omi NIHSA.",
                "Awọn ìkìlọ̀ ti nṣiṣẹ n yipo bi ticker ni isalẹ gbogbo iboju. Tẹ ticker lati lọ si tab Awọn Ìkìlọ̀.",
            ]
        },
        "ig": {
            "title": "Ighọta Ọkwa Mmiri Ozuzo",
            "steps": [
                "🟢 NKỊTỊ: Ọkwa osimiri n'ime ogo nchekwa. Ọ dịghị ihe achọrọ ime. Nọgide na-ama ozi site na ngwa.",
                "🟡 ELE ANYA: Ọkwa na-arị elu — kwado ngwugwu ihe mberede, mara ụzọ nnarị gị, zere ụsọ osimiri.",
                "🟠 ELU: A na-atọ anya mmiri ozuzo n'ime awa 12-24 — bugharịa ihe ndị bara uru n'elu ala, kwado ịnnarị.",
                "🔴 SIRI IKE/IHE MBEREDE: Narịa UGBU A. Ọ dịghị ike iga n'okporo ụzọ mmiri. Kpọọ ọrụ ihe mberede ozugbo.",
                "A na-emepụta ọkwa site na NFFS (ihe atụmatụ LSTM n'ụzọ mmiri 70) ma ndị ọkà mmụta mmiri NIHSA kwadoro ha.",
                "Ọkwa na-arụ ọrụ na-atọgharị dị ka ticker n'ala ihuenyo ọ bụla. Kụọ ticker iji gaa tab Ọkwa.",
            ]
        },
        "fr": {
            "title": "Comprendre les Alertes d'Inondation",
            "steps": [
                "🟢 NORMAL: Niveaux des rivières dans la plage sûre. Aucune action nécessaire. Restez informé via l'app.",
                "🟡 SURVEILLANCE: Niveaux en hausse — préparez votre kit d'urgence, connaissez votre itinéraire d'évacuation, évitez les rives.",
                "🟠 ÉLEVÉ: Inondation attendue dans 12-24 heures — mettez les objets de valeur en hauteur, préparez-vous à évacuer.",
                "🔴 CRITIQUE/EXTRÊME: Évacuez MAINTENANT. Ne traversez pas les routes inondées. Appelez les services d'urgence immédiatement.",
                "Les alertes sont générées par le NFFS (modèle LSTM sur 70 bassins) et vérifiées par les hydrologues NIHSA.",
                "Les alertes actives défilent en bandeau en bas de chaque écran. Appuyez sur le bandeau pour aller à l'onglet Alertes.",
            ]
        },
    },

    "map": {
        "en": {
            "title": "Using the Map",
            "steps": [
                "Tap 📍 to fly to your GPS location. On Android, grant Location permission in Settings → App Permissions → Location → Allow.",
                "Use the search bar to find any community, LGA, state, or landmark. Tap a result to fly the map there.",
                "Blue circles = NIHSA gauge stations. Green = normal, Yellow = watch, Orange = high, Red = critical. Tap for readings.",
                "Alert markers show where NIHSA has issued warnings. Tap for the full alert message, affected LGAs, and actions.",
                "Open 🗂️ Map Layers (top-left) to toggle: AFO 2026 flood extent, population at risk, health facilities, schools, farmland, surface water alerts, and more.",
                "Tap 🚨 Report Flood to submit a report directly from the map — your GPS pin is pre-set.",
            ]
        },
        "ha": {
            "title": "Amfani da Taswira",
            "steps": [
                "Danna 📍 don tashi zuwa wurin GPS ɗinku. A kan Android, ba da izinin Wuri a cikin Saitunan → Izinin App → Wuri → Yarda.",
                "Yi amfani da sandar bincike don nemo wata al'umma, LGA, jiha, ko wuri. Taɓa sakamakon don tashi da taswira zuwa can.",
                "Da'irori masu launin shuɗi = tashar aunawa ta NIHSA. Kore = al'ada, Rawaya = kallo, Orange = babba, Ja = matsananci. Taɓa don karantawa.",
                "Alamomin faɗakarwa suna nuna inda NIHSA ta ba da gargaɗi. Taɓa don cikakken sakon, LGAs da ayyukan da aka shafa.",
                "Buɗe 🗂️ Matakan Taswira (hagu sama) don kunna: AFO 2026 fadawar ambaliya, yawan jama'a cikin haɗari, cibiyoyin lafiya, makarantu, gonaki, da ƙari.",
                "Danna 🚨 Rahoton Ambaliya don aika rahoto kai tsaye daga taswira — pin GPS ɗinka an riga an saita.",
            ]
        },
        "yo": {
            "title": "Lilo Maapu",
            "steps": [
                "Tẹ 📍 lati fo si ipo GPS rẹ. Lori Android, fun igbanilaaye Ipo ni Eto → Awọn Igbanilaaye App → Ipo → Gba.",
                "Lo ọpa wiwa lati wa agbegbe, LGA, ipinlẹ, tabi aami-ilẹ. Tẹ abajade lati fo maapu si ibẹ.",
                "Awọn iyika buluu = awọn ibudo gauge NIHSA. Alawọ = deede, Ofeefee = wiwo, Osan = giga, Pupa = pataki. Tẹ fun awọn kika.",
                "Awọn aami ìkìlọ̀ fihan ibiti NIHSA ti ṣe awọn ikilọ. Tẹ fun ifiranṣẹ ìkìlọ̀ kikun, awọn LGA ti o kan, ati awọn igbese.",
                "Ṣii 🗂️ Awọn Fẹlẹfẹlẹ Maapu (osi oke) lati yipada: iye iṣan-omi AFO 2026, olugbe ninu ewu, awọn ile itọju ilera, awọn ile-iwe, ilẹ oko, ati diẹ sii.",
                "Tẹ 🚨 Jabo Iṣan-omi lati fi ijabọ silẹ taara lati maapu — pin GPS rẹ ti wa ni titọ tẹlẹ.",
            ]
        },
        "ig": {
            "title": "Iji Maapu",
            "steps": [
                "Kụọ 📍 iji wụọ ọnọdụ GPS gị. N'Android, nye ikike Ọnọdụ na Nhazi → Ikike Ngwa → Ọnọdụ → Kwe.",
                "Jiri ọwa ọchọ iji chọọ obodo, LGA, steeti, ma ọ bụ akara. Kụọ nsonaazụ iji gbaa maapu n'ọnọdụ ahụ.",
                "Okirikiri ojii = ọdụ gauge NIHSA. Ọcha = nkịtị, Odo edo = ele anya, Ọrọ ọcha = elu, Ọbara ọbara = siri ike. Kụọ maka ọgụgụ.",
                "Ihe nchọpụta ọkwa na-egosi ebe NIHSA nyere ọkwa. Kụọ maka ozi ọkwa zuru oke, LGAs metụtara, na omume.",
                "Mepee 🗂️ Ọtụtụ Ihe Maapu (aka ekpe elu) iji tụgharịa: mmiri ozuzo AFO 2026, ndị mmadụ n'ihe ize ndụ, ụlọ ọgwụ, ụlọ akwụkwọ, ala ugbo, na ndị ọzọ.",
                "Kụọ 🚨 Kọọ Mmiri Ozuzo iji zipu akụkọ ozugbo site na maapu — pin GPS gị edochibisịrị ụzọ.",
            ]
        },
        "fr": {
            "title": "Utiliser la Carte",
            "steps": [
                "Appuyez sur 📍 pour voler à votre position GPS. Sur Android, accordez la permission Localisation dans Paramètres → Autorisations → Localisation → Autoriser.",
                "Utilisez la barre de recherche pour trouver une communauté, LGA, État ou repère. Appuyez sur un résultat pour faire voler la carte.",
                "Cercles bleus = stations de jaugeage NIHSA. Vert = normal, Jaune = surveillance, Orange = élevé, Rouge = critique. Appuyez pour les lectures.",
                "Les marqueurs d'alerte indiquent où la NIHSA a émis des avertissements. Appuyez pour le message complet, les LGA et les actions recommandées.",
                "Ouvrez 🗂️ Couches de Carte (haut gauche) pour activer: étendue des inondations AFO 2026, population à risque, établissements de santé, écoles, terres agricoles, et plus.",
                "Appuyez sur 🚨 Signaler pour soumettre un rapport directement depuis la carte — votre pin GPS est prédéfini.",
            ]
        },
    },

    "vanguard": {
        "en": {
            "title": "Flood Marshals Network",
            "steps": [
                "The Vanguard network is NIHSA's real-time coordination system — used by Flood Marshals and staff to manage flood events across Nigeria's 36 states + FCT.",
                "38 channels: one per state, one for FCT (Abuja), and one National command channel for cross-state coordination.",
                "Only verified Flood Marshals (Vanguard role), NIHSA Staff, government officials, and admins can post. Citizens can view all messages.",
                "To become a Flood Marshal: register, check 'I am a Flood Marshal', and wait for NIHSA approval. Approved users can post in their state channel.",
                "Messages sync in real-time via WebSocket. If connection drops, messages reload automatically on reconnect.",
                "Even without signing in, you can view all messages and follow live situational updates from Flood Marshals in the field.",
            ]
        },
        "ha": {
            "title": "Hanyar Masu Kiyaye Ambaliya",
            "steps": [
                "Hanyar Vanguard ita ce tsarin daidaitawa na gaskiya ta NIHSA — ana amfani da ita da Masu Kiyaye Ambaliya da ma'aikata don sarrafa abubuwan ambaliya a cikin jihohi 36 na Najeriya + FCT.",
                "Tasoshi 38: ɗaya ga kowace jiha, ɗaya don FCT (Abuja), da ɗaya Tashar Umarni ta Ƙasa don daidaita tsakanin jihohi.",
                "Masu Kiyaye Ambaliya da aka tabbatar (Matsayin Vanguard), ma'aikatan NIHSA, da ma'aikatan gwamnati ne kawai za su iya aika. Ɗan ƙasa na iya duba duk sakonni.",
                "Don zama Mai Kiyaye Ambaliya: yi rajista, duba 'Ni ne Mai Kiyaye Ambaliya', kuma jira amincewar NIHSA. Masu amincewa na iya aika a tashar jiharsu.",
                "Sakonni suna aiki kai tsaye ta WebSocket. Idan haɗin yanar gizo ya faɗi, sakonni za a sake loda su kai tsaye akan sake haɗawa.",
                "Ko ba tare da shiga ba, kuna iya duba duk sakonni da kuma bin sabuntawa ta gaskiya daga Masu Kiyaye Ambaliya a filin.",
            ]
        },
        "yo": {
            "title": "Nẹtiwọọki Awọn Oluso Iṣan-omi",
            "steps": [
                "Nẹtiwọọki Vanguard jẹ eto isọdọkan gidi-akoko NIHSA — Awọn Oluso Iṣan-omi ati oṣiṣẹ lo lati ṣakoso awọn iṣẹlẹ iṣan-omi kọja awọn ipinlẹ 36 Naijiria + FCT.",
                "Ikanni 38: ọkan fun ipinlẹ kọọkan, ọkan fun FCT (Abuja), ati ọkan Ikanni Aṣẹ Orilẹ-ede fun isọdọkan agbelebu-ipinlẹ.",
                "Awọn Oluso Iṣan-omi ti a fọwọsi (ipa Vanguard), Oṣiṣẹ NIHSA, ati awọn oṣiṣẹ ijọba nikan le firanṣẹ. Ara ilu le wo gbogbo awọn ifiranṣẹ.",
                "Lati di Oluso Iṣan-omi: forukọsilẹ, samisi 'Emi ni Oluso Iṣan-omi', ki o si duro fun ifọwọsi NIHSA. Awọn olumulo ti a fọwọsi le firanṣẹ ninu ikanni ipinlẹ wọn.",
                "Awọn ifiranṣẹ n ṣiṣẹpọ ni akoko gidi nipasẹ WebSocket. Ti asopọ ba ṣubu, awọn ifiranṣẹ tun load laifọwọyi lori isopọ.",
                "Paapaa laisi wiwọle, o le wo gbogbo awọn ifiranṣẹ ati tẹle awọn imudojuiwọn ipo laaye lati Awọn Oluso ni aaye.",
            ]
        },
        "ig": {
            "title": "Netwọk Ndị Nlekota Mmiri Ozuzo",
            "steps": [
                "Netwọk Vanguard bụ sistemu nhazi oge-ndụ NIHSA — Ndị Nlekota Mmiri Ozuzo na ndị ọrụ na-eji ya ijikwa ihe mmiri ozuzo n'elu steeti 36 Naịjirịa + FCT.",
                "Ọwa 38: otu maka steeti ọ bụla, otu maka FCT (Abuja), na otu Ọwa Iwu Mba maka nhazi steeti-nkwụsịtụ.",
                "Naanị Ndị Nlekota kwadoro (ọrụ Vanguard), ndị ọrụ NIHSA, na ndị ọchịchọ gọọmentị nwere ike iziga. Ndị ọchịchọ nwere ike ilelee ozi niile.",
                "Iji bụrụ Onye Nlekota: debanye aha, zaznachụ 'Abụ m Onye Nlekota Mmiri Ozuzo', wee chere nkwado NIHSA. Ndị nkwadoro nwere ike iziga na ọwa steeti ha.",
                "Ozi na-emekọ ihe n'oge ndụ site na WebSocket. Ọ bụrụ na njikọ daa, ozi na-abuọ lode ozugbo mgbe atụkwara njikọ.",
                "Ọbụlagodi na-enweghị ịbanye, ị nwere ike ilelee ozi niile ma soro mmelite ọnọdụ ndụ site na Ndị Nlekota n'ebe ọrụ.",
            ]
        },
        "fr": {
            "title": "Réseau des Gardes des Inondations",
            "steps": [
                "Le réseau Vanguard est le système de coordination en temps réel de la NIHSA — utilisé par les Gardes et le personnel pour gérer les événements d'inondation à travers les 36 États du Nigeria + FCT.",
                "38 canaux: un par État, un pour le FCT (Abuja), et un canal de commandement national pour la coordination inter-États.",
                "Seuls les Gardes vérifiés (rôle Vanguard), le personnel NIHSA et les fonctionnaires gouvernementaux peuvent poster. Les citoyens peuvent voir tous les messages.",
                "Pour devenir Garde: inscrivez-vous, cochez 'Je suis Garde des Inondations', et attendez l'approbation NIHSA. Les utilisateurs approuvés peuvent poster dans leur canal d'État.",
                "Les messages se synchronisent en temps réel via WebSocket. Si la connexion est perdue, les messages se rechargent automatiquement à la reconnexion.",
                "Même sans se connecter, vous pouvez voir tous les messages et suivre les mises à jour de situation en direct des Gardes sur le terrain.",
            ]
        },
    },

    "dashboard": {
        "en": {
            "title": "Dashboard Guide",
            "steps": [
                "Toggle between Annual (AFO 2026) and Weekly views at the top of the Dashboard tab.",
                "Annual view shows the full 2026 Flood Outlook — exposure data across 17 layers for all of Nigeria.",
                "Weekly view shows the current 7-day forecast — data is uploaded by NIHSA Admin when available.",
                "The exposure cards show totals for communities, people, health centres, schools, farmland (ha), roads (km), electricity, and markets at flood risk.",
                "Use the state dropdown to filter data by state and see state-specific exposure figures and map links.",
                "Tap 'Flood Animation' or 'Flood Extent Map' to open interactive NFFS model output maps for that state or nationally.",
            ]
        },
        "ha": {
            "title": "Jagoran Allon Bayanai",
            "steps": [
                "Canza tsakanin ra'ayi na Shekara (AFO 2026) da na Mako a sama na tab na Allon Bayanai.",
                "Ra'ayi na Shekara yana nuna cikakken Hasashen Ambaliya 2026 — bayanan fallasa a cikin yadudduka 17 ga dukkan Najeriya.",
                "Ra'ayi na Mako yana nuna hasashen kwanaki 7 na yanzu — ana loda bayanan ta masu kulawa na NIHSA idan ana da su.",
                "Katunan fallasa suna nuna jimlar al'umma, mutane, cibiyoyin lafiya, makarantu, gonaki (ha), hanyoyi (km), wutar lantarki, da kasuwanni cikin haɗarin ambaliya.",
                "Yi amfani da zaɓi na jiha don tace bayanan ta jiha da ganin adadin fallasa na jiha da haɗin taswira.",
                "Danna 'Motsin Ambaliya' ko 'Taswira ta Fadawar Ambaliya' don buɗe taswira mai aiki na fitarwar samfurin NFFS.",
            ]
        },
        "yo": {
            "title": "Itọsọna Paali Alaye",
            "steps": [
                "Yipada laarin awọn iwoye Lọdọọdún (AFO 2026) ati Ọsẹ ni oke tab Paali Alaye.",
                "Iwoye Lọdọọdún fihan Asọtẹlẹ Iṣan-omi 2026 ni kikun — data ifihan kọja awọn fẹlẹfẹlẹ 17 fun Naijiria gbogbo.",
                "Iwoye Ọsẹ fihan asọtẹlẹ ọjọ 7 lọwọlọwọ — data gbesoke nipasẹ NIHSA Admin nigba ti o wa.",
                "Awọn kaadi ifihan fihan awọn apapọ fun awọn agbegbe, eniyan, awọn ile itọju ilera, awọn ile-iwe, ilẹ oko (ha), awọn ọna (km), ina mọnamọna, ati awọn ọja ninu ewu iṣan-omi.",
                "Lo akojọ silẹ ipinlẹ lati ṣàlẹmọ data nipasẹ ipinlẹ ki o wo awọn nọmba ifihan ipinlẹ kan pato ati awọn ọna asopọ maapu.",
                "Tẹ 'Agbeka Iṣan-omi' tabi 'Maapu Iye Iṣan-omi' lati ṣii awọn maapu iṣẹ-awoṣe NFFS fun ipinlẹ yẹn tabi ti orilẹ-ede.",
            ]
        },
        "ig": {
            "title": "Nduzi Penu Ozi",
            "steps": [
                "Gbanwee n'etiti ọhụụ Ọdụn (AFO 2026) na Izu n'elu tab Penu Ozi.",
                "Ọhụụ Ọdụn na-egosi Atụmatụ Mmiri Ozuzo 2026 zuru oke — data mficha n'elu ọkwa 17 maka Naịjirịa niile.",
                "Ọhụụ Izu na-egosi atụmatụ ụbọchị 7 ugbu a — a na-ebutere data site na Admin NIHSA mgbe ọ dị.",
                "Kaadị mficha na-egosi ngụkọ maka obodo, ndị mmadụ, ụlọ ọgwụ, ụlọ akwụkwọ, ala ugbo (ha), okporo ụzọ (km), ọkụ eletrik, na ahia n'ihe ize ndụ mmiri ozuzo.",
                "Jiri dropụdaụn steeti iji lelee data site n'steeti hụ ọnụọgụ mficha steeti ya na njikọ maapu.",
                "Kụọ 'Ngagharị Mmiri Ozuzo' ma ọ bụ 'Maapu Ọdịdị Mmiri Ozuzo' iji mepee maapu mmepụta ihe atụmatụ NFFS.",
            ]
        },
        "fr": {
            "title": "Guide du Tableau de Bord",
            "steps": [
                "Basculez entre les vues Annuelle (AFO 2026) et Hebdomadaire en haut de l'onglet Tableau de Bord.",
                "La vue Annuelle montre les Perspectives d'Inondation 2026 complètes — données d'exposition sur 17 couches pour tout le Nigeria.",
                "La vue Hebdomadaire montre les prévisions des 7 prochains jours — données mises à jour par l'Admin NIHSA lorsque disponibles.",
                "Les cartes d'exposition montrent les totaux pour les communautés, personnes, centres de santé, écoles, terres agricoles (ha), routes (km), électricité et marchés à risque.",
                "Utilisez la liste déroulante d'État pour filtrer par État et voir les chiffres d'exposition spécifiques et les liens de carte.",
                "Appuyez sur 'Animation d'Inondation' ou 'Carte d'Étendue' pour ouvrir les cartes de sortie du modèle NFFS interactives.",
            ]
        },
    },

    "assistant": {
        "en": {
            "title": "Using the AI Assistant",
            "steps": [
                "NIHSA FloodAI is trained specifically for Nigerian flood safety and hydrology. Do NOT ask it general questions — use it only for flood, water, and emergency topics.",
                "Type your question in the chat box and tap Ask, or tap 🎤 to speak — your voice is transcribed by Whisper AI and sent automatically.",
                "The AI detects your language automatically and responds in Hausa, Yoruba, Igbo, French, or English.",
                "You can ask about: flood risk for your area, what a gauge reading means, evacuation procedures, flood safety tips, NFFS model data, river levels, AFO 2026 forecast.",
                "Daily limits: Citizens = 5 prompts/day, Vanguard = 10, Researchers/Government = 20, NIHSA Staff = 50. Sign in to use prompts.",
                "The AI can trigger actions: open the flood report form, navigate to a tab, or show a tutorial — just ask naturally.",
            ]
        },
        "ha": {
            "title": "Amfani da Mataimaki na AI",
            "steps": [
                "NIHSA FloodAI an horar da shi musamman don amincin ambaliya da ilimin ruwa na Najeriya. KADA ka yi tambayoyin gaba ɗaya — yi amfani da shi kawai don batu na ambaliya, ruwa, da gaggawa.",
                "Rubuta tambayarka a cikin akwatin tattaunawa kuma danna Tambayi, ko danna 🎤 don magana — ana fassara muryarku ta Whisper AI kuma ana aika kai tsaye.",
                "AI yana gano harsheka kai tsaye kuma yana amsa da Hausa, Yoruba, Igbo, Faransanci, ko Turanci.",
                "Kuna iya tambaya game da: haɗarin ambaliya a yankinku, abin da karantawar aunawa ke nufi, hanyoyin kwasawa, shawarwarin amincin ambaliya, bayanai na samfurin NFFS, matakan kogi, hasashe na AFO 2026.",
                "Iyakar yau da kullun: Ɗan ƙasa = tambayoyi 5/rana, Vanguard = 10, Masu bincike/Gwamnati = 20, Ma'aikatan NIHSA = 50. Shiga don amfani da tambayoyi.",
                "AI na iya kunna ayyuka: buɗe fom na rahoton ambaliya, tafiya zuwa tab, ko nuna tutorial — kawai tambayi da dabi'a.",
            ]
        },
        "yo": {
            "title": "Lilo Oluranlowo AI",
            "steps": [
                "NIHSA FloodAI jẹ ikẹkọ ni pataki fun aabo iṣan-omi ati imọ-omi Naijiria. MAṢE beere awọn ibeere gbogbogbo — lo nikan fun awọn koko iṣan-omi, omi, ati pajawiri.",
                "Tẹ ibeere rẹ sinu apoti ibaraẹnisọrọ ki o tẹ Beere, tabi tẹ 🎤 lati sọrọ — ohun rẹ jẹ tumọ nipasẹ Whisper AI ti a si firanṣẹ laifọwọyi.",
                "AI ṣawari ede rẹ laifọwọyi ati dahun ni Hausa, Yoruba, Igbo, Faranse, tabi Gẹẹsi.",
                "O le beere nipa: eewu iṣan-omi fun agbegbe rẹ, ohun ti kika gauge tumọ si, awọn ilana iṣapá, awọn imọran aabo iṣan-omi, data awoṣe NFFS, awọn ipele odò, asọtẹlẹ AFO 2026.",
                "Awọn opin ojoojumọ: Ara ilu = awọn ibeere 5/ọjọ, Vanguard = 10, Awọn oniwadi/Ijọba = 20, Oṣiṣẹ NIHSA = 50. Wọle lati lo awọn ibeere.",
                "AI le mu awọn iṣe ṣẹ: ṣii fọọmu ijabọ iṣan-omi, lọ si tab, tabi fihan itọnisọna — beere nipa ti ara.",
            ]
        },
        "ig": {
            "title": "Iji Onye Enyemaka AI",
            "steps": [
                "E zigara NIHSA FloodAI ọzụzụ kpọmkwem maka nchekwa mmiri ozuzo na mmụta mmiri Naịjirịa. ACHỌGHỊ ịjụ ajụjụ izugbe — jiri ya naanị maka ihe mmiri ozuzo, mmiri, na ihe mberede.",
                "Dee ajụjụ gị n'ime igbe mkparịta ụka wee kụọ Jụọ, ma ọ bụ kụọ 🎤 iji kwuo — a na-atụgharịa olu gị site na Whisper AI ma zigaa ozugbo.",
                "AI na-achọpụta asụsụ gị ozugbo ma zaghachi n'Hausa, Yoruba, Igbo, Faransị, ma ọ bụ Bekee.",
                "Ị nwere ike ịjụ maka: ihe ize ndụ mmiri ozuzo maka mpaghara gị, ihe ọgụgụ gauge pụtara, usoro nnarị, ndụmọdụ nchekwa mmiri ozuzo, data ihe atụmatụ NFFS, ọkwa osimiri, atụmatụ AFO 2026.",
                "Oke ụbọchị: Ndị ọchịchọ = ajụjụ 5/ụbọchị, Vanguard = 10, Ndị nyocha/Gọọmentị = 20, Ndị ọrụ NIHSA = 50. Banye iji jiri ajụjụ.",
                "AI nwere ike ibute omume: mepee ụdị akụkọ mmiri ozuzo, gaa tab, ma ọ bụ gosipụta nkuzi — jụọ n'ụzọ dị mfe.",
            ]
        },
        "fr": {
            "title": "Utiliser l'Assistant IA",
            "steps": [
                "NIHSA FloodAI est formé spécifiquement pour la sécurité des inondations et l'hydrologie nigériane. Ne posez PAS de questions générales — utilisez-le uniquement pour les sujets d'inondation, d'eau et d'urgence.",
                "Tapez votre question dans la boîte de chat et appuyez sur Demander, ou appuyez sur 🎤 pour parler — votre voix est transcrite par Whisper AI et envoyée automatiquement.",
                "L'IA détecte votre langue automatiquement et répond en Haoussa, Yoruba, Igbo, Français ou Anglais.",
                "Vous pouvez demander: le risque d'inondation pour votre zone, ce que signifie une lecture de jauge, les procédures d'évacuation, les conseils de sécurité, les données NFFS, les niveaux des rivières, les prévisions AFO 2026.",
                "Limites quotidiennes: Citoyens = 5 requêtes/jour, Vanguard = 10, Chercheurs/Gouvernement = 20, Personnel NIHSA = 50. Connectez-vous pour utiliser les requêtes.",
                "L'IA peut déclencher des actions: ouvrir le formulaire de rapport, naviguer vers un onglet, ou afficher un tutoriel — demandez naturellement.",
            ]
        },
    },
}

# ============================================================================
# FUNCTION CALLING TOOLS
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to_report",
            "description": "User wants to report a flood. Open the flood reporting form.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prefill_location": {"type": "string", "description": "Location to pre-fill"},
                    "prefill_description": {"type": "string", "description": "Description to pre-fill"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to_tab",
            "description": "User wants to navigate to a different tab in the app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab": {
                        "type": "string",
                        "enum": ["map", "dashboard", "vanguard", "assistant", "alerts"],
                        "description": "The tab to navigate to"
                    }
                },
                "required": ["tab"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_tutorial",
            "description": "User needs help understanding how to use a feature of the app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["reporting", "alerts", "map", "vanguard", "dashboard", "general", "assistant"],
                        "description": "The tutorial topic"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_location",
            "description": "User wants to find a specific location on the map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Location search query, The exact location name only. Examples: 'Lokoja', 'Makurdi', 'Kogi State', 'Benue River', 'Lagos'. NOT full sentences. "}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_flood_status",
            "description": "User asks about current flood conditions for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location to check"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "User requests human assistance or wants to talk to a real person.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"}
                },
                "required": []
            }
        }
    }
]

# ============================================================================
# ROLE LABELS
# ============================================================================

ROLE_LABELS = {
    "citizen":     "Citizen (5 prompts/day)",
    "vanguard":    "Flood Marshal — Vanguard (10 prompts/day)",
    "researcher":  "Researcher (20 prompts/day)",
    "government":  "Government Official (20 prompts/day)",
    "nihsa_staff": "NIHSA Staff (50 prompts/day)",
    "sub_admin":   "NIHSA Sub-Admin (50 prompts/day)",
    "admin":       "NIHSA Administrator (unlimited)",
}

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NIHSA AI Wrapper starting — Cloudflare STT + MeloTTS")
    await init_db_pool()
    await _cleanup_old_usage()
    yield
    logger.info("NIHSA AI Wrapper shutting down")
    await close_db_pool()

app = FastAPI(
    title="NIHSA AI Assistant Wrapper",
    description="Flood intelligence AI with Cloudflare STT/TTS — Nigeria Hydrological Services Agency",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "stt_provider": "Cloudflare Workers AI — whisper-large-v3-turbo",
        "tts_provider": "Cloudflare Workers AI — MeloTTS (no Google credentials needed)",
        "llm_provider": "DeepSeek Chat",
        "whisper_improvements": ["vad_filter", "hydrology_prompt", "beam_count=5", "no_hallucination_loop"],
    }


@app.post("/ai/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = Form(default="default"),
):
    """
    Transcribe audio to text using Cloudflare Whisper Large v3 Turbo.
    Improvements: hydrology context prompt, VAD, beam_count=5.
    """
    if not check_rate_limit(session_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait.")

    audio_data = await audio.read()
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_data) > 24 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 24MB). Please record a shorter clip.")

    try:
        text, detected_lang, confidence = await transcribe_audio_cloudflare(audio_data)
        if not text:
            raise HTTPException(status_code=400, detail="No speech detected. Please try again in a quiet environment.")
        return TranscribeResponse(text=text, detected_language=detected_lang, confidence=confidence)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed. Please use text input instead.")


@app.post("/ai/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Chat with DeepSeek using a rich NIHSA hydrology system prompt.
    Includes live flood context from the NIHSA backend.
    """
    auth_header = request.headers.get("Authorization", "")
    user_data = await verify_user_with_main_backend(auth_header)

    if not user_data:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in to use the AI assistant."
        )

    user_id = user_data.get("id", body.session_id)
    limit, role = get_user_quota(user_data)

    allowed, remaining, _ = await check_daily_quota(user_id, role)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {limit} prompts reached. Your quota resets tomorrow."
        )

    if not check_rate_limit(f"{user_id}_{body.session_id}", limit=10, window=60):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    # Detect language
    language = body.language
    if not language and body.messages:
        last_msg = body.messages[-1].content
        language = await detect_language_deepseek(last_msg)
        if language == "en":
            language = detect_language_keywords(last_msg)
    language = language or "en"

    # ── Build flood context from what the frontend sends ──────────────────────
    # The frontend already holds the real verified alerts fetched from the DB.
    # Using these is more reliable than the wrapper making a second API call,
    # and it means the AI sees exactly what the user sees on screen.
    if body.active_alerts is not None:
        if body.active_alerts:
            critical = [a for a in body.active_alerts if a.get("level") in ("CRITICAL", "HIGH", "EXTREME")]
            flood_context = (
                f"ACTIVE VERIFIED ALERTS (from NIHSA database, confirmed by coordinators): "
                f"{len(body.active_alerts)} total, {len(critical)} critical/high/extreme.\n"
            )
            for a in body.active_alerts[:8]:
                flood_context += f"  ⚠ [{a.get('level','')}] {a.get('title','')} — {a.get('state','')}"
                lgas = a.get("lgas") or []
                if lgas:
                    flood_context += f" (LGAs: {', '.join(lgas[:4])})"
                msg_snippet = (a.get("message") or "")[:120]
                if msg_snippet:
                    flood_context += f"\n    → {msg_snippet}"
                flood_context += "\n"
        else:
            flood_context = "ACTIVE ALERTS: None at this time. All published NIHSA alerts have been resolved."
    else:
        # Fallback: fetch from backend if frontend didn't send alerts
        flood_context = await fetch_flood_context()

    # ── Location context: personalise advice to the user's area ───────────────
    location_context = ""
    if body.user_location:
        loc = body.user_location
        lat = loc.get("lat")
        lng = loc.get("lng")
        address = loc.get("address", "")
        coords = f"{lat:.4f}°N, {lng:.4f}°E" if lat and lng else "unknown"

        # Determine which active alerts (if any) match this user's state
        nearby_alerts = []
        if body.active_alerts:
            user_state = address.split(",")[-1].strip().lower() if address else ""
            for a in body.active_alerts:
                alert_state = (a.get("state") or "").lower()
                if user_state and (user_state in alert_state or alert_state in user_state):
                    nearby_alerts.append(a)

        nearby_str = ""
        if nearby_alerts:
            nearby_str = f"\nALERTS IN USER'S STATE ({address.split(',')[-1].strip() if address else 'their area'}):\n"
            for a in nearby_alerts[:4]:
                nearby_str += f"  🔴 [{a.get('level','')}] {a.get('title','')} — LGAs: {', '.join((a.get('lgas') or [])[:4])}\n"
        else:
            nearby_str = f"\nNo active NIHSA alerts currently in {address.split(',')[-1].strip() if address else 'their state'}."

        location_context = (
            f"\n{'='*50}\n"
            f"USER'S CURRENT LOCATION (GPS-verified, do NOT ask them for location):\n"
            f"  Address: {address}\n"
            f"  Coordinates: {coords}\n"
            f"{nearby_str}\n"
            f"ABSOLUTE RULE: NEVER ask the user to provide, share, or confirm their location.\n"
            f"You already have it above. Use it immediately when answering any location-based question.\n"
            f"{'='*50}"
        )
    else:
        location_context = (
            "\nUSER LOCATION: Not yet available (GPS still loading or permission denied). "
            "If they ask about their area, ask them to name their state or LGA so you can check alerts for them."
        )

    system_prompt = get_system_prompt(language)
    role_label = ROLE_LABELS.get(role, "User")

    full_system = (
        f"{system_prompt}\n\n"
        f"USER CONTEXT:\n"
        f"  Role: {role_label}\n"
        f"  Prompts remaining today: {remaining}/{limit}\n"
        f"{location_context}\n\n"
        f"LIVE NIHSA SITUATION REPORT:\n{flood_context}"
    )

    messages = [{"role": "system", "content": full_system}]
    for msg in body.messages[-10:]:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=600,
            temperature=0.5,  # Lower = more factual, less hallucination
        )

        message = response.choices[0].message
        await increment_usage(user_id, role)

        action = None
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            action = {
                "type": tool_call.function.name,
                "params": json.loads(tool_call.function.arguments)
            }
            # Generate a real, helpful reply alongside the action.
            # We ask DeepSeek to produce the guidelines the user needs PLUS
            # acknowledge the action being taken — no more "Let me help with that."
            tool_name = tool_call.function.name
            tool_params = json.loads(tool_call.function.arguments)

            # Build a context-aware follow-up prompt
            action_context = {
                "navigate_to_report": (
                    "The user wants to report flooding. "
                    "Tell them you are opening the report form for them. "
                    "Also give them 3–4 immediate safety guidelines for their current situation "
                    "based on their location and any active alerts."
                ),
                "escalate_to_human": (
                    "The user needs human assistance. "
                    "Tell them you are opening the report form so NIHSA coordinators can see their situation. "
                    "Critically, give them the most important flood safety guidelines RIGHT NOW while they wait — "
                    "what to do, what NOT to do, where to go. Be specific to their location and any active alerts."
                ),
                "get_flood_status": (
                    f"The user is asking about flood conditions at: {tool_params.get('location', 'their area')}. "
                    "Using the active alerts and location data in your context, give them a direct, specific answer. "
                    "State the current risk level, any active alerts for that area, and what actions they should take."
                ),
                "show_tutorial": (
                    f"You are opening the {tool_params.get('topic', 'general')} tutorial. "
                    "Give a one-sentence description of what they will learn, then open it."
                ),
                "navigate_to_tab": (
                    f"You are navigating to the {tool_params.get('tab', '')} tab. "
                    "Give one sentence explaining what they will find there."
                ),
                "search_location": (
                    f"You are searching the map for: {tool_params.get('query', '')}. "
                    "Tell them you are showing it on the map and give any relevant flood risk info for that area."
                ),
            }.get(tool_name, "Give the user helpful flood safety guidance relevant to their request and location.")

            try:
                follow_up = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": full_system},
                        *[{"role": m.role, "content": m.content} for m in body.messages[-6:]],
                        {"role": "system", "content": (
                            f"INSTRUCTION FOR THIS REPLY ONLY: {action_context} "
                            "Keep your response under 120 words. Be direct and actionable. "
                            "Do NOT say 'Let me help' or 'I will help'. Start with the actual content."
                        )},
                    ],
                    max_tokens=200,
                    temperature=0.4,
                )
                reply = follow_up.choices[0].message.content or "Opening that for you now."
            except Exception:
                # Fallback per tool if the second call fails
                reply = {
                    "navigate_to_report": "🚨 Opening the flood report form. Please attach a photo, voice note, or video of the flooding so NIHSA coordinators can verify and respond.",
                    "escalate_to_human": "🆘 Opening the report form for NIHSA review. While you wait: move to higher ground, do not cross flooded water, call 112 for life-threatening emergencies.",
                    "get_flood_status": f"Checking flood conditions for {tool_params.get('location', 'your area')} — see the map for live gauge readings.",
                    "show_tutorial": f"Opening the {tool_params.get('topic', '')} guide for you.",
                    "navigate_to_tab": f"Taking you to the {tool_params.get('tab', '')} tab.",
                    "search_location": f"Showing {tool_params.get('query', '')} on the map.",
                }.get(tool_name, "Done — please check the screen for the result.")
        else:
            reply = message.content or "I'm here to help with flood safety. Please ask a hydrology-related question."

        return ChatResponse(reply=reply, action=action, detected_language=language)

    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable. Please try again.")


@app.post("/ai/speak")
async def speak_endpoint(
    text: str = Form(...),
    language: str = Form(default="en"),
    session_id: str = Form(default="default"),
):
    """
    Convert text to speech using Cloudflare MeloTTS.
    No Google credentials required — uses your existing Cloudflare API token.
    Supported natively: en, fr. ha/yo/ig fall back to English voice.
    Returns MP3 audio.
    """
    if not check_rate_limit(session_id, limit=30):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    if len(text) > 1000:
        text = text[:1000]

    try:
        audio_bytes = await synthesize_speech(text, language)
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline"},
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=503, detail="Text-to-speech temporarily unavailable.")

@app.get("/ai/quota/fast")
async def get_quota_fast(req: Request):
    """
    Fast quota check — reads quota directly from DB without verifying the token
    against the main backend. Token is decoded locally. Falls back to defaults
    if the DB is unreachable. Used for initial page load to avoid the double
    network hop cold start penalty.
    """
    from jose import jwt, JWTError
    SECRET_KEY = os.environ.get("SECRET_KEY", "")
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {"remaining": 0, "limit": 0, "authenticated": False, "role": None}
    token = auth_header[7:]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub") or payload.get("user_id")
        role = (payload.get("role") or "citizen").lower()
    except JWTError:
        return {"remaining": 0, "limit": 0, "authenticated": False, "role": None}
    
    limit = QUOTA_LIMITS.get(role, 5)
    allowed, remaining, _ = await check_daily_quota(user_id, role)
    return {
        "remaining": remaining if allowed else 0,
        "limit": limit,
        "authenticated": True,
        "role": role,
        "reset_at": (datetime.now().date() + timedelta(days=1)).isoformat()
    }
    
@app.get("/ai/quota")
async def get_quota(req: Request):
    """Get remaining daily prompts for authenticated user."""
    auth_header = req.headers.get("Authorization", "")
    user_data = await verify_user_with_main_backend(auth_header)

    if not user_data:
        return {
            "remaining": 0, "limit": 0,
            "authenticated": False, "role": None,
            "reset_at": (datetime.now().date() + timedelta(days=1)).isoformat()
        }

    user_id = user_data.get("id", "unknown")
    role = user_data.get("role", "citizen").lower()
    limit = QUOTA_LIMITS.get(role, 5)
    allowed, remaining, _ = await check_daily_quota(user_id, role)

    return {
        "remaining": remaining if allowed else 0,
        "limit": limit,
        "authenticated": True,
        "role": role,
        "reset_at": (datetime.now().date() + timedelta(days=1)).isoformat()
    }


@app.get("/ai/tutorials/{topic}", response_model=TutorialResponse)
async def get_tutorial(topic: str, lang: str = "en"):
    """
    Get tutorial content for a topic in the specified language.
    All content is accurate to the real NIHSA app and available in all 5 languages.
    """
    if topic not in TUTORIALS:
        raise HTTPException(status_code=404, detail=f"Tutorial topic '{topic}' not found.")

    topic_data = TUTORIALS[topic]
    # Fall back to English if requested language not available
    if lang not in topic_data:
        lang = "en"

    content = topic_data[lang]
    return TutorialResponse(
        title=content["title"],
        steps=content["steps"],
        language=lang,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
