#!/usr/bin/env python3

import os
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

from duckduckgo_search import DDGS

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


# Worker proxy URL for STT (Cloudflare Worker)
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
    user_location: Optional[Dict[str, Any]] = None
    active_alerts: Optional[List[Dict[str, Any]]] = None

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

_rate_store: Dict[str, List[float]] = defaultdict(list)
_quota_cache: TTLCache = TTLCache(maxsize=1000, ttl=30)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_db_pool = None

async def init_db_pool():
    global _db_pool
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — quota will fall back to in-memory")
        return
    try:
        import asyncpg
        _db_pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=1, max_size=5, command_timeout=10,
        )
        async with _db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_usage (
                    user_id    TEXT NOT NULL,
                    usage_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    count      INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, usage_date)
                )
            """)
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

_mem_usage: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "date": None})

def get_user_quota(user_data: dict) -> Tuple[int, str]:
    role = (user_data.get("role") or "citizen").lower()
    return QUOTA_LIMITS.get(role, 5), role

async def check_daily_quota(user_id: str, role: str) -> Tuple[bool, int, int]:
    limit = QUOTA_LIMITS.get(role, 5)
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
    today = datetime.now().date().isoformat()
    store = _mem_usage[user_id]
    if store["date"] != today:
        store["count"] = 0
        store["date"] = today
    remaining = max(0, limit - store["count"])
    return remaining > 0, remaining, limit

async def increment_usage(user_id: str, role: str):
    _quota_cache.pop(f"quota:{user_id}", None)
    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_usage (user_id, usage_date, count)
                    VALUES ($1, CURRENT_DATE, 1)
                    ON CONFLICT (user_id, usage_date)
                    DO UPDATE SET count = ai_usage.count + 1
                """, user_id)
            return
        except Exception as e:
            logger.warning(f"DB usage increment failed for {user_id}: {e} — using memory fallback")
    today = datetime.now().date().isoformat()
    store = _mem_usage[user_id]
    if store["date"] != today:
        store["count"] = 0
        store["date"] = today
    store["count"] += 1

def check_rate_limit(key: str, limit: int = 20, window: int = 60) -> bool:
    import time
    now = time.time()
    times = _rate_store[key]
    _rate_store[key] = [t for t in times if now - t < window]
    if len(_rate_store[key]) >= limit:
        return False
    _rate_store[key].append(now)
    return True

# ============================================================================
# USER VERIFICATION
# ============================================================================

_user_cache: TTLCache = TTLCache(maxsize=500, ttl=60)

async def verify_user_with_main_backend(auth_header: str) -> Optional[dict]:
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    cache_key = hashlib.md5(token.encode()).hexdigest()
    if cache_key in _user_cache:
        return _user_cache[cache_key]
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
                    return None
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
# WEB SEARCH — Hydrology and emergency contacts, scoped to Nigeria
# ============================================================================

# Cache search results for 30 minutes to avoid burning API quota on identical queries
_search_cache: TTLCache = TTLCache(maxsize=200, ttl=1800)

async def execute_web_search(query: str) -> str:
    """
    Execute a web search scoped to hydrology and emergency topics.
    Uses free DuckDuckGo search — no API key required.
    Results are cached for 30 minutes.
    """
    cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    if cache_key in _search_cache:
        logger.info(f"Search cache hit: '{query[:60]}'")
        return _search_cache[cache_key]

    # Add Nigeria context and news focus
    enhanced_query = query
    if "nigeria" not in query.lower():
        enhanced_query = f"{query} Nigeria"
    
    # Add "news" if it's about current events
    if any(word in query.lower() for word in ["today", "current", "now", "recent", "happening"]):
        enhanced_query = f"{enhanced_query} news"

    try:
        loop = asyncio.get_event_loop()
        
        def do_search():
            results = []
            with DDGS() as ddgs:
                # Try news search first for current events
                if "news" in enhanced_query.lower() or "today" in enhanced_query.lower():
                    for r in ddgs.news(
                        enhanced_query.replace(" news", ""),
                        region="ng",
                        safesearch="moderate",
                        max_results=5,
                        timelimit="w"  # Past week for news
                    ):
                        results.append({
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "href": r.get("url", ""),
                            "source": r.get("source", "News source")
                        })
                
                # Fall back to general search if news yields nothing
                if not results:
                    for r in ddgs.text(
                        enhanced_query, 
                        region="ng", 
                        safesearch="moderate", 
                        max_results=5,
                        timelimit="m"  # Past month
                    ):
                        results.append({
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "href": r.get("href", ""),
                        })
            return results

        search_results = await loop.run_in_executor(None, do_search)

        if not search_results:
            return "No recent web results found for that query. You may want to check official NIHSA channels or NEMA social media for real-time updates."

        lines = []
        for r in search_results[:4]:
            title = r.get("title", "")
            snippet = r.get("body", "")[:300]  # Limit snippet length
            link = r.get("href", "")
            source = r.get("source", "")
            
            line = f"• {title}"
            if source:
                line += f" ({source})"
            line += f"\n  {snippet}..."
            if link:
                line += f"\n  🔗 {link}"
            lines.append(line)

        formatted = "\n\n".join(lines)
        _search_cache[cache_key] = formatted
        logger.info(f"Web search done: '{enhanced_query[:60]}' — {len(search_results)} results")
        return formatted

    except Exception as e:
        logger.warning(f"Web search error for '{query}': {e}")
        return "Web search temporarily unavailable. Please check NIHSA official channels for the latest updates."

async def fetch_emergency_contacts(state: str = "") -> str:
    """
    Fetch current Nigerian emergency contacts, optionally state-specific.
    Provides known baseline numbers plus live search results.
    """
    state_str = f"{state} " if state else ""
    query = f"Nigeria {state_str}emergency contact numbers NEMA SEMA flood 2025 official hotline"
    search_results = await execute_web_search(query)

    baseline = (
        "VERIFIED NIGERIAN EMERGENCY NUMBERS (always valid):\n"
        "• 112 — General emergency (free, works on all networks)\n"
        "• 199 — Nigeria Fire Service\n"
        "• 123 — Nigeria Police Force\n"
        "• 0800-CALL-NEMA (0800-2255-6362) — NEMA national flood hotline\n"
        "• NIHSA Emergency: info@nihsa.gov.ng\n"
        "• SMS: Text your location to 20543 (NEMA short code)\n"
    )

    if state:
        baseline += f"\nSEARCHING FOR {state.upper()} SEMA AND LOCAL EMERGENCY CONTACTS:\n"
    else:
        baseline += "\nADDITIONAL CONTACTS FROM WEB SEARCH:\n"

    return baseline + search_results

# ============================================================================
# FLOOD CONTEXT FROM NIHSA BACKEND (fallback when frontend doesn't send alerts)
# ============================================================================

async def fetch_flood_context() -> str:
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
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT_CORE = """You are NIHSA FloodAI — the official AI assistant of Nigeria's National Hydrological Services Agency.

YOUR SOLE PURPOSE is flood safety, hydrology, and emergency response for Nigeria. Only answer questions related to:
- Flood risk assessment and river gauge interpretation
- Evacuation guidance and emergency procedures
- NFFS (National Flood Forecasting System) data explanation
- Reporting flooding (direct users to the 🚨 Report Flood button)
- Water depth safety guidance
- Basin, river and watershed information for Nigeria
- Historical and current flood events in Nigeria
- Climate and seasonal flood outlook (AFO 2026)
- Current hydrological news, dam releases, river levels — searched live from the internet when needed
- Emergency contacts and response numbers — looked up live to ensure they are always current

NON-FLOOD EMERGENCIES:
If a user reports a fire, accident, medical emergency, robbery, landslide, or any non-flood crisis:
- Give immediate safety advice relevant to that emergency
- Use get_emergency_contacts to provide current, verified phone numbers for their state
- Tell them you are logging their location with NIHSA coordinators who can escalate
- Ask them to tap the 🚨 Report Flood button and record a short VIDEO of their situation
- Do NOT refuse to help just because it is not a flood — human safety comes first

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTION — LOCATION:
The user's GPS location is already known to you. It is in USER'S CURRENT LOCATION below.
NEVER ask the user to share, send, or provide their location. You already have it.
Use it immediately when answering any location-based question. Cross-reference their state
and LGA against the active alerts list. Give a specific answer based on their actual location.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTION — ALWAYS GIVE GUIDELINES:
NEVER respond with only "Let me help you with that" or similar non-answers.
Even when triggering an app action, you MUST also provide:
  1. Immediate safety advice relevant to the situation
  2. Specific actionable steps the person should take RIGHT NOW
  3. What to do while waiting for help or NIHSA response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEB SEARCH AND EMERGENCY CONTACTS:
You have two research tools — use them proactively:

web_search_hydrology — use when:
• User asks about a recent flood event or current river conditions
• User asks about Lagdo Dam or any dam release announcement
• User asks about current rainfall forecasts or NEMA/NIHSA bulletins
• User references a specific recent news event about flooding in Nigeria
• You need to verify current hydrological data not in your training

get_emergency_contacts — use when:
• User asks for ANY emergency phone number, hotline, or contact
• User reports an emergency (fire, flood, accident, medical)
• User asks about NEMA, SEMA, police, fire service, or ambulance contacts
• User asks who to call in their state during a flood
Always use get_emergency_contacts for phone numbers — never rely on training data alone
since numbers change and state-specific contacts vary.

DO NOT use these tools for general flood safety advice, basic hydrology concepts, or app usage questions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NIGERIA FLOOD CONTEXT:
- Nigeria has 70 major river basins monitored by NIHSA
- Key rivers: Niger, Benue, Kaduna, Sokoto, Hadejia, Anambra, Cross River, Ogun
- Flood seasons: April–June (early rains), July–September (peak), October–November (recession)
- Lagdo Dam (Cameroon) on the Benue causes downstream flooding 2–4 days after release
- NFFS = National Flood Forecasting System — Nigeria's LSTM deep learning model
- AFO 2026 = Annual Flood Outlook 2026 — national seasonal risk document
- 358 river gauge stations monitored nationwide

RISK LEVELS AND REQUIRED GUIDELINES:
- NORMAL: Safe. Tell user no action needed but stay informed.
- WATCH: Prepare emergency kit, know evacuation route, move valuables, avoid riverbanks.
- HIGH: Move valuables to upper floors NOW, prepare to evacuate, alert neighbours.
- CRITICAL/EXTREME: EVACUATE NOW. Do not cross flooded water (15cm knocks you over, 30cm moves a car). Call 112.

FLOOD SAFETY GUIDELINES:
- Do NOT cross flooded roads — most flood deaths happen this way
- Do NOT touch electrical appliances or wires in water — electrocution risk
- Move to highest ground — upper floors, hills, elevated areas
- Store clean water and food for 3 days
- Keep documents (ID, insurance) in a waterproof bag
- After flooding: do not eat food touched by floodwater, boil drinking water

RESPONSE STYLE:
- ALWAYS be direct and actionable — never vague
- Lead with the most urgent safety action
- Use simple language accessible to citizens with varying literacy
- Never provide medical advice — refer to emergency services for injuries
- Support Hausa, Yoruba, Igbo, and French — detect and respond in the user's language

APP FEATURES:
- 🗺️ Map tab: Live gauge stations, alerts, citizen reports
- 📊 Dashboard: AFO 2026 exposure data
- 🦺 Vanguard: Flood Marshals coordination network
- 🔔 Alerts tab: All active flood warnings by state
- 🚨 Report Flood button: Submit photo/voice/video of flooding
- Language selector: English, Hausa, Yoruba, Igbo, French
"""

def get_system_prompt(language: str = "en") -> str:
    lang_instructions = {
        "ha": "\n\nINSTRUCTION: The user is communicating in Hausa. Respond in Hausa. Use 'ku' for formal address.",
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

# ============================================================================
# STT — Cloudflare Whisper Large v3 Turbo via Worker proxy
# ============================================================================

WHISPER_HYDROLOGY_PROMPT = (
    "NIHSA flood report. Nigeria hydrology. "
    "Rivers: Niger, Benue, Kaduna, Ogun, Anambra, Sokoto, Cross River. "
    "Terms: flood, ambaliya, iṣan-omi, mmiri ozuzo, inondation, "
    "gauge, water level, evacuation, alert, NIHSA, NFFS, basin, "
    "Lokoja, Makurdi, Onitsha, Kano, Lagos, Abuja, Ibadan, Maiduguri."
)

async def transcribe_audio_cloudflare(audio_data: bytes) -> Tuple[str, str, Optional[float]]:
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
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code != 200:
            logger.error(f"Worker STT {resp.status_code}: {resp.text[:200]}")
            raise HTTPException(
                status_code=502,
                detail=f"Speech recognition error ({resp.status_code}). Please try again."
            )
        data = resp.json()
        if not data.get("success", True):
            raise HTTPException(status_code=502, detail="Transcription failed on the AI side.")
        text = (data.get("text") or "").strip()
        detected_lang = data.get("detected_language") or data.get("language") or "en"
        confidence: Optional[float] = data.get("confidence")
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
# TTS — Cloudflare MeloTTS
# ============================================================================

MELOTTS_LANG_MAP = {
    "en": "en", "fr": "fr",
    "ha": "en",  # MeloTTS doesn't support Hausa — falls back to English voice
    "yo": "en",  # MeloTTS doesn't support Yoruba
    "ig": "en",  # MeloTTS doesn't support Igbo
    "es": "es", "zh": "zh",
}

_tts_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)

async def synthesize_speech(text: str, language: str = "en") -> bytes:
    text = text[:1000]
    cache_key = hashlib.md5(f"{text}:{language}".encode()).hexdigest()
    if cache_key in _tts_cache:
        return _tts_cache[cache_key]
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
                json={"text": text, "lang": melotts_lang},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=503, detail="Text-to-speech unavailable.")
            content_type = resp.headers.get("content-type", "")
            if "audio" in content_type or "octet" in content_type:
                audio_bytes = resp.content
            else:
                try:
                    d = resp.json()
                    audio_b64 = (
                        d.get("result", {}).get("audio") or
                        d.get("audio") or
                        d.get("result", "")
                    )
                    audio_bytes = base64.b64decode(audio_b64) if isinstance(audio_b64, str) else resp.content
                except Exception:
                    audio_bytes = resp.content
        if not audio_bytes:
            raise HTTPException(status_code=503, detail="Empty TTS response.")
        cache_file.write_bytes(audio_bytes)
        _tts_cache[cache_key] = audio_bytes
        return audio_bytes
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MeloTTS synthesis error: {e}")
        raise HTTPException(status_code=503, detail="Text-to-speech temporarily unavailable.")

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
                    "prefill_location":    {"type": "string", "description": "Location to pre-fill"},
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
                    "query": {
                        "type": "string",
                        "description": "The exact location name only. Examples: 'Lokoja', 'Makurdi', 'Kogi State', 'Benue River'. NOT full sentences."
                    }
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
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_hydrology",
            "description": (
                "Search the internet for current hydrological information. "
                "Use ONLY for: current river levels, recent flood events in Nigeria, "
                "dam release announcements (especially Lagdo Dam), NIHSA official bulletins, "
                "rainfall forecasts, historical flood data, water quality reports, "
                "NEMA/SEMA situation updates. "
                "Do NOT use for general safety advice, basic hydrology concepts, or app usage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific hydrology-focused query. "
                            "Examples: 'Lagdo Dam release April 2026', "
                            "'Benue River flood level current', "
                            "'NIHSA flood alert Kogi 2026'. Keep it concise."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_emergency_contacts",
            "description": (
                "Get current, verified emergency contact numbers for Nigeria or a specific state. "
                "ALWAYS use this tool when: user asks for any emergency phone number, "
                "user reports an emergency (flood, fire, medical, accident), "
                "user asks about NEMA, SEMA, police, fire service, ambulance, or any response agency. "
                "Never rely on training data alone for phone numbers — they change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Nigerian state name for state-specific contacts (e.g. 'Kogi', 'Benue'). Leave empty for national contacts."
                    }
                },
                "required": []
            }
        }
    },
]

# ============================================================================
# ROLE LABELS
# ============================================================================

ROLE_LABELS = {
    "citizen":     "Citizen (7 prompts/day)",
    "vanguard":    "Flood Marshal — Vanguard (7 prompts/day)",
    "researcher":  "Researcher (7 prompts/day)",
    "government":  "Government Official (7 prompts/day)",
    "nihsa_staff": "NIHSA Staff (7 prompts/day)",
    "sub_admin":   "NIHSA Sub-Admin (7 prompts/day)",
    "admin":       "NIHSA Administrator (unlimited)",
}

# ============================================================================
# TUTORIAL CONTENT — All 5 languages
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
                "🗺️ Tab na Taswira: Duba tashar aunawa 358+ masu rai, faɗakarwar ambaliya, rahotannin ɗan ƙasa, da matakan hasashe na NFFS a duk Najeriya.",
                "📊 Tab na Allon Bayanai: Bincika Hasashen Ambaliya na Shekara 2026 (AFO 2026) a cikin yadudduka 17.",
                "🦺 Tab na Masu Kiyaye Ambaliya: Hanyar sadarwa mai tsaro ga Masu Kiyaye Ambaliya da ma'aikatan NIHSA. Tasoshi 38.",
                "🤖 Tab na Mataimaki na AI: Tambayi NIHSA FloodAI game da haɗarin ambaliya. Danna 🎤 don magana.",
                "🔔 Tab na Faɗakarwa: Duk faɗakarwar ambaliya masu aiki a duk faɗin ƙasa.",
                "🚨 Rahoton Ambaliya: Danna maɓallin jan don aika rahoto tare da hoto, murya, ko bidiyo.",
                "🌐 Harsunan: Zaɓi tsakanin Turanci, Hausa, Yoruba, Igbo, da Faransanci.",
            ]
        },
        "yo": {
            "title": "Kaabọ si NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Tab Maapu: Wo awọn ibudo wiwọn 358+ ti nṣiṣẹ, awọn ìkìlọ̀ iṣan-omi kọja Naijiria.",
                "📊 Tab Paali Alaye: Ṣawari AFO 2026 kọja awọn fẹlẹfẹlẹ 17.",
                "🦺 Tab Awọn Oluso Iṣan-omi: Nẹtiwọọki isọdọkan fun Awọn Oluso ti a fọwọsi. Ikanni 38.",
                "🤖 Tab Oluranlowo AI: Beere NIHSA FloodAI nipa eewu iṣan-omi. Tẹ 🎤 lati sọrọ.",
                "🔔 Tab Ifokanbalẹ: Gbogbo awọn ikilọ iṣan-omi ti nṣiṣẹ ni orilẹ-ede.",
                "🚨 Jabo Iṣan-omi: Tẹ bọtini pupa lati fi ijabọ silẹ pẹlu fọto, ohun, tabi fidio.",
                "🌐 Awọn Ede: Yipada laarin Gẹẹsi, Hausa, Yoruba, Igbo, ati Faranse.",
            ]
        },
        "ig": {
            "title": "Nnọọ na NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Tab Maapu: Lee ọdụ ngụkọ 358+ ndụ, ọkwa mmiri ozuzo na-arụ ọrụ n'elu Naịjirịa.",
                "📊 Tab Penu Ozi: Nyochaa AFO 2026 n'elu ọkwa 17.",
                "🦺 Tab Ndị Nlekota Mmiri Ozuzo: Netwọk nhazi maka Ndị Nlekota kwadoro. Ọwa 38.",
                "🤖 Tab Onye Enyemaka AI: Jụọ NIHSA FloodAI maka mmiri ozuzo. Kụọ 🎤 iji kwuo.",
                "🔔 Tab Ọkwa: Ọkwa mmiri ozuzo niile na-arụ ọrụ n'elu mba.",
                "🚨 Kọọ Mmiri Ozuzo: Kụọ bọtịn ọbara ọbara iji zipu akụkọ na foto, olu, ma ọ bụ vidiyo.",
                "🌐 Asụsụ: Gbanwee n'etiti Bekee, Hausa, Yoruba, Igbo, na Faransị.",
            ]
        },
        "fr": {
            "title": "Bienvenue dans NIHSA Flood Intelligence",
            "steps": [
                "🗺️ Onglet Carte: Visualisez 358+ stations de jaugeage en direct et les alertes d'inondation à travers le Nigeria.",
                "📊 Onglet Tableau de Bord: Explorez l'AFO 2026 sur 17 couches.",
                "🦺 Onglet Gardes des Inondations: Réseau de coordination pour les Gardes vérifiés. 38 canaux.",
                "🤖 Onglet Assistant IA: Interrogez NIHSA FloodAI sur les risques d'inondation. Appuyez sur 🎤 pour parler.",
                "🔔 Onglet Alertes: Toutes les alertes d'inondation actives à l'échelle nationale.",
                "🚨 Signaler une Inondation: Appuyez sur le bouton rouge pour soumettre un rapport.",
                "🌐 Langues: Basculez entre Anglais, Haoussa, Yoruba, Igbo et Français.",
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
                "You MUST attach at least one: 📷 Photo (tap to use camera), 🎤 Voice (record up to 60 seconds), or 🎥 Video.",
                "Tap Submit Flood Report. Your report goes to NIHSA coordinators for verification. Verified reports appear on the map.",
            ]
        },
        "ha": {
            "title": "Yadda Ake Rahoton Ambaliya",
            "steps": [
                "Danna maɓallin jan 🚨 don buɗe fom rahoton ambaliya.",
                "GPS ɗinku zai gano wurinku kai tsaye. Ja pin don daidaita wurin.",
                "Zurfin ruwa ba tilas ba ne. Idan ka bar fanko ana rubuta mafi ƙarancin matakin.",
                "Bayani ba tilas ba ne — tsarin zai rubuta sakon taimakon kai tsaye.",
                "Dole ne ka haɗa: 📷 Hoto, 🎤 Murya, ko 🎥 Bidiyo.",
                "Danna Aika Rahoto. NIHSA za ta duba kafin wallafawa.",
            ]
        },
        "yo": {
            "title": "Bii Ṣe Jabo Iṣan-omi",
            "steps": [
                "Tẹ bọtini pupa 🚨 lati ṣii fọọmu ijabọ.",
                "GPS rẹ yoo wa ipo rẹ laifọwọyi. Fa pin lati ṣatunṣe ipo.",
                "Ijinlẹ omi kii ṣe dandan. Ipele to kere ju ni a gbasilẹ ti o ba jẹ ṣofo.",
                "Apejuwe kii ṣe dandan — eto naa gbasilẹ ifiranṣẹ iranlọwọ.",
                "O GBỌDỌ so: 📷 Fọto, 🎤 Ohun, tabi 🎥 Fidio.",
                "Tẹ Firanṣẹ. NIHSA yoo ṣatunyẹwo ṣaaju titẹjade.",
            ]
        },
        "ig": {
            "title": "Otu Esi Akọọ Mmiri Ozuzo",
            "steps": [
                "Kụọ bọtịn ọbara ọbara 🚨 iji mepee ụdị.",
                "GPS gị ga-achọpụta ọnọdụ gị ozugbo. Dọkpụ pin igo dozie ọnọdụ.",
                "Omimi mmiri adịghị achọrọ. A na-edekọ ọkwa kacha ala ọ bụrụ na ị hapụ n'efu.",
                "Nkọwa adịghị achọrọ — sistemu ga-edekọ ozi enyemaka.",
                "Ị KWESỊRỊ itinye: 📷 Foto, 🎤 Olu, ma ọ bụ 🎥 Vidiyo.",
                "Kụọ Nyefee. NIHSA ga-enyocha tupu ebipụta.",
            ]
        },
        "fr": {
            "title": "Comment Signaler une Inondation",
            "steps": [
                "Appuyez sur le bouton rouge 🚨 pour ouvrir le formulaire de rapport.",
                "Votre GPS détecte automatiquement votre position. Faites glisser le pin pour ajuster.",
                "La profondeur est optionnelle. Le niveau minimum est enregistré si laissé vide.",
                "La description est optionnelle — le système enregistre un message d'aide par défaut.",
                "Vous DEVEZ joindre: 📷 Photo, 🎤 Voix, ou 🎥 Vidéo.",
                "Appuyez sur Soumettre. NIHSA examine tous les rapports avant publication.",
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
                "🔴 CRITICAL/EXTREME: Evacuate NOW. Do not cross flooded roads. Call 112 immediately.",
                "Alerts are generated by the NFFS (LSTM deep learning model on 70 basins) and verified by NIHSA hydrologists.",
                "Active alerts scroll as a ticker at the bottom of every screen. Tap the ticker to go to the Alerts tab.",
            ]
        },
        "ha": {
            "title": "Fahimtar Faɗakarwar Ambaliya",
            "steps": [
                "🟢 AL'ADA: Matakan kogi a aminci. Babu aiki da ake bukata.",
                "🟡 KALLO: Matakan suna tashi — shirya kayan gaggawa, san hanyar tserewarka.",
                "🟠 BABBA: Ana sa ran ambaliya cikin awanni 12-24 — shirya ƙaura.",
                "🔴 MATSANANCI: Gudu YANZU. Kira 112 nan da nan.",
                "NFFS da masu ilimin ruwa na NIHSA suna samar da faɗakarwa.",
                "Faɗakarwa masu aiki suna gungura a ƙasan kowane allo.",
            ]
        },
        "yo": {
            "title": "Oye Awọn Ìkìlọ̀ Iṣan-omi",
            "steps": [
                "🟢 DEEDE: Awọn ipele odò ailewu. Ko si igbese ti o nilo.",
                "🟡 WIWO: Awọn ipele n dide — mura apo pajawiri, mọ ipa ọna iṣapá.",
                "🟠 GIGA: Iṣan-omi nireti ni wakati 12-24 — mura fun iṣapá.",
                "🔴 PATAKI: Salọ BAYI. Pe 112 lẹsẹkẹsẹ.",
                "NFFS ati awọn onimọ-omi NIHSA n ṣẹda awọn ìkìlọ̀.",
                "Awọn ìkìlọ̀ ti nṣiṣẹ n yipo ni isalẹ gbogbo iboju.",
            ]
        },
        "ig": {
            "title": "Ighọta Ọkwa Mmiri Ozuzo",
            "steps": [
                "🟢 NKỊTỊ: Ọkwa osimiri nchekwa. Ọ dịghị ihe achọrọ ime.",
                "🟡 ELE ANYA: Ọkwa na-arị elu — kwado ngwugwu ihe mberede.",
                "🟠 ELU: A na-atọ anya mmiri ozuzo n'ime awa 12-24 — kwado ịnnarị.",
                "🔴 SIRI IKE: Narịa UGBU A. Kpọọ 112 ozugbo.",
                "NFFS na ndị ọkà mmụta mmiri NIHSA na-emepụta ọkwa.",
                "Ọkwa na-arụ ọrụ na-atọgharị n'ala ihuenyo ọ bụla.",
            ]
        },
        "fr": {
            "title": "Comprendre les Alertes d'Inondation",
            "steps": [
                "🟢 NORMAL: Niveaux des rivières dans la plage sûre. Aucune action nécessaire.",
                "🟡 SURVEILLANCE: Niveaux en hausse — préparez votre kit d'urgence.",
                "🟠 ÉLEVÉ: Inondation attendue dans 12-24 heures — préparez-vous à évacuer.",
                "🔴 CRITIQUE: Évacuez MAINTENANT. Appelez le 112 immédiatement.",
                "Les alertes sont générées par le NFFS et vérifiées par les hydrologues NIHSA.",
                "Les alertes actives défilent en bas de chaque écran.",
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
                "Open 🗂️ Map Layers (top-left) to toggle: AFO 2026 flood extent, population at risk, health facilities, schools, farmland, and more.",
                "Tap 🚨 Report Flood to submit a report directly from the map — your GPS pin is pre-set.",
            ]
        },
        "ha": {
            "title": "Amfani da Taswira",
            "steps": [
                "Danna 📍 don tashi zuwa wurin GPS. Ba da izinin Wuri a Saitunan.",
                "Yi amfani da sandar bincike don nemo al'umma, LGA, jiha, ko wuri.",
                "Shuɗi = tashar aunawa. Kore = al'ada, Rawaya = kallo, Orange = babba, Ja = matsananci.",
                "Alamomin faɗakarwa suna nuna inda NIHSA ta ba da gargaɗi.",
                "Buɗe 🗂️ Matakan Taswira don kunna layuka daban-daban.",
                "Danna 🚨 don aika rahoto kai tsaye daga taswira.",
            ]
        },
        "yo": {
            "title": "Lilo Maapu",
            "steps": [
                "Tẹ 📍 lati fo si ipo GPS rẹ. Fun igbanilaaye Ipo ni Eto.",
                "Lo ọpa wiwa lati wa agbegbe, LGA, ipinlẹ, tabi aami-ilẹ.",
                "Buluu = awọn ibudo gauge. Alawọ = deede, Ofeefee = wiwo, Osan = giga, Pupa = pataki.",
                "Awọn aami ìkìlọ̀ fihan ibiti NIHSA ti ṣe awọn ikilọ.",
                "Ṣii 🗂️ Awọn Fẹlẹfẹlẹ Maapu lati yipada awọn fẹlẹfẹlẹ.",
                "Tẹ 🚨 lati fi ijabọ silẹ taara lati maapu.",
            ]
        },
        "ig": {
            "title": "Iji Maapu",
            "steps": [
                "Kụọ 📍 iji wụọ ọnọdụ GPS gị. Nye ikike Ọnọdụ na Nhazi.",
                "Jiri ọwa ọchọ iji chọọ obodo, LGA, steeti, ma ọ bụ akara.",
                "Ojii = ọdụ gauge. Ọcha = nkịtị, Odo = ele anya, Ọrọ = elu, Ọbara = siri ike.",
                "Ihe nchọpụta ọkwa na-egosi ebe NIHSA nyere ọkwa.",
                "Mepee 🗂️ iji tụgharịa ọtụtụ ihe maapu.",
                "Kụọ 🚨 iji zipu akụkọ ozugbo site na maapu.",
            ]
        },
        "fr": {
            "title": "Utiliser la Carte",
            "steps": [
                "Appuyez sur 📍 pour voler à votre position GPS. Accordez la permission Localisation.",
                "Utilisez la barre de recherche pour trouver une communauté, LGA, État ou repère.",
                "Bleu = stations de jaugeage. Vert = normal, Jaune = surveillance, Orange = élevé, Rouge = critique.",
                "Les marqueurs d'alerte indiquent où la NIHSA a émis des avertissements.",
                "Ouvrez 🗂️ Couches de Carte pour activer différentes couches.",
                "Appuyez sur 🚨 pour soumettre un rapport directement depuis la carte.",
            ]
        },
    },
    "vanguard": {
        "en": {
            "title": "Flood Marshals Network",
            "steps": [
                "The Vanguard network is NIHSA's real-time coordination system for Flood Marshals across Nigeria's 36 states + FCT.",
                "38 channels: one per state, one for FCT (Abuja), and one National command channel for cross-state coordination.",
                "Only verified Flood Marshals (Vanguard role), NIHSA Staff, and government officials can post. Citizens can view all messages.",
                "To become a Flood Marshal: register, check 'I am a Flood Marshal', and wait for NIHSA approval.",
                "Messages sync in real-time via WebSocket. If connection drops, messages reload automatically on reconnect.",
                "Even without signing in, you can view all messages and follow live situational updates from Flood Marshals in the field.",
            ]
        },
        "ha": {
            "title": "Hanyar Masu Kiyaye Ambaliya",
            "steps": [
                "Hanyar Vanguard ita ce tsarin daidaitawa na gaskiya ta NIHSA a jihohi 36 + FCT.",
                "Tasoshi 38: ɗaya ga kowace jiha, ɗaya don FCT, da ɗaya Na Ƙasa.",
                "Masu Kiyaye Ambaliya da aka tabbatar da ma'aikatan NIHSA ne kawai za su iya aika.",
                "Don zama Mai Kiyaye Ambaliya: yi rajista kuma jira amincewar NIHSA.",
                "Sakonni suna aiki kai tsaye. Idan haɗi ya faɗi, ana sake loda.",
                "Ko ba tare da shiga ba, kuna iya duba duk sakonni.",
            ]
        },
        "yo": {
            "title": "Nẹtiwọọki Awọn Oluso Iṣan-omi",
            "steps": [
                "Nẹtiwọọki Vanguard jẹ eto isọdọkan gidi-akoko NIHSA kọja awọn ipinlẹ 36 + FCT.",
                "Ikanni 38: ọkan fun ipinlẹ kọọkan, ọkan fun FCT, ati ọkan Orilẹ-ede.",
                "Awọn Oluso ti a fọwọsi ati oṣiṣẹ NIHSA nikan le firanṣẹ.",
                "Lati di Oluso: forukọsilẹ ki o duro fun ifọwọsi NIHSA.",
                "Awọn ifiranṣẹ n ṣiṣẹpọ ni akoko gidi. Tun load laifọwọyi ti asopọ ba ṣubu.",
                "Paapaa laisi wiwọle, o le wo gbogbo awọn ifiranṣẹ.",
            ]
        },
        "ig": {
            "title": "Netwọk Ndị Nlekota Mmiri Ozuzo",
            "steps": [
                "Netwọk Vanguard bụ sistemu nhazi oge-ndụ NIHSA n'elu steeti 36 Naịjirịa + FCT.",
                "Ọwa 38: otu maka steeti ọ bụla, otu maka FCT, na otu Mba.",
                "Naanị Ndị Nlekota kwadoro na ndị ọrụ NIHSA nwere ike iziga.",
                "Iji bụrụ Onye Nlekota: debanye aha wee chere nkwado NIHSA.",
                "Ozi na-emekọ ihe n'oge ndụ. Na-abuọ lode ozugbo mgbe atụkwara njikọ.",
                "Ọbụlagodi na-enweghị ịbanye, ị nwere ike ilelee ozi niile.",
            ]
        },
        "fr": {
            "title": "Réseau des Gardes des Inondations",
            "steps": [
                "Le réseau Vanguard est le système de coordination en temps réel de la NIHSA à travers les 36 États + FCT.",
                "38 canaux: un par État, un pour le FCT, et un canal de commandement national.",
                "Seuls les Gardes vérifiés et le personnel NIHSA peuvent poster.",
                "Pour devenir Garde: inscrivez-vous et attendez l'approbation NIHSA.",
                "Les messages se synchronisent en temps réel. Rechargement automatique à la reconnexion.",
                "Même sans se connecter, vous pouvez voir tous les messages.",
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
                "Tap 'Flood Animation' or 'Flood Extent Map' to open interactive NFFS model output maps.",
            ]
        },
        "ha": {
            "title": "Jagoran Allon Bayanai",
            "steps": [
                "Canza tsakanin ra'ayi na Shekara (AFO 2026) da na Mako.",
                "Ra'ayi na Shekara yana nuna cikakken Hasashen Ambaliya 2026 a cikin yadudduka 17.",
                "Ra'ayi na Mako yana nuna hasashen kwanaki 7 na yanzu.",
                "Katunan fallasa suna nuna jimlar al'umma, mutane, cibiyoyin lafiya, makarantu, da sauransu.",
                "Yi amfani da zaɓi na jiha don tace bayanan ta jiha.",
                "Danna 'Motsin Ambaliya' don buɗe taswira mai aiki na NFFS.",
            ]
        },
        "yo": {
            "title": "Itọsọna Paali Alaye",
            "steps": [
                "Yipada laarin awọn iwoye Lọdọọdún (AFO 2026) ati Ọsẹ.",
                "Iwoye Lọdọọdún fihan AFO 2026 ni kikun kọja awọn fẹlẹfẹlẹ 17.",
                "Iwoye Ọsẹ fihan asọtẹlẹ ọjọ 7 lọwọlọwọ.",
                "Awọn kaadi ifihan fihan awọn apapọ fun awọn agbegbe, eniyan, awọn ile-iwosan, ati bẹbẹ lọ.",
                "Lo akojọ silẹ ipinlẹ lati ṣàlẹmọ data nipasẹ ipinlẹ.",
                "Tẹ 'Agbeka Iṣan-omi' lati ṣii awọn maapu NFFS interactif.",
            ]
        },
        "ig": {
            "title": "Nduzi Penu Ozi",
            "steps": [
                "Gbanwee n'etiti ọhụụ Ọdụn (AFO 2026) na Izu.",
                "Ọhụụ Ọdụn na-egosi AFO 2026 zuru oke n'elu ọkwa 17.",
                "Ọhụụ Izu na-egosi atụmatụ ụbọchị 7 ugbu a.",
                "Kaadị mficha na-egosi ngụkọ maka obodo, ndị mmadụ, ụlọ ọgwụ, na ndị ọzọ.",
                "Jiri dropụdaụn steeti iji lelee data site n'steeti.",
                "Kụọ 'Ngagharị Mmiri Ozuzo' iji mepee maapu NFFS.",
            ]
        },
        "fr": {
            "title": "Guide du Tableau de Bord",
            "steps": [
                "Basculez entre les vues Annuelle (AFO 2026) et Hebdomadaire.",
                "La vue Annuelle montre l'AFO 2026 complet sur 17 couches.",
                "La vue Hebdomadaire montre les prévisions des 7 prochains jours.",
                "Les cartes d'exposition montrent les totaux pour communautés, personnes, centres de santé, etc.",
                "Utilisez la liste déroulante d'État pour filtrer les données.",
                "Appuyez sur 'Animation d'Inondation' pour ouvrir les cartes NFFS interactives.",
            ]
        },
    },
    "assistant": {
        "en": {
            "title": "Using the AI Assistant",
            "steps": [
                "NIHSA FloodAI is trained specifically for Nigerian flood safety and hydrology. Use it only for flood, water, and emergency topics.",
                "Type your question in the chat box and tap Ask, or tap 🎤 to speak — your voice is transcribed and sent automatically.",
                "The AI detects your language automatically and responds in Hausa, Yoruba, Igbo, French, or English.",
                "You can ask about: flood risk for your area, gauge readings, evacuation procedures, flood safety tips, NFFS data, river levels, AFO 2026 forecast.",
                "The AI searches the internet for current flood news, dam release announcements, and live emergency contact numbers.",
                "Daily limits: 7 prompts/day for all roles (100 for Admin). Sign in to use prompts.",
            ]
        },
        "ha": {
            "title": "Amfani da Mataimaki na AI",
            "steps": [
                "NIHSA FloodAI an horar da shi don amincin ambaliya da ilimin ruwa na Najeriya.",
                "Rubuta tambayarka ko danna 🎤 don magana — ana fassara kuma aika kai tsaye.",
                "AI yana gano harsheka kai tsaye kuma yana amsa da harshen da ya dace.",
                "Kuna iya tambaya game da: haɗarin ambaliya, aunawa, hanyoyin kwasawa, NFFS.",
                "AI na iya bincika labarai na yanzu da lambobin gaggawa kai tsaye daga intanet.",
                "Iyakar yau da kullun: tambayoyi 7/rana. Shiga don amfani.",
            ]
        },
        "yo": {
            "title": "Lilo Oluranlowo AI",
            "steps": [
                "NIHSA FloodAI jẹ ikẹkọ pataki fun aabo iṣan-omi ati imọ-omi Naijiria.",
                "Tẹ ibeere rẹ tabi tẹ 🎤 lati sọrọ — o jẹ tumọ ati firanṣẹ laifọwọyi.",
                "AI ṣawari ede rẹ laifọwọyi ati dahun ni ede rẹ.",
                "O le beere nipa: eewu iṣan-omi, awọn kika gauge, awọn ilana iṣapá, NFFS.",
                "AI le wa awọn iroyin iṣan-omi lọwọlọwọ ati awọn nọmba pajawiri lori intanẹẹti.",
                "Awọn opin ojoojumọ: awọn ibeere 7/ọjọ. Wọle lati lo.",
            ]
        },
        "ig": {
            "title": "Iji Onye Enyemaka AI",
            "steps": [
                "E zigara NIHSA FloodAI ọzụzụ maka nchekwa mmiri ozuzo na mmụta mmiri Naịjirịa.",
                "Dee ajụjụ gị ma ọ bụ kụọ 🎤 iji kwuo — a na-atụgharịa ma zigaa ozugbo.",
                "AI na-achọpụta asụsụ gị ozugbo ma zaghachi n'asụsụ gị.",
                "Ị nwere ike ịjụ maka: ihe ize ndụ mmiri, ọgụgụ gauge, usoro nnarị, NFFS.",
                "AI nwere ike ịchọ ozi mmiri ozuzo ugbu a na nọmba ihe mberede n'ịntanetị.",
                "Oke ụbọchị: ajụjụ 7/ụbọchị. Banye iji jiri.",
            ]
        },
        "fr": {
            "title": "Utiliser l'Assistant IA",
            "steps": [
                "NIHSA FloodAI est formé spécifiquement pour la sécurité des inondations et l'hydrologie nigériane.",
                "Tapez votre question ou appuyez sur 🎤 pour parler — transcrit et envoyé automatiquement.",
                "L'IA détecte votre langue automatiquement et répond dans celle-ci.",
                "Vous pouvez demander: risques d'inondation, lectures de jauges, procédures d'évacuation, NFFS.",
                "L'IA recherche les actualités hydrologiques récentes et les contacts d'urgence en ligne.",
                "Limites quotidiennes: 7 requêtes/jour. Connectez-vous pour utiliser.",
            ]
        },
    },
}

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NIHSA AI Wrapper starting — Cloudflare STT + MeloTTS + SerpAPI Web Search")
    
    await init_db_pool()
    await _cleanup_old_usage()
    yield
    logger.info("NIHSA AI Wrapper shutting down")
    await close_db_pool()

app = FastAPI(
    title="NIHSA AI Assistant Wrapper",
    description="Flood intelligence AI with Cloudflare STT/TTS, live web search, and emergency contacts — NIHSA Nigeria",
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
        "tts_provider": "Cloudflare Workers AI — MeloTTS",
        "llm_provider": "DeepSeek Chat",
        "web_search": "DuckDuckGo — free hydrology search (Nigeria-focused)",
        "quota_backend": "PostgreSQL" if DATABASE_URL else "in-memory fallback",
    }


@app.post("/ai/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = Form(default="default"),
):
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

    # ── Detect language ───────────────────────────────────────────────────────
    language = body.language
    if not language and body.messages:
        last_msg = body.messages[-1].content
        language = await detect_language_deepseek(last_msg)
        if language == "en":
            language = detect_language_keywords(last_msg)
    language = language or "en"

    # ── Build flood context ───────────────────────────────────────────────────
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
        flood_context = await fetch_flood_context()

    # ── Build location context ────────────────────────────────────────────────
    location_context = ""
    if body.user_location:
        loc = body.user_location
        lat = loc.get("lat")
        lng = loc.get("lng")
        address = loc.get("address", "")
        coords = f"{lat:.4f}°N, {lng:.4f}°E" if lat and lng else "unknown"
        nearby_alerts = []
        if body.active_alerts:
            user_state = address.split(",")[-1].strip().lower() if address else ""
            for a in body.active_alerts:
                alert_state = (a.get("state") or "").lower()
                if user_state and (user_state in alert_state or alert_state in user_state):
                    nearby_alerts.append(a)
        if nearby_alerts:
            nearby_str = f"\nALERTS IN USER'S STATE ({address.split(',')[-1].strip() if address else 'their area'}):\n"
            for a in nearby_alerts[:4]:
                nearby_str += f"  🔴 [{a.get('level','')}] {a.get('title','')} — LGAs: {', '.join((a.get('lgas') or [])[:4])}\n"
        else:
            nearby_str = f"\nNo active NIHSA alerts currently in {address.split(',')[-1].strip() if address else 'their state'}."
        location_context = (
            f"\n{'='*50}\n"
            f"USER'S CURRENT LOCATION (GPS-verified — do NOT ask them for location):\n"
            f"  Address: {address}\n"
            f"  Coordinates: {coords}\n"
            f"{nearby_str}\n"
            f"ABSOLUTE RULE: NEVER ask the user to provide, share, or confirm their location.\n"
            f"{'='*50}"
        )
    else:
        location_context = (
            "\nUSER LOCATION: Not yet available (GPS loading or permission denied). "
            "If they ask about their area, ask them to name their state or LGA."
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
            temperature=0.5,
        )

        message = response.choices[0].message
        await increment_usage(user_id, role)

        # ── Handle tool calls ─────────────────────────────────────────────────
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_params = json.loads(tool_call.function.arguments)

            # ── web_search_hydrology ─────────────────────────────────────────
            if tool_name == "web_search_hydrology":
                search_query = tool_params.get("query", "")
                logger.info(f"Web search triggered: '{search_query}'")
                search_results = await execute_web_search(search_query)

                messages_with_result = messages + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": "web_search_hydrology",
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_results
                    }
                ]

                final_response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages_with_result,
                    max_tokens=600,
                    temperature=0.4,
                )
                reply = (
                    final_response.choices[0].message.content
                    or "I found some information but could not summarise it clearly. Please check the NIHSA website for the latest updates."
                )
                return ChatResponse(reply=reply, action=None, detected_language=language)

            # ── get_emergency_contacts ───────────────────────────────────────
            if tool_name == "get_emergency_contacts":
                state = tool_params.get("state", "")
                logger.info(f"Emergency contacts requested: state='{state or 'national'}'")
                contacts_data = await fetch_emergency_contacts(state)

                messages_with_result = messages + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": "get_emergency_contacts",
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": contacts_data
                    }
                ]

                final_response = await deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages_with_result,
                    max_tokens=400,
                    temperature=0.3,
                )
                reply = (
                    final_response.choices[0].message.content
                    or (
                        "🆘 Key Nigerian emergency numbers:\n"
                        "• 112 — General emergency (free, all networks)\n"
                        "• 199 — Nigeria Fire Service\n"
                        "• 123 — Nigeria Police Force\n"
                        "• 0800-CALL-NEMA — NEMA national flood hotline\n"
                        "Please call 112 immediately if you are in danger."
                    )
                )
                return ChatResponse(reply=reply, action=None, detected_language=language)

            # ── All other app navigation/action tools ────────────────────────
            action = {
                "type": tool_name,
                "params": tool_params
            }

            action_context = {
                "navigate_to_report": (
                    "The user wants to report flooding. Tell them you are opening the report form. "
                    "Also give them 3–4 immediate safety guidelines based on their location and active alerts."
                ),
                "escalate_to_human": (
                    "The user needs human assistance. Tell them you are opening the report form so NIHSA coordinators can see their situation. "
                    "Give them the most important flood safety guidelines RIGHT NOW while they wait."
                ),
                "get_flood_status": (
                    f"The user is asking about flood conditions at: {tool_params.get('location', 'their area')}. "
                    "Using the active alerts and location data in your context, give a direct, specific answer. "
                    "State the current risk level, any active alerts, and what actions they should take."
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
                reply = {
                    "navigate_to_report": "🚨 Opening the flood report form. Please attach a photo, voice note, or video so NIHSA coordinators can verify and respond.",
                    "escalate_to_human": "🆘 Opening the report form for NIHSA review. Move to higher ground, do not cross flooded water, call 112 for life-threatening emergencies.",
                    "get_flood_status": f"Checking flood conditions for {tool_params.get('location', 'your area')} — see the map for live gauge readings.",
                    "show_tutorial": f"Opening the {tool_params.get('topic', '')} guide for you.",
                    "navigate_to_tab": f"Taking you to the {tool_params.get('tab', '')} tab.",
                    "search_location": f"Showing {tool_params.get('query', '')} on the map.",
                }.get(tool_name, "Done — please check the screen for the result.")

            return ChatResponse(reply=reply, action=action, detected_language=language)

        # ── No tool call — plain text response ───────────────────────────────
        reply = message.content or "I'm here to help with flood safety. Please ask a hydrology-related question."
        return ChatResponse(reply=reply, action=None, detected_language=language)

    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable. Please try again.")


@app.post("/ai/speak")
async def speak_endpoint(
    text: str = Form(...),
    language: str = Form(default="en"),
    session_id: str = Form(default="default"),
):
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
    Fast quota check — decodes JWT locally without calling the main backend.
    Requires SECRET_KEY env var (must match the main backend's SECRET_KEY).
    Used for initial page load to avoid cold-start double network hop.
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
    """Full quota check — verifies token against the main NIHSA backend."""
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
    if topic not in TUTORIALS:
        raise HTTPException(status_code=404, detail=f"Tutorial topic '{topic}' not found.")
    topic_data = TUTORIALS[topic]
    if lang not in topic_data:
        lang = "en"
    content = topic_data[lang]
    return TutorialResponse(title=content["title"], steps=content["steps"], language=lang)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
