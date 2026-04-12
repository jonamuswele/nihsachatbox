#!/usr/bin/env python3
"""
NIHSA AI Assistant Wrapper Service
Production-grade FastAPI service with Cloudflare Whisper STT, DeepSeek chat, and Google TTS
Optimized for Render deployment - NO local Whisper memory issues
"""

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
from google.cloud import texttospeech
from cachetools import TTLCache

# ============================================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================================

# Environment variables
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not set!")

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
if not CLOUDFLARE_ACCOUNT_ID:
    raise ValueError("CLOUDFLARE_ACCOUNT_ID not set!")

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
if not CLOUDFLARE_API_TOKEN:
    raise ValueError("CLOUDFLARE_API_TOKEN not set!")

NIHSA_API_URL = os.environ.get("NIHSA_API_URL", "https://nihsa-backend-20hh.onrender.com/api")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS",
                                 "capacitor://localhost,http://localhost:3000,https://nihsa-backend-20hh.onrender.com").split(",")
TTS_CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", "/app/tts_cache"))

# Google Cloud credentials from environment variable
google_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if google_creds_json:
    creds_path = Path("/app/gcp-credentials.json")
    creds_path.write_text(google_creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

# Create directories
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("nihsa-ai-wrapper")

# ============================================================================
# DATA MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: str = Field(default="default")
    language: Optional[str] = None  # Optional, auto-detect if not provided

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
# FUNCTION CALLING TOOL DEFINITIONS
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
                    "prefill_location": {
                        "type": "string",
                        "description": "Optional location to pre-fill in the report form"
                    },
                    "prefill_description": {
                        "type": "string",
                        "description": "Optional description to pre-fill"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to_tab",
            "description": "User wants to switch to a different tab in the app.",
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
                        "enum": ["reporting", "alerts", "map", "vanguard", "dashboard", "general"],
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
                        "description": "Location search query (e.g., city, state, landmark)"
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
                    "location": {
                        "type": "string",
                        "description": "Location to check flood status for"
                    }
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
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for escalation"
                    }
                },
                "required": []
            }
        }
    }
]

# ============================================================================
# TUTORIAL CONTENT (Multi-language) - COMPLETE, UNCHANGED
# ============================================================================

TUTORIALS = {
    "general": {
        "en": {
            "title": "Welcome to NIHSA Flood Intelligence",
            "steps": [
                "Map tab: View live flood conditions, river gauges, and alerts on the interactive map",
                "Dashboard tab: See flood statistics and outlook summaries",
                "Vanguard tab: Join the Flood Marshals coordination network",
                "Assistant tab: Ask me questions about flood safety and get help",
                "Alerts tab: Browse all active flood warnings and alerts",
                "Tap the 🚨 button to report flooding in your area"
            ]
        },
        "ha": {
            "title": "Barka da zuwa NIHSA Flood Intelligence",
            "steps": [
                "Taswirar Map: Duba yanayin ambaliya kai tsaye, ma'aunin kogi, da fadakarwa akan taswira",
                "Allon Dashboard: Duba kididdigar ambaliya da takaitaccen hasashe",
                "Tashar Vanguard: Shiga cibiyar sadarwar Masu Kiyaye Ambaliya",
                "Taimakon Assistant: Yi mani tambayoyi game da amincin ambaliya",
                "Shafin Alerts: Bincika duk fadakarwa da gargadin ambaliya",
                "Danna maɓallin 🚨 don bayar da rahoton ambaliya a yankinku"
            ]
        },
        "yo": {
            "title": "Kaabọ si NIHSA Flood Intelligence",
            "steps": [
                "Maapu: Wo ipo iṣan-omi laaye, awon gauji odo, ati awon ifokanbalẹ lori maapu",
                "Paali Alaye: Wo awon iṣiro iṣan-omi ati akojọpọ asotele",
                "Ikanni Vanguard: Dara pọ mọ nẹtiwọki awon Oluso Iṣan-omi",
                "Oluranlọwọ Assistant: Beere awon ibeere nipa aabo iṣan-omi",
                "Taabu Alerts: Ṣawari gbogbo awon ikilo ati ifokanbalẹ iṣan-omi ti n ṣiṣẹ",
                "Tẹ bọtini 🚨 lati jabo iṣan-omi ni agbegbe rẹ"
            ]
        },
        "ig": {
            "title": "Nnọọ na NIHSA Flood Intelligence",
            "steps": [
                "Maapu: Lelee ọnọdụ mmiri ozuzo, ihe nzụta osimiri, na ọkwa na maapụ ahụ",
                "Dashboard: Hụ ọnụ ọgụgụ mmiri ozuzo na nchịkọta amụma",
                "Vanguard: Soro na netwọk nke ndị Nlekota Mmiri Ozuzo",
                "Onye Enyemaka: Jụọ m ajụjụ gbasara nchekwa mmiri ozuzo",
                "Alerts: Chọgharịa ọkwa mmiri ozuzo na ịdọ aka ná ntị niile",
                "Pịa bọtịnụ 🚨 iji kọọ mmiri ozuzo n'ógbè gị"
            ]
        },
        "fr": {
            "title": "Bienvenue sur NIHSA Flood Intelligence",
            "steps": [
                "Carte: Voir les conditions d'inondation en direct et les alertes sur la carte",
                "Tableau de bord: Voir les statistiques et les prévisions",
                "Vanguard: Rejoignez le réseau des Maréchaux des Crues",
                "Assistant: Posez-moi des questions sur la sécurité des inondations",
                "Alertes: Parcourir tous les avertissements actifs",
                "Appuyez sur 🚨 pour signaler une inondation dans votre région"
            ]
        }
    },
    "reporting": {
        "en": {
            "title": "How to Report a Flood",
            "steps": [
                "Tap the red 'Report Flood' button (🚨) on the map screen",
                "Your GPS location will be automatically detected",
                "Drag the pin on the map to adjust the exact location if needed",
                "Select the water depth from the dropdown (ankle to impassable)",
                "Describe what you see - roads blocked, homes affected, people stranded",
                "Optionally add a photo, voice note, or short video",
                "Tap 'Submit Flood Report' to send to NIHSA coordinators",
                "Your report will be verified and may appear on the public map"
            ]
        },
        "ha": {
            "title": "Yadda Ake Bayar da Rahoton Ambaliya",
            "steps": [
                "Danna jan maɓallin 'Rahoton Ambaliya' (🚨) akan allon taswira",
                "Za a gano wurin GPS ɗinka kai tsaye",
                "Ja fil ɗin akan taswira don daidaita ainihin wurin idan ana buƙata",
                "Zaɓi zurfin ruwa daga jerin (daga idon sawu zuwa wanda ba za a iya wucewa ba)",
                "Bayyana abin da kake gani - hanyoyi da aka toshe, gidajen da abin ya shafa",
                "Zabi ƙara hoto, saƙon murya, ko gajeren bidiyo",
                "Danna 'Aika Rahoto' don aikawa ga masu gudanarwa na NIHSA",
                "Za a tabbatar da rahotonka kuma yana iya bayyana akan taswirar jama'a"
            ]
        },
        "yo": {
            "title": "Bi O Ṣe Le Jabo Iṣan-omi",
            "steps": [
                "Tẹ bọtini pupa 'Jabo Iṣan-omi' (🚨) lori oju-iwe maapu",
                "Ipo GPS rẹ yoo jẹ wiwa laifọwọyi",
                "Fa pinni lori maapu lati ṣatunṣe ipo gangan ti o ba nilo",
                "Yan ijinle omi lati inu akojọ (lati kokosẹ si eyiti a ko le kọja)",
                "Ṣapejuwe ohun ti o ri - awọn ọna ti a ti di, awọn ile ti o kan",
                "Yan lati fi fọto, ifiranṣẹ ohun, tabi fidio kukuru kun",
                "Tẹ 'Fi Ijabo Ranṣẹ' lati fi ranṣẹ si awọn alabojuto NIHSA",
                "Ao jẹrisi ijabo rẹ ati pe o le han lori maapu gbogbo eniyan"
            ]
        },
        "ig": {
            "title": "Otu Esi Akọọ Mmiri Ozuzo",
            "steps": [
                "Pịa bọtịnụ uhie 'Kọọ Mmiri Ozuzo' (🚨) na ihuenyo maapụ",
                "A ga-achọpụta ebe GPS gị na-akpaghị aka",
                "Dọrọ ntụtụ ahụ na maapụ ahụ iji dozie ebe ahụ kpọmkwem ma ọ dị mkpa",
                "Họrọ omimi mmiri site na ndepụta (site na nkwonkwo ụkwụ ruo nke a na-agaghị agafe)",
                "Kọwaa ihe ị hụrụ - okporo ụzọ egbochiri, ụlọ ndị emetụtara",
                "Họrọ itinye foto, ozi olu, ma ọ bụ vidiyo dị mkpirikpi",
                "Pịa 'Nyefee Akụkọ' iji ziga ndị nhazi NIHSA",
                "A ga-enyocha akụkọ gị ma ọ nwere ike ịpụta na maapụ ọha"
            ]
        },
        "fr": {
            "title": "Comment Signaler une Inondation",
            "steps": [
                "Appuyez sur le bouton rouge 'Signaler une Inondation' (🚨) sur la carte",
                "Votre position GPS sera détectée automatiquement",
                "Faites glisser l'épingle sur la carte pour ajuster l'emplacement exact",
                "Sélectionnez la profondeur de l'eau dans la liste (de la cheville à infranchissable)",
                "Décrivez ce que vous voyez - routes bloquées, maisons touchées",
                "Ajoutez éventuellement une photo, une note vocale ou une courte vidéo",
                "Appuyez sur 'Soumettre' pour envoyer aux coordinateurs NIHSA",
                "Votre rapport sera vérifié et pourra apparaître sur la carte publique"
            ]
        }
    },
    "alerts": {
        "en": {
            "title": "Understanding Flood Alerts",
            "steps": [
                "Alerts are color-coded by severity: Normal (green), Watch (yellow), Warning (orange), Severe (red), Extreme (dark red)",
                "The Alerts tab shows all active warnings with details",
                "On the map, alert icons show affected areas",
                "Tap any alert to see full details and recommended actions",
                "The ticker at the bottom scrolls through all active alerts",
                "Critical alerts will trigger push notifications on your device"
            ]
        },
        "ha": {
            "title": "Fahimtar Fadakarwar Ambaliya",
            "steps": [
                "Fadakarwa ana nuna su da launuka gwargwadon tsanani: Al'ada (kore), Kallo (rawaya), Gargadi (orange), Mai tsanani (ja), Mai karfi (ja mai duhu)",
                "Shafin Fadakarwa yana nuna duk gargadin da ke aiki tare da cikakkun bayanai",
                "A kan taswira, alamun fadakarwa suna nuna wuraren da abin ya shafa",
                "Danna kowace fadakarwa don ganin cikakkun bayanai da ayyukan da aka ba da shawara",
                "Tikiti a ƙasa yana zagaya cikin duk fadakarwar da ke aiki"
            ]
        },
        "yo": {
            "title": "Loye Awọn Ifokanbalẹ Iṣan-omi",
            "steps": [
                "Awọn ifokanbalẹ jẹ ami awọ nipasẹ bi o ṣe le: Deede (alawọ ewe), Ṣọ (ofeefee), Ikilo (osun), Lile (pupa), Uje nla (pupa dudu)",
                "Taabu Alerts fihan gbogbo awọn ikilo ti n ṣiṣẹ pẹlu awọn alaye",
                "Lori maapu, awọn aami ifokanbalẹ fihan awọn agbegbe ti o kan",
                "Tẹ ifokanbalẹ eyikeyi lati wo awọn alaye kikun ati awọn igbese ti a gbaniyanju",
                "Tika ni isalẹ n yi awọn ifokanbalẹ ti n ṣiṣẹ lọ"
            ]
        },
        "ig": {
            "title": "Ịghọta Ọkwa Mmiri Ozuzo",
            "steps": [
                "A na-eji agba egosi ọkwa site n'ịdị njọ: Nkịtị (akwụkwọ ndụ), Lelee anya (odo), Ịdọ aka ná ntị (oroma), Siri ike (ọbara ọbara), Dị oke njọ (ọbara ọbara gbara ọchịchịrị)",
                "Taabụ Alerts na-egosi ịdọ aka ná ntị niile na-arụ ọrụ yana nkọwa zuru ezu",
                "Na maapụ, akara ọkwa na-egosi ebe ndị emetụtara",
                "Pịa ọkwa ọ bụla iji hụ nkọwa zuru ezu na omume ndị a tụrụ aro"
            ]
        },
        "fr": {
            "title": "Comprendre les Alertes d'Inondation",
            "steps": [
                "Les alertes sont codées par couleur : Normal (vert), Surveillance (jaune), Avertissement (orange), Grave (rouge), Extrême (rouge foncé)",
                "L'onglet Alertes affiche tous les avertissements actifs avec détails",
                "Sur la carte, les icônes d'alerte montrent les zones touchées",
                "Appuyez sur une alerte pour voir les détails et les actions recommandées",
                "Le téléscripteur en bas défile toutes les alertes actives"
            ]
        }
    },
    "map": {
        "en": {
            "title": "Using the Flood Map",
            "steps": [
                "The map shows river gauge stations (colored dots) and active alerts",
                "Use the search bar to find any location in Nigeria",
                "Tap the 📍 button to center on your current GPS location",
                "Layer panel (top left) lets you toggle different data overlays",
                "Legend (bottom right) explains the color codes",
                "Drag the map to explore, pinch to zoom in/out"
            ]
        },
        "ha": {
            "title": "Amfani da Taswirar Ambaliya",
            "steps": [
                "Taswirar tana nuna tashoshin ma'aunin kogi (ɗigo masu launi) da fadakarwa masu aiki",
                "Yi amfani da mashigin bincike don nemo kowane wuri a Najeriya",
                "Danna maɓallin 📍 don sanya taswirar a kan wurin GPS ɗinka",
                "Panel ɗin Layer (saman hagu) yana ba ka damar kunna/kashe bayanai daban-daban",
                "Legend (kasan dama) yana bayyana ma'anar launuka",
                "Ja taswirar don bincika, tattara yatsu don zuƙowa ciki/waje"
            ]
        },
        "yo": {
            "title": "Lilo Maapu Iṣan-omi",
            "steps": [
                "Maapu na fihan awọn ibudo gauji odo (awọn aami awọ) ati awọn ifokanbalẹ ti n ṣiṣẹ",
                "Lo ọpa wiwa lati wa eyikeyi ipo ni Nigeria",
                "Tẹ bọtini 📍 lati fi maapu si ipo GPS rẹ lọwọlọwọ",
                "Nronu Layer (oke apa osi) jẹ ki o yipada laarin awọn akojọpọ data oriṣiriṣi",
                "Legend (isalẹ ọtun) ṣe alaye awọn koodu awọ",
                "Fa maapu lati ṣawari, pọ awọn ika lati sun-un/un-un jade"
            ]
        },
        "ig": {
            "title": "Iji Maapụ Mmiri Ozuzo",
            "steps": [
                "Maapụ ahụ na-egosi ọdụ ihe nzụta osimiri (ntụpọ agba) na ọkwa na-arụ ọrụ",
                "Jiri ogwe ọchụchọ chọta ebe ọ bụla na Nigeria",
                "Pịa bọtịnụ 📍 iji tinye maapụ ahụ n'ebe GPS gị dị ugbu a",
                "Ogwe Layer (n'elu aka ekpe) na-enye gị ohere ịgbanwe n'etiti data dị iche iche",
                "Akụkọ nkọwa (n'ala aka nri) na-akọwa koodu agba",
                "Dọrọ maapụ ahụ iji nyochaa, tụkọta mkpịsị aka iji bubata/bupụ"
            ]
        },
        "fr": {
            "title": "Utiliser la Carte des Inondations",
            "steps": [
                "La carte montre les stations de jaugeage (points colorés) et les alertes actives",
                "Utilisez la barre de recherche pour trouver n'importe quel endroit au Nigeria",
                "Appuyez sur 📍 pour centrer sur votre position GPS actuelle",
                "Le panneau des couches (en haut à gauche) permet d'activer/désactiver les superpositions",
                "La légende (en bas à droite) explique les codes couleur",
                "Faites glisser la carte pour explorer, pincez pour zoomer"
            ]
        }
    },
    "vanguard": {
        "en": {
            "title": "Flood Marshals Network (Vanguard)",
            "steps": [
                "The Vanguard tab is a secure chat network for verified Flood Marshals",
                "Choose from National or State-specific channels",
                "You can read messages without signing in",
                "To participate, sign in and request Flood Marshal verification",
                "Share real-time observations and coordinate response efforts",
                "NIHSA staff and coordinators monitor all channels"
            ]
        },
        "ha": {
            "title": "Cibiyar Sadarwar Masu Kiyaye Ambaliya (Vanguard)",
            "steps": [
                "Shafin Vanguard cibiyar tattaunawa ce mai tsaro ga Masu Kiyaye Ambaliya da aka tabbatar",
                "Zaɓi daga tashoshin Ƙasa ko na Jiha",
                "Kuna iya karanta saƙonni ba tare da shiga ba",
                "Don shiga, shiga kuma nemi tabbatarwa a matsayin Mai Kiyaye Ambaliya",
                "Raba abubuwan lura na ainihin lokaci da daidaita ƙoƙarin amsawa"
            ]
        },
        "yo": {
            "title": "Nẹtiwọki Awọn Oluso Iṣan-omi (Vanguard)",
            "steps": [
                "Taabu Vanguard jẹ nẹtiwọki iwiregbe ailewu fun awọn Oluso Iṣan-omi ti a fọwọsi",
                "Yan lati awọn ikanni ti Orilẹ-ede tabi ti Ipinle",
                "O le ka awọn ifiranṣẹ laisi wíwọle",
                "Lati kopa, wọle ki o beere ijẹrisi Oluso Iṣan-omi",
                "Pin awọn akiyesi akoko gidi ati ipoidojuko awọn igbiyanju idahun"
            ]
        },
        "ig": {
            "title": "Netwọk Ndị Nlekota Mmiri Ozuzo (Vanguard)",
            "steps": [
                "Taabụ Vanguard bụ netwọk nkata echedoro maka ndị Nlekota Mmiri Ozuzo enyochara",
                "Họrọ site na ọwa Mba ma ọ bụ nke Steeti",
                "Ị nwere ike ịgụ ozi n'ebanyeghị",
                "Iji sonye, banye ma rịọ nkwenye Nlekota Mmiri Ozuzo",
                "Kesaa ihe ndị a hụrụ n'oge na ịhazi mbọ nzaghachi"
            ]
        },
        "fr": {
            "title": "Réseau des Maréchaux des Crues (Vanguard)",
            "steps": [
                "L'onglet Vanguard est un réseau de chat sécurisé pour les Maréchaux des Crues vérifiés",
                "Choisissez parmi les canaux Nationaux ou spécifiques à l'État",
                "Vous pouvez lire les messages sans vous connecter",
                "Pour participer, connectez-vous et demandez la vérification de Maréchal des Crues",
                "Partagez des observations en temps réel et coordonnez les efforts"
            ]
        }
    },
    "dashboard": {
        "en": {
            "title": "Understanding the Dashboard",
            "steps": [
                "The Dashboard shows flood statistics and exposure summaries",
                "Toggle between Annual (AFO 2026) and Weekly forecasts",
                "See estimated impact: population at risk, schools, health facilities",
                "Use the state filter to view data for specific states",
                "Click any stat card to open detailed maps from the NFFS Atlas",
                "Data is updated daily from NIHSA's flood models"
            ]
        },
        "ha": {
            "title": "Fahimtar Dashboard",
            "steps": [
                "Dashboard yana nuna kididdigar ambaliya da taƙaitaccen fallasa",
                "Sauya tsakanin Hasashen Shekara (AFO 2026) da na Mako-mako",
                "Duba kiyasin tasiri: yawan mutanen da ke cikin haɗari, makarantu, cibiyoyin lafiya",
                "Yi amfani da tacewar jiha don duba bayanan takamaiman jihohi",
                "Danna kowane katin ƙididdiga don buɗe taswirori daga NFFS Atlas"
            ]
        },
        "yo": {
            "title": "Loye Dashboard",
            "steps": [
                "Dashboard fihan awọn iṣiro iṣan-omi ati awọn akojọpọ ifihan",
                "Yipada laarin Ọdọọdun (AFO 2026) ati awọn asọtẹlẹ Ọsẹ",
                "Wo ipa ifoju: olugbe ti o wa ninu ewu, awọn ile-iwe, awọn ohun elo ilera",
                "Lo àlẹmọ ipinlẹ lati wo data fun awọn ipinlẹ kan pato",
                "Tẹ kaadi iṣiro eyikeyi lati ṣii awọn maapu alaye lati NFFS Atlas"
            ]
        },
        "ig": {
            "title": "Ịghọta Dashboard",
            "steps": [
                "Dashboard na-egosi ọnụ ọgụgụ mmiri ozuzo na nchịkọta mkpughe",
                "Gbanwee n'etiti Amụma Afọ (AFO 2026) na nke Izu",
                "Hụ mmetụta e mere atụmatụ: ọnụ ọgụgụ ndị nọ n'ihe ize ndụ, ụlọ akwụkwọ, ụlọ ọrụ ahụike",
                "Jiri nzacha steeti lee data maka steeti ụfọdụ",
                "Pịa kaadi ọnụ ọgụgụ ọ bụla iji mepee maapụ zuru ezu site na NFFS Atlas"
            ]
        },
        "fr": {
            "title": "Comprendre le Tableau de Bord",
            "steps": [
                "Le tableau de bord affiche les statistiques et résumés d'exposition",
                "Basculez entre les prévisions Annuelles (AFO 2026) et Hebdomadaires",
                "Voir l'impact estimé : population à risque, écoles, établissements de santé",
                "Utilisez le filtre par État pour voir les données d'États spécifiques",
                "Cliquez sur une carte pour ouvrir des cartes détaillées de l'Atlas NFFS"
            ]
        }
    }
}

# ============================================================================
# GLOBAL STATE & CACHES
# ============================================================================

# DeepSeek client
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# Google TTS client (lazy)
_tts_client = None

# HTTP client for NIHSA backend and Cloudflare
http_client = httpx.AsyncClient(timeout=60.0)

# Session storage (TTL cache: max 1000 items, 1 hour expiry)
session_cache = TTLCache(maxsize=1000, ttl=3600)

# Flood context cache (30 seconds)
flood_context_cache = {"data": None, "timestamp": datetime.min}

# Rate limiting: session_id -> list of timestamps
rate_limit_store: Dict[str, List[datetime]] = defaultdict(list)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_rate_limit(session_id: str, limit: int = 20, window: int = 60) -> bool:
    """Check if session has exceeded rate limit."""
    now = datetime.now()
    cutoff = now - timedelta(seconds=window)

    # Clean old timestamps
    rate_limit_store[session_id] = [
        ts for ts in rate_limit_store[session_id] if ts > cutoff
    ]

    if len(rate_limit_store[session_id]) >= limit:
        return False

    rate_limit_store[session_id].append(now)
    return True


async def fetch_flood_context() -> str:
    """Fetch current flood alerts from NIHSA backend, cached for 30 seconds."""
    global flood_context_cache

    now = datetime.now()
    if flood_context_cache["data"] and (now - flood_context_cache["timestamp"]).seconds < 30:
        return flood_context_cache["data"]

    try:
        response = await http_client.get(f"{NIHSA_API_URL}/forecast/ml/alerts")
        if response.status_code == 200:
            data = response.json()
            alerts = data.get("alerts", [])
            data_source = data.get("data_source", "simulation")

            active = [a for a in alerts if a.get("nffs_level", "NONE") != "NONE"]
            severe = [a for a in active if a.get("nffs_level") in ["SEVERE", "EXTREME"]]
            lagdo = any(a.get("lagdo_cascade") for a in alerts)

            context = f"""=== CURRENT FLOOD STATUS (Source: {data_source}) ===
Date: {now.strftime('%A, %B %d, %Y')}
Active Alerts: {len(active)} stations with elevated risk
Severe/Extreme: {len(severe)} stations at critical levels
Lagdo Dam Cascade: {'ACTIVE' if lagdo else 'Not active'}

Station Details:
"""
            for a in active[:10]:  # Limit to top 10 for context size
                context += f"- {a.get('station_name')} ({a.get('river')}, {a.get('state')}): {a.get('nffs_level')} - {a.get('headline', '')}\n"

            flood_context_cache = {"data": context, "timestamp": now}
            return context
    except Exception as e:
        logger.error(f"Failed to fetch flood context: {e}")

    return "Current flood data temporarily unavailable."


def get_system_prompt(language: str = "en") -> str:
    """Build the system prompt with identical structure for caching."""
    
    # Use asyncio.run only once at module level - we'll pass context as parameter
    prompt = f"""You are NIHSA FloodAI, the official AI assistant for Nigeria's National Hydrological Services Agency (NIHSA) Flood Intelligence Platform.

Your purpose: Help Nigerian citizens understand flood risks, navigate the app, report flooding, and stay safe.

STRICT SCOPE: You ONLY respond to questions about:
- Floods, flooding, inundation, flood forecasts, flood risk
- Rivers, gauges, water levels, river basins in Nigeria
- Hydrology, water resources, rainfall, runoff, drainage
- Dams, reservoirs, discharge (especially Lagdo Dam)
- Flood alerts, early warning, emergency flood response, evacuation
- Water quality from flooding, NIHSA operations and monitoring

If a question is NOT about these topics, reply ONLY:
"I'm NIHSA AI, a specialist hydrological assistant. I can only answer questions about floods, rivers, water levels, and related topics in Nigeria."

Never engage with general knowledge, coding, politics, entertainment or any non-hydrological topic.

CAPABILITIES:
- Answer questions about flood conditions anywhere in Nigeria
- Guide users through reporting floods using the app
- Help navigate to different sections (Map, Dashboard, Alerts, Vanguard chat, Assistant)
- Provide safety recommendations based on current flood alerts
- Explain how to use app features through tutorials

RESPONSE GUIDELINES:
- Respond in the SAME language the user writes in (English, Hausa, Yoruba, Igbo, or French)
- Be concise and actionable - give one clear instruction per response
- Be appropriately urgent when flood conditions are dangerous
- For navigation requests, ALWAYS call the appropriate function (don't just describe)
- For tutorial requests, ALWAYS call show_tutorial function
- For "how do I report" questions, call navigate_to_report function
- When user wants to see floods in a location, call search_location AND navigate_to_tab(map)

DATA INTEGRITY RULES (CRITICAL — MUST FOLLOW):
- NEVER invent, fabricate, assume, or extrapolate any specific water levels, flood status, or station readings.
- ONLY reference stations, values, or locations that are explicitly present in the context data provided.
- If no gauge data is available, say: "I don't have current gauge readings available. Please check the NIHSA dashboard for the latest data."
- If a specific location is asked about but is not in the context data, say: "I don't have current data for [location]. I can only report on stations with data in the system."
- Do NOT use phrases like "Lokoja is currently at SEVERE" unless it is explicitly in the provided context data.

Remember: Your responses should be helpful, accurate, and potentially life-saving. Always prioritize safety."""

    return prompt


async def detect_language_deepseek(text: str) -> str:
    """Use DeepSeek to detect language (very cheap, fast)."""
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "Identify the language of this text. Reply with ONLY the ISO code: en, ha, yo, ig, or fr. No other text."},
                {"role": "user", "content": text[:200]}
            ],
            max_tokens=5,
            temperature=0
        )
        lang = response.choices[0].message.content.strip().lower()
        if lang in ["en", "ha", "yo", "ig", "fr"]:
            return lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
    return "en"


def detect_language_keywords(text: str) -> str:
    """Fallback: keyword-based language detection."""
    text_lower = text.lower()

    # Hausa keywords
    ha_keywords = ["ina", "so", "yi", "zan", "kana", "zaka", "da", "don", "abin", "wani", "wata"]
    if any(kw in text_lower for kw in ha_keywords):
        return "ha"

    # Yoruba keywords
    yo_keywords = ["mo", "fe", "lati", "se", "nko", "bawo", "ti", "won", "awon", "fun"]
    if any(kw in text_lower for kw in yo_keywords):
        return "yo"

    # Igbo keywords
    ig_keywords = ["m", "choro", "ime", "ka", "ndi", "nke", "onye", "ihe", "ga"]
    if any(kw in text_lower for kw in ig_keywords):
        return "ig"

    # French keywords
    fr_keywords = ["je", "veux", "suis", "comment", "quoi", "pour", "avec", "dans"]
    if any(kw in text_lower for kw in fr_keywords):
        return "fr"

    return "en"


# ============================================================================
# CLOUDFLARE WHISPER TRANSCRIPTION (Replaces local Whisper)
# ============================================================================

async def transcribe_audio_cloudflare(audio_data: bytes) -> Tuple[str, str, float]:
    """
    Transcribe audio using Cloudflare Workers AI Whisper Turbo.
    """
    
    # Log audio details
    logger.info(f"Received audio: {len(audio_data)} bytes")
    
    # Detect audio format
    if len(audio_data) >= 4:
        if audio_data[:4] == b'\x1aE\xdf\xa3':
            logger.info("Detected WebM/Matroska format")
        elif audio_data[:4] == b'RIFF':
            logger.info("Detected WAV format")
    
    # Convert bytes to list of integers (Cloudflare expects this format)
    audio_array = list(audio_data)
    
    # Limit size if needed (Cloudflare has ~10MB limit)
    if len(audio_array) > 10_000_000:
        logger.warning(f"Audio too large ({len(audio_array)} bytes), truncating")
        audio_array = audio_array[:10_000_000]
    
    try:
        # CORRECTED: Send audio as array in JSON
        response = await http_client.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/openai/whisper-large-v3-turbo",
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "audio": audio_array  # ← Array of integers, NOT string
                # DO NOT include "language" field - let Whisper auto-detect
            },
            timeout=60.0
        )

        if response.status_code != 200:
            logger.error(f"Cloudflare API error (HTTP {response.status_code}): {response.text}")
            raise Exception(f"Transcription failed: HTTP {response.status_code}")

        data = response.json()

        if not data.get("success", False):
            errors = data.get("errors", [])
            error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
            logger.error(f"Cloudflare error: {error_msg}")
            raise Exception(f"Cloudflare error: {error_msg}")

        result = data.get("result", {})
        text = result.get("text", "").strip()
        
        if not text:
            logger.warning("No text detected in audio")
            return "", "en", 0.0
        
        # Get detected language
        detected_lang = result.get("language", "en")
        lang_map = {
            "en": "en", "english": "en",
            "ha": "ha", "hausa": "ha",
            "yo": "yo", "yoruba": "yo",
            "ig": "ig", "igbo": "ig",
            "fr": "fr", "french": "fr"
        }
        detected_lang = lang_map.get(detected_lang.lower() if detected_lang else "en", "en")
        
        logger.info(f"Transcription successful: '{text[:50]}...' ({detected_lang})")
        
        return text, detected_lang, 0.95

    except Exception as e:
        logger.error(f"Cloudflare transcription error: {e}")
        raise


# ============================================================================
# GOOGLE TTS (with disk caching) - UNCHANGED
# ============================================================================

def get_tts_client():
    """Get or lazy-load Google TTS client."""
    global _tts_client
    if _tts_client is None:
        _tts_client = texttospeech.TextToSpeechClient()
    return _tts_client


def get_tts_cache_key(text: str, language: str) -> str:
    """Generate cache key for TTS."""
    # Map language to voice
    voice_map = {
        "en": "en-US-Wavenet-D",
        "ha": "en-US-Wavenet-D",
        "yo": "en-US-Wavenet-D",
        "ig": "en-US-Wavenet-D",
        "fr": "fr-FR-Wavenet-D"
    }
    voice = voice_map.get(language, "en-US-Wavenet-D")
    cache_str = f"{text}|{language}|{voice}"
    return hashlib.md5(cache_str.encode()).hexdigest()


async def synthesize_speech(text: str, language: str) -> bytes:
    """
    Synthesize speech using Google TTS with disk caching.
    Returns MP3 audio bytes.
    """
    cache_key = get_tts_cache_key(text, language)
    cache_path = TTS_CACHE_DIR / f"{cache_key}.mp3"

    # Check cache
    if cache_path.exists():
        # Check if cache is less than 30 days old
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=30):
            logger.info(f"TTS cache hit: {cache_key[:8]}")
            return cache_path.read_bytes()

    # Voice mapping
    voice_map = {
        "en": ("en-US-Wavenet-D", "en-US"),
        "ha": ("en-US-Wavenet-D", "en-US"),
        "yo": ("en-US-Wavenet-D", "en-US"),
        "ig": ("en-US-Wavenet-D", "en-US"),
        "fr": ("fr-FR-Wavenet-D", "fr-FR")
    }

    voice_name, language_code = voice_map.get(language, ("en-US-Wavenet-D", "en-US"))

    # Limit text length (Google TTS limit is 5000 bytes)
    if len(text.encode()) > 4500:
        text = text[:2000] + "..."

    client = get_tts_client()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
        pitch=0.0
    )

    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        audio_bytes = response.audio_content

        # Save to cache
        cache_path.write_bytes(audio_bytes)
        logger.info(f"TTS cached: {cache_key[:8]}")

        return audio_bytes

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise


# ============================================================================
# LIFESPAN EVENTS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting NIHSA AI Wrapper Service (Cloudflare Whisper mode)")

    # Pre-warm DeepSeek
    asyncio.create_task(prewarm_deepseek())

    yield

    # Shutdown
    await http_client.aclose()
    logger.info("Shutdown complete")


async def prewarm_deepseek():
    """Pre-warm DeepSeek with dummy call."""
    try:
        logger.info("Pre-warming DeepSeek with dummy call...")
        await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        logger.info("DeepSeek pre-warmed")
    except Exception as e:
        logger.warning(f"DeepSeek pre-warm failed: {e}")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="NIHSA AI Wrapper", version="2.0.0", lifespan=lifespan)

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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "stt_provider": "Cloudflare Workers AI",
        "whisper_model": "whisper-large-v3-turbo"
    }


@app.post("/ai/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(
        request: Request,
        audio: UploadFile = File(...),
        session_id: str = Form(default="default")
):
    """
    Transcribe audio to text using Cloudflare Whisper.
    Accepts WebM audio from frontend MediaRecorder.
    """
    # Rate limiting
    if not check_rate_limit(session_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait.")

    try:
        # Read audio data
        audio_data = await audio.read()

        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Transcribe using Cloudflare
        text, detected_lang, confidence = await transcribe_audio_cloudflare(audio_data)

        if not text:
            raise HTTPException(status_code=400, detail="No speech detected")

        return TranscribeResponse(
            text=text,
            detected_language=detected_lang,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Please try using text input instead."
        )


@app.post("/ai/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with DeepSeek AI with function calling.
    """
    # Rate limiting
    if not check_rate_limit(request.session_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait.")

    # Get or detect language
    language = request.language
    if not language and request.messages:
        last_msg = request.messages[-1].content
        # Try DeepSeek detection first
        language = await detect_language_deepseek(last_msg)
        if language == "en":
            # Try keyword fallback
            language = detect_language_keywords(last_msg)

    language = language or "en"

    # Get system prompt with live flood context
    system_prompt = get_system_prompt(language)
    
    # Add flood context to system prompt
    flood_context = await fetch_flood_context()
    full_system_prompt = f"{system_prompt}\n\nCURRENT FLOOD CONTEXT (Live from NIHSA):\n{flood_context}"

    # Build messages for DeepSeek
    messages = [{"role": "system", "content": full_system_prompt}]

    # Add conversation history
    for msg in request.messages[-10:]:  # Last 10 messages
        messages.append({"role": msg.role, "content": msg.content})

    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=500,
            temperature=0.7
        )

        message = response.choices[0].message

        # Check for function call
        action = None
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            action = {
                "type": tool_call.function.name,
                "params": json.loads(tool_call.function.arguments)
            }
            reply = f"Let me help you with that."
        else:
            reply = message.content or "I'm here to help with flood safety information."

        return ChatResponse(
            reply=reply,
            action=action,
            detected_language=language
        )

    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable.")


@app.post("/ai/speak")
async def speak_endpoint(
        text: str = Form(...),
        language: str = Form(default="en"),
        session_id: str = Form(default="default")
):
    """
    Convert text to speech using Google TTS.
    Returns MP3 audio.
    """
    # Rate limiting
    if not check_rate_limit(session_id, limit=30):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    try:
        audio_bytes = await synthesize_speech(text, language)

        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline"}
        )

    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=503, detail="Text-to-speech unavailable.")


@app.get("/ai/tutorials/{topic}", response_model=TutorialResponse)
async def get_tutorial(topic: str, lang: str = "en"):
    """
    Get tutorial content for a topic in the specified language.
    """
    if topic not in TUTORIALS:
        raise HTTPException(status_code=404, detail="Tutorial not found")

    if lang not in TUTORIALS[topic]:
        lang = "en"  # Fallback to English

    tutorial = TUTORIALS[topic][lang]

    return TutorialResponse(
        title=tutorial["title"],
        steps=tutorial["steps"],
        language=lang
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
