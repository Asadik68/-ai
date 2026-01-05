import os
import json
import re
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai


# 1. ENV & LOGGING

load_dotenv()

OCLC_CLIENT_ID = os.getenv("OCLC_CLIENT_ID")
OCLC_CLIENT_SECRET = os.getenv("OCLC_CLIENT_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OCLC_SYMBOL = "UAE"
BASE_URL_CI = "https://discovery.api.oclc.org/worldcat-org-ci/search"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uaeu-library-ai")


# 2. GEMINI SETUP

GEMINI_MODEL = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    logger.info("Gemini AI enabled")
else:
    logger.warning("Gemini API key missing")


# 3. FASTAPI APP

app = FastAPI(title="UAEU Library AI Assistant (CI Async)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. MODELS

class AISearchRequest(BaseModel):
    query: str
    limit: int = 5


class Book(BaseModel):
    title: str
    author: str
    format: str
    call_number: Optional[str]
    year: str
    link: str


class AISearchResponse(BaseModel):
    ai_response: str
    books: List[Book]


# 5. OCLC TOKEN MANAGER

_token_cache = {"token": None, "expires_at": datetime.utcnow()}
_token_lock = asyncio.Lock()


async def get_oclc_access_token() -> Optional[str]:
    async with _token_lock:
        if (
            _token_cache["token"]
            and datetime.utcnow() < _token_cache["expires_at"]
        ):
            return _token_cache["token"]

        url = "https://oauth.oclc.org/token"
        data = {
            "grant_type": "client_credentials",
            "scope": "WorldCatDiscoveryAPI"
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    url,
                    data=data,
                    auth=(OCLC_CLIENT_ID, OCLC_CLIENT_SECRET),
                )
                r.raise_for_status()
                payload = r.json()

            _token_cache["token"] = payload["access_token"]
            expires = int(payload.get("expires_in", 1200))
            _token_cache["expires_at"] = datetime.utcnow() + timedelta(seconds=expires - 60)

            logger.info("OCLC token refreshed")
            return _token_cache["token"]

        except Exception:
            logger.exception("OCLC authentication failed")
            return None


# 6. SPELLING NORMALIZATION

async def normalize_query(query: str) -> str:
    if not GEMINI_MODEL:
        return query

    prompt = (
        "Correct spelling only. "
        "Do not add or remove words. "
        "Return the same language.\n\n"
        f"Query: \"{query}\""
    )

    try:
        result = GEMINI_MODEL.generate_content(prompt)
        corrected = result.text.strip().replace("\n", " ")

        if len(corrected.split()) > len(query.split()) + 2:
            return query

        if corrected.lower() != query.lower():
            logger.info("Spelling corrected: %s → %s", query, corrected)

        return corrected
    except Exception:
        logger.exception("Spelling normalization failed")
        return query

# ============================
# 7. CI SEARCH
# ============================
async def search_ci_brief(query: str, limit: int) -> List[Dict[str, Any]]:
    token = await get_oclc_access_token()
    if not token:
        return []

    query = await normalize_query(query)

    url = f"{BASE_URL_CI}/brief-bibs"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {"q": query, "heldBySymbol": OCLC_SYMBOL, "limit": limit}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
            return r.json().get("briefRecords", [])
    except Exception:
        logger.exception("CI search failed")
        return []


# 8. CI DETAILS (TOP 3 ONLY)

async def get_ci_details(oclc_number: str) -> Dict[str, Any]:
    token = await get_oclc_access_token()
    if not token:
        return {}

    url = f"{BASE_URL_CI}/bibs-detailed-holdings"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {"oclcNumber": oclc_number, "heldBySymbol": OCLC_SYMBOL}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers, params=params)
            if r.status_code != 200:
                return {}

        data = r.json()
        records = data.get("briefRecords", [])
        if not records:
            return {}

        holdings = records[0].get("institutionHolding", {}).get("detailedHoldings", [])
        for h in holdings:
            if "callNumber" in h:
                cn = h["callNumber"]

                # ✅ normalize to string
                if isinstance(cn, dict):
                    return {"call_number": cn.get("displayCallNumber")}
                if isinstance(cn, str):
                    return {"call_number": cn}

        return {}
    except Exception:
        logger.exception("CI details fetch failed")
        return {}

# ============================
# 9. PROCESS BOOK
# ============================
async def process_book(b: Dict[str, Any], fetch_details: bool) -> Book:
    oclc = b.get("oclcNumber")
    details = await get_ci_details(oclc) if fetch_details and oclc else {}

    fmt = b.get("specificFormat", "Unknown")
    is_digital = any(k in fmt.lower() for k in ("ebook", "digital", "online"))

    return Book(
        title=b.get("title", "Untitled"),
        author=b.get("creator", "Unknown"),
        format="eBook" if is_digital else "Printed Book",
        call_number=None if is_digital else (
            str(details.get("call_number")) if details.get("call_number") else None
        )
        ,
        year=b.get("date", ""),
        link=f"https://uaeu.on.worldcat.org/search?queryString={oclc}" if oclc else ""
    )

# ============================
# 10. GEMINI RESPONSE
# ============================
def ask_gemini(user_query: str, books: List[Book]) -> str:
    if not GEMINI_MODEL:
        return "الخدمة غير متوفرة حالياً."

    arabic = bool(re.search(r"[\u0600-\u06FF]", user_query))

    if arabic:
        language_rule = "اكتب الإجابة باللغة العربية فقط، وباتجاه من اليمين إلى اليسار (RTL)."
        format_block = """
الصيغة المطلوبة (اتجاه من اليمين إلى اليسار):

العنوان:
المؤلف:
النوع:
الموقع:
الرابط:
"""
    else:
        language_rule = "Respond in ENGLISH only (Left to Right)."
        format_block = """
FORMAT (Left to Right):

Title:
Author:
Format:
Location:
Link:
"""

    books_json = json.dumps(
        [b.dict() for b in books],
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
أنت أمين مكتبة محترف في مكتبة جامعة الإمارات.
اتبع التعليمات بدقة ولا تضف أي نص غير مطلوب.

قاعدة اللغة والاتجاه:
{language_rule}

سؤال المستخدم:
\"\"\"{user_query}\"\"\"

الكتب المتوفرة:
{books_json}

قواعد الإخراج (مهمة جداً):
1. استخدم نفس لغة واتجاه سؤال المستخدم.
2. لا تضف مقدمات أو عبارات ختامية.
3. لا تعتذر.
4. لا تطلب من المستخدم البحث بكلمات أخرى.
5. اعرض الكتب فقط.
6. استخدم الصيغة التالية حرفياً وبنفس الاتجاه.

{format_block}

ضع سطرًا فارغًا بين كل كتاب.
"""

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception:
        logger.exception("Gemini generation failed")
        return "حدث خطأ أثناء معالجة الطلب."

# ============================
# 11. ENDPOINTS
# ============================
@app.get("/")
def home():
    return {"status": "running", "symbol": OCLC_SYMBOL}


@app.post("/ai-search", response_model=AISearchResponse)
async def ai_search(req: AISearchRequest):
    raw = await search_ci_brief(req.query, req.limit)

    books: List[Book] = []
    for i, b in enumerate(raw):
        books.append(await process_book(b, fetch_details=i < 3))

    ai_text = ask_gemini(req.query, books)
    return AISearchResponse(ai_response=ai_text, books=books)
