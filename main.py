"""
GeoPulse Risk API — Backend
Deployable on Railway, Render, or any Docker host.
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import anthropic
import asyncio
import hashlib
import json
import os
import time

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GeoPulse Risk API",
    description="Real-time AI-powered geopolitical risk scoring",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
API_KEYS = set(os.getenv("VALID_API_KEYS", "demo-key-123").split(","))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "3600"))  # 1h cache

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Simple in-memory cache + rate limiting
cache: dict = {}
rate_limits: dict = {}  # api_key -> list of timestamps

# ─── Rate Limiter ─────────────────────────────────────────────────────────────

RATE_LIMITS = {
    "free": 10,      # requests per day
    "pro": 1000,
    "enterprise": 999999,
}

DEMO_KEY_TIER = {
    "demo-key-123": "free",
}

def get_tier(api_key: str) -> str:
    return DEMO_KEY_TIER.get(api_key, "pro")

def check_rate_limit(api_key: str):
    tier = get_tier(api_key)
    limit = RATE_LIMITS[tier]
    now = time.time()
    day_ago = now - 86400

    if api_key not in rate_limits:
        rate_limits[api_key] = []

    rate_limits[api_key] = [t for t in rate_limits[api_key] if t > day_ago]

    if len(rate_limits[api_key]) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Tier '{tier}' allows {limit} requests/day."
        )

    rate_limits[api_key].append(now)

# ─── Auth ─────────────────────────────────────────────────────────────────────

def require_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    check_rate_limit(x_api_key)
    return x_api_key

# ─── AI Engine ────────────────────────────────────────────────────────────────

RISK_PROMPT = """You are a geopolitical risk intelligence engine used by financial institutions and corporations.

Analyze the current geopolitical risk for: {country}

Respond ONLY with a valid JSON object. No markdown, no explanation outside JSON:
{{
  "country": "{country}",
  "overall_score": <integer 0-100>,
  "risk_level": <"LOW"|"MODERATE"|"ELEVATED"|"HIGH"|"CRITICAL">,
  "dimensions": {{
    "armed_conflict": <0-100>,
    "political_instability": <0-100>,
    "economic_collapse": <0-100>,
    "sanctions_exposure": <0-100>,
    "terrorism": <0-100>,
    "cyber_threat": <0-100>
  }},
  "key_triggers": [<string>, <string>, <string>],
  "trend": <"DETERIORATING"|"STABLE"|"IMPROVING">,
  "investment_impact": "<one sentence>",
  "outlook_30_days": "<one sentence>",
  "confidence": <0-100>,
  "timestamp": "{timestamp}"
}}"""

async def score_country(country: str) -> dict:
    cache_key = hashlib.md5(f"{country}{datetime.utcnow().strftime('%Y-%m-%d-%H')}".encode()).hexdigest()

    if cache_key in cache:
        cached = cache[cache_key]
        if time.time() - cached["_cached_at"] < CACHE_TTL_SECONDS:
            return {**cached["data"], "cached": True}

    prompt = RISK_PROMPT.format(
        country=country,
        timestamp=datetime.utcnow().isoformat()
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    cache[cache_key] = {"data": data, "_cached_at": time.time()}
    return {**data, "cached": False}

# ─── Models ───────────────────────────────────────────────────────────────────

class BatchRequest(BaseModel):
    countries: list[str]

class CustomRequest(BaseModel):
    country: str
    focus: Optional[str] = None  # e.g. "energy sector", "supply chain"

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api": "GeoPulse Risk API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "GET  /risk/{country}",
            "POST /risk/batch",
            "GET  /risk/trending",
            "GET  /health",
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/risk/{country}")
async def get_risk(country: str, api_key: str = Depends(require_api_key)):
    """Get risk score for a single country."""
    if len(country) < 2 or len(country) > 60:
        raise HTTPException(400, "Invalid country name.")
    try:
        result = await score_country(country.strip().title())
        return result
    except json.JSONDecodeError:
        raise HTTPException(500, "AI response parsing failed.")
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.post("/risk/batch")
async def get_risk_batch(body: BatchRequest, api_key: str = Depends(require_api_key)):
    """Score up to 10 countries in parallel."""
    if len(body.countries) > 10:
        raise HTTPException(400, "Maximum 10 countries per batch request.")
    if len(body.countries) == 0:
        raise HTTPException(400, "Provide at least one country.")

    results = await asyncio.gather(
        *[score_country(c.strip().title()) for c in body.countries],
        return_exceptions=True
    )

    output = []
    for country, result in zip(body.countries, results):
        if isinstance(result, Exception):
            output.append({"country": country, "error": str(result)})
        else:
            output.append(result)

    return {"results": output, "count": len(output)}

@app.get("/risk/global/trending")
async def get_trending(api_key: str = Depends(require_api_key)):
    """Returns top 5 highest-risk countries right now (uses cache when available)."""
    hotspots = ["Ukraine", "Russia", "Iran", "Sudan", "Myanmar", "Haiti", "North Korea", "Israel"]

    results = await asyncio.gather(
        *[score_country(c) for c in hotspots],
        return_exceptions=True
    )

    scored = []
    for country, r in zip(hotspots, results):
        if not isinstance(r, Exception):
            scored.append(r)

    scored.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    return {"trending": scored[:5], "as_of": datetime.utcnow().isoformat()}

@app.get("/usage")
def usage(api_key: str = Depends(require_api_key)):
    """Returns current usage for the API key."""
    tier = get_tier(api_key)
    used = len(rate_limits.get(api_key, []))
    limit = RATE_LIMITS[tier]
    return {
        "api_key": api_key[:8] + "...",
        "tier": tier,
        "used_today": used,
        "limit_today": limit,
        "remaining": max(0, limit - used),
    }

