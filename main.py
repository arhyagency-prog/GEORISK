"""
GeoPulse Risk API — Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import anthropic
import asyncio
import hashlib
import json
import os
import time

app = FastAPI(
    title="GeoPulse Risk API",
    description="Real-time AI-powered geopolitical risk scoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
cache: dict = {}

PROMPT = """You are a geopolitical risk intelligence engine.
Analyze the current geopolitical risk for: {country}

Respond ONLY with a raw JSON object, no markdown, no backticks:
{{
  "country": "{country}",
  "overall_score": <integer 0-100>,
  "risk_level": "LOW|MODERATE|ELEVATED|HIGH|CRITICAL",
  "dimensions": {{
    "armed_conflict": <0-100>,
    "political_instability": <0-100>,
    "economic_collapse": <0-100>,
    "sanctions_exposure": <0-100>,
    "terrorism": <0-100>,
    "cyber_threat": <0-100>
  }},
  "key_triggers": ["<trigger 1>", "<trigger 2>", "<trigger 3>"],
  "trend": "DETERIORATING|STABLE|IMPROVING",
  "investment_impact": "<one sentence>",
  "outlook_30_days": "<one sentence>",
  "confidence": <0-100>,
  "timestamp": "{timestamp}"
}}"""

async def score_country(country: str) -> dict:
    cache_key = hashlib.md5(
        f"{country}{datetime.utcnow().strftime('%Y-%m-%d-%H')}".encode()
    ).hexdigest()

    if cache_key in cache:
        cached = cache[cache_key]
        if time.time() - cached["_cached_at"] < CACHE_TTL:
            return {**cached["data"], "cached": True}

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": PROMPT.format(
                country=country,
                timestamp=datetime.utcnow().isoformat()
            )
        }]
    )

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    cache[cache_key] = {"data": data, "_cached_at": time.time()}
    return {**data, "cached": False}


class BatchRequest(BaseModel):
    countries: list[str]


@app.get("/")
def root():
    return {
        "api": "GeoPulse Risk API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "GET  /risk/{country}",
            "POST /risk/batch",
            "GET  /risk/global/trending",
            "GET  /health",
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/risk/{country}")
async def get_risk(country: str):
    if len(country) < 2 or len(country) > 60:
        raise HTTPException(400, "Invalid country name.")
    try:
        return await score_country(country.strip().title())
    except json.JSONDecodeError:
        raise HTTPException(500, "AI response parsing failed.")
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.post("/risk/batch")
async def get_risk_batch(body: BatchRequest):
    if len(body.countries) > 10:
        raise HTTPException(400, "Maximum 10 countries per batch.")
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
async def get_trending():
    hotspots = ["Ukraine", "Russia", "Iran", "Sudan", "Myanmar", "Haiti", "North Korea", "Israel"]
    results = await asyncio.gather(
        *[score_country(c) for c in hotspots],
        return_exceptions=True
    )
    scored = [r for r in results if not isinstance(r, Exception)]
    scored.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    return {"trending": scored[:5], "as_of": datetime.utcnow().isoformat()}
