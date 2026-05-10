"""
modules/realtime.py
Real-time data: weather, time/date (Melbourne), news, web search.
No API keys needed — uses free public sources.
"""

import re
import json
import urllib.request
import urllib.parse
import webbrowser
from datetime import datetime
import pytz

from config import TIMEZONE, USER_COUNTRY, WEATHER_CITY, USER_CITY


# ── Time & Date ────────────────────────────────────────────────────────────────

def get_melbourne_time() -> datetime:
    """Always returns current Melbourne time, regardless of system clock timezone."""
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz)


def get_datetime_string() -> str:
    """Human-readable Melbourne date and time."""
    now = get_melbourne_time()
    return now.strftime("%A, %d %B %Y — %I:%M %p AEST")


def get_time_response() -> str:
    now = get_melbourne_time()
    return f"It's {now.strftime('%I:%M %p')} here in Melbourne — {now.strftime('%A, %d %B %Y')}."


# ── Weather ────────────────────────────────────────────────────────────────────

def get_weather() -> str:
    """
    Fetches Melbourne weather from wttr.in — completely free, no API key.
    Returns a short description like: "Partly cloudy +18°C 65% 15km/h"
    """
    try:
        url = f"https://wttr.in/{WEATHER_CITY}?format=%C+%t+Humidity:%h+Wind:%w"
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8").strip()
        return raw
    except Exception:
        return "Weather unavailable right now"


def get_weather_response() -> str:
    weather = get_weather()
    now = get_melbourne_time()
    hour = now.hour
    if hour < 12:
        time_part = "this morning"
    elif hour < 17:
        time_part = "this afternoon"
    else:
        time_part = "this evening"
    return f"In Melbourne {time_part}: {weather}."


def get_full_weather() -> str:
    """Detailed weather for when user asks for full forecast."""
    try:
        url = f"https://wttr.in/{WEATHER_CITY}?format=3"
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return "Couldn't fetch weather right now."


# ── Web Search (DuckDuckGo, no API key) ───────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> str:
    """
    Searches DuckDuckGo Instant Answers API — free, no key needed.
    Returns a short summary of the top result.
    """
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Try abstract (Wikipedia-style answer)
        if data.get("AbstractText"):
            return data["AbstractText"][:400]

        # Try answer (direct factual)
        if data.get("Answer"):
            return data["Answer"]

        # Try related topics
        topics = data.get("RelatedTopics", [])
        if topics and isinstance(topics[0], dict):
            return topics[0].get("Text", "")[:300]

        return f"I searched for '{query}' but couldn't find a clear answer."
    except Exception as e:
        return f"Search failed: {e}"


# ── System info ───────────────────────────────────────────────────────────────

def get_system_stats() -> str:
    """Returns GPU/CPU/RAM stats."""
    import subprocess
    import torch

    parts = []

    # GPU via PyTorch
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0)  / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        name  = torch.cuda.get_device_name(0)
        parts.append(f"GPU: {name} — {used:.1f}/{total:.0f} GB VRAM used")

    # nvidia-smi for temp + utilisation
    try:
        out = subprocess.run(
            "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader",
            capture_output=True, text=True, shell=True, timeout=3
        ).stdout.strip()
        if out:
            temp, util = out.split(", ")
            parts.append(f"GPU temp: {temp}°C, utilisation: {util}")
    except Exception:
        pass

    # RAM
    try:
        out = subprocess.run(
            'wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value',
            capture_output=True, text=True, shell=True, timeout=3
        ).stdout
        lines = {l.split("=")[0]: l.split("=")[1] for l in out.strip().splitlines() if "=" in l}
        free  = int(lines.get("FreePhysicalMemory", 0))  / 1024**2
        total = int(lines.get("TotalVisibleMemorySize", 0)) / 1024**2
        parts.append(f"RAM: {total - free:.1f}/{total:.0f} GB used")
    except Exception:
        pass

    return "\n".join(parts) if parts else "System stats unavailable."


# ── Location (IP-based, only called when needed) ───────────────────────────────

def get_location() -> dict:
    """
    Auto-detects city, country, timezone from IP address.
    Free, no API key. Falls back to config values if offline.
    Only hits the internet when explicitly called.
    """
    try:
        url = "http://ip-api.com/json/?fields=city,country,timezone,lat,lon"
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "city":     data.get("city",     WEATHER_CITY),
            "country":  data.get("country",  USER_COUNTRY),
            "timezone": data.get("timezone", TIMEZONE),
            "lat":      data.get("lat",      0),
            "lon":      data.get("lon",      0),
        }
    except Exception:
        # Offline fallback — use config values
        return {
            "city":     WEATHER_CITY,
            "country":  USER_COUNTRY,
            "timezone": TIMEZONE,
            "lat":      0,
            "lon":      0,
        }


# ── Nearby Places Search ───────────────────────────────────────────────────────

def find_nearby(query: str) -> str:
    """
    Opens Google Maps in the browser searching for query near current location.
    Only hits the internet when called — never runs in the background.
    Falls back gracefully if location detection fails.
    """
    try:
        loc  = get_location()
        city = loc["city"]
        lat  = loc["lat"]
        lon  = loc["lon"]

        # If we got real coordinates, use them for more accurate results
        if lat and lon:
            encoded = urllib.parse.quote(query)
            url = (
                f"https://www.google.com/maps/search/{encoded}"
                f"/@{lat},{lon},14z"
            )
        else:
            # Fall back to city name search
            encoded = urllib.parse.quote(f"{query} near {city}")
            url = f"https://www.google.com/maps/search/{encoded}"

        webbrowser.open(url)
        return f"Opening Google Maps for {query} near {city}."

    except Exception:
        # Last resort — open Maps with just the query
        encoded = urllib.parse.quote(query)
        webbrowser.open(f"https://www.google.com/maps/search/{encoded}")
        return f"Opening Google Maps for {query}."


# ── Intent router ─────────────────────────────────────────────────────────────

def parse_and_handle(text: str) -> str | None:
    text_lower = text.lower().strip()

    # Time / date
    if re.search(r"\b(time|date|day|today|what day|what time)\b", text_lower):
        if re.search(r"\b(weather|temperature|temp|forecast)\b", text_lower):
            pass  # fall through to weather
        else:
            return get_time_response()

    # Weather
    if re.search(r"\b(weather|temperature|temp|forecast|raining|hot|cold|sunny)\b", text_lower):
        if "forecast" in text_lower or "week" in text_lower:
            return get_full_weather()
        return get_weather_response()

    # System stats
    if re.search(r"\b(gpu|cpu|ram|memory|vram|stats|system info|performance)\b", text_lower):
        return get_system_stats()

    # Nearby places — must come BEFORE generic web search
    # catches: "find a coffee bar near me", "any restaurants nearby",
    #          "where's a pharmacy close by", "look for a gym around here"
    m = re.search(
        r"(?:find|search for|look for|where(?:'s| is)|any|get me)\s+(.+?)\s+"
        r"(?:near me|nearby|close by|around here|near here)",
        text_lower
    )
    if m:
        return find_nearby(m.group(1).strip())

    # Web search
    m = re.search(r"(?:search|look up|google|find|what is|who is|tell me about)\s+(.+)", text_lower)
    if m:
        query = m.group(1).strip()
        result = web_search(query)
        return result if result else None

    return None