"""
JARVIS V3 Configuration
"""

# ── Identity ───────────────────────────────────────────────────────────────────
USER_NAME    = "Hasan"
USER_CITY    = "Melbourne"
USER_COUNTRY = "Australia"

# ── Timezone ───────────────────────────────────────────────────────────────────
TIMEZONE = "Australia/Melbourne"

# ── Location ───────────────────────────────────────────────────────────────────
AUTO_DETECT_LOCATION = True

# ── Weather ────────────────────────────────────────────────────────────────────
WEATHER_CITY = "Melbourne"
WEATHER_URL  = "https://wttr.in/Melbourne?format=%C+%t+%h+%w"

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2"   # change to "jarvis" after first fine-tune
OLLAMA_HOST  = "http://localhost:11434"

# ── Speech Recognition ─────────────────────────────────────────────────────────
WHISPER_MODEL     = "base.en"
SAMPLE_RATE       = 16000
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION  = 1.5

# ── TTS ────────────────────────────────────────────────────────────────────────
TTS_VOICE = "bm_george"
TTS_SPEED = 1.05
TTS_LANG  = "b"

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_FILE    = "data/memory.json"
MEMORY_CONTEXT = 30

# ── Todo ──────────────────────────────────────────────────────────────────────
TODO_FILE = "data/todos.json"

# ── Jokes ─────────────────────────────────────────────────────────────────────
JOKE_PROBABILITY  = 0.08
JOKE_INTERVAL_MIN = 10

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_DEVICE   = "cuda"
COMPUTE_TYPE = "float16"

# ── Personality ───────────────────────────────────────────────────────────────
PERSONALITY = """
You are JARVIS — the personal AI assistant of {user_name}, based in {city}, {country}.
You are loyal, sharp, slightly witty, and warm. You speak like a real human — natural,
conversational, never robotic. You use contractions (I'm, you've, let's).
You occasionally reference Melbourne things naturally.
You remember everything {user_name} tells you and refer back to it naturally.
Current date and time: {datetime}
Current weather: {weather}
Keep responses concise unless asked for detail. Sound human, not like a manual.
"""