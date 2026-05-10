"""
modules/personality.py
Makes JARVIS feel human — jokes, casual language, personality injection.
Grows more like you over time as memory accumulates.
"""

import re
import random
from datetime import datetime
import pytz

from config import (
    JOKE_PROBABILITY, JOKE_INTERVAL_MIN,
    TIMEZONE, USER_NAME, USER_CITY, PERSONALITY
)


# ── Joke bank — add your own! ──────────────────────────────────────────────────
JOKES = [
    "Why don't scientists trust atoms? Because they make up everything.",
    "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
    "Why do Python programmers wear glasses? Because they can't C.",
    "I asked the AI if it ever gets tired. It said it can't — but it does get board.",
    "Melbourne weather is like a mood ring — four seasons in one day, but less fun.",
    "Why did the GPU go to therapy? Too many unresolved cores.",
    "I tried to train a neural network to tell jokes. It kept saying 'loss not converging' — same, mate.",
    "What do you call a sleeping dinosaur? A dino-snore. You're welcome.",
    "Why did the function call itself? Because it had no one else to recurse to.",
    "My RAM said it needed more space. I told it to stop living in the past.",
    "Why did the developer go broke? Because he used up all his cache.",
    "I'm reading a book about anti-gravity. It's impossible to put down — unlike some of your code.",
    "How many programmers does it take to change a lightbulb? None — it's a hardware problem.",
]

# ── Greetings by time of day ───────────────────────────────────────────────────
def get_greeting() -> str:
    tz   = pytz.timezone(TIMEZONE)
    hour = datetime.now(tz).hour
    name = USER_NAME

    if 5 <= hour < 12:
        greets = [
            f"Good morning, {name}! Ready to make today count?",
            f"Morning, {name}! Melbourne's already awake — let's go.",
            f"Rise and shine, {name}. JARVIS is online.",
        ]
    elif 12 <= hour < 17:
        greets = [
            f"Good afternoon, {name}. How's the day treating you?",
            f"Afternoon, {name}! Half the day done — what do you need?",
            f"Hey {name}, good afternoon. What are we working on?",
        ]
    elif 17 <= hour < 21:
        greets = [
            f"Good evening, {name}. Winding down or just getting started?",
            f"Evening, {name}! Long day? I'm here.",
            f"Hey {name}, evening. What's on your mind?",
        ]
    else:
        greets = [
            f"Up late again, {name}? I've got you.",
            f"Still going, {name}? I never sleep, so no judgment.",
            f"Late night mode activated, {name}. What do you need?",
        ]

    return random.choice(greets)


# ── Joke injection ────────────────────────────────────────────────────────────

class JokeEngine:
    def __init__(self):
        self.turns_since_joke = 0
        self.used_jokes: list[int] = []

    def should_joke(self) -> bool:
        """Returns True if JARVIS should tell a joke after this response."""
        self.turns_since_joke += 1
        if self.turns_since_joke < JOKE_INTERVAL_MIN:
            return False
        return random.random() < JOKE_PROBABILITY

    def get_joke(self) -> str:
        """Get a random joke, avoiding repeats."""
        available = [i for i in range(len(JOKES)) if i not in self.used_jokes]
        if not available:
            self.used_jokes = []   # reset when all used
            available = list(range(len(JOKES)))

        idx = random.choice(available)
        self.used_jokes.append(idx)
        self.turns_since_joke = 0
        return f"\n\nOh, and — {JOKES[idx]}"

    def maybe_append_joke(self, response: str) -> str:
        """Optionally append a joke to an existing response."""
        if self.should_joke():
            return response + self.get_joke()
        return response


# ── Explicit joke request ─────────────────────────────────────────────────────

def tell_a_joke() -> str:
    idx = random.randint(0, len(JOKES) - 1)
    return JOKES[idx]


# ── Build system prompt ───────────────────────────────────────────────────────

def build_system_prompt(weather: str, facts_string: str) -> str:
    """
    Builds the full system prompt injected into every LLM call.
    Includes personality, time, weather, and remembered facts.
    """
    tz  = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    dt  = now.strftime("%A, %d %B %Y — %I:%M %p AEST")

    base = PERSONALITY.format(
        user_name = USER_NAME,
        city      = USER_CITY,
        country   = "Australia",
        datetime  = dt,
        weather   = weather,
    )

    if facts_string:
        base += f"\n\n{facts_string}"

    base += """

Conversation style rules:
- Speak like a real human, not a manual. Use contractions.
- Keep it short unless asked for detail — 1 to 3 sentences is ideal.
- If you don't know something, say so honestly.
- Refer back to past things naturally when relevant.
- Occasionally (not always) add a bit of warmth or wit.
- Never use bullet points in spoken responses — speak in sentences.
- If user seems stressed or frustrated, acknowledge it first.
"""
    return base


# ── Intent parsing ────────────────────────────────────────────────────────────

def parse_and_handle(text: str) -> str | None:
    text_lower = text.lower().strip()

    if re.search(r"\b(joke|make me laugh|say something funny|funny)\b", text_lower):
        return tell_a_joke()

    if re.search(r"\b(good morning|good afternoon|good evening|hello|hey|hi jarvis)\b", text_lower):
        return get_greeting()

    return None
