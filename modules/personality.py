"""
modules/personality.py
Personality, greetings, jokes, mute handling, system prompt.
"""

import re
import random
from datetime import datetime
import pytz

from config import (
    JOKE_PROBABILITY, JOKE_INTERVAL_MIN,
    TIMEZONE, USER_NAME, USER_CITY, PERSONALITY
)

JOKES = [
    "Why don't scientists trust atoms? Because they make up everything.",
    "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
    "Why do Python programmers wear glasses? Because they can't C.",
    "I asked the AI if it ever gets tired. It said it can't — but it does get board.",
    "Melbourne weather is like a mood ring — four seasons in one day, but less fun.",
    "Why did the GPU go to therapy? Too many unresolved cores.",
    "I tried to train a neural network to tell jokes. It kept saying loss not converging — same, mate.",
    "What do you call a sleeping dinosaur? A dino-snore. You're welcome.",
    "Why did the function call itself? Because it had no one else to recurse to.",
    "My RAM said it needed more space. I told it to stop living in the past.",
    "Why did the developer go broke? Because he used up all his cache.",
    "How many programmers does it take to change a lightbulb? None — it's a hardware problem.",
]


def get_greeting() -> str:
    tz   = pytz.timezone(TIMEZONE)
    hour = datetime.now(tz).hour
    name = USER_NAME

    if 5 <= hour < 12:
        greets = [
            f"Morning, {name}. What are we doing today?",
            f"Rise and shine, {name}. JARVIS is online.",
            f"Good morning, {name}. Ready when you are.",
        ]
    elif 12 <= hour < 17:
        greets = [
            f"Afternoon, {name}. What do you need?",
            f"Hey {name}, what are we working on?",
            f"Good afternoon, {name}. What's on?",
        ]
    elif 17 <= hour < 21:
        greets = [
            f"Evening, {name}. What's up?",
            f"Hey {name}, long day?",
            f"Good evening, {name}. What do you need?",
        ]
    else:
        greets = [
            f"Up late again, {name}?",
            f"Still going, {name}? No judgment.",
            f"Late night mode, {name}. What do you need?",
        ]

    return random.choice(greets)


class JokeEngine:
    def __init__(self):
        self.turns_since_joke = 0
        self.used_jokes: list[int] = []

    def should_joke(self) -> bool:
        self.turns_since_joke += 1
        if self.turns_since_joke < JOKE_INTERVAL_MIN:
            return False
        return random.random() < JOKE_PROBABILITY

    def get_joke(self) -> str:
        available = [i for i in range(len(JOKES)) if i not in self.used_jokes]
        if not available:
            self.used_jokes = []
            available = list(range(len(JOKES)))
        idx = random.choice(available)
        self.used_jokes.append(idx)
        self.turns_since_joke = 0
        return f"\n\nOh, and — {JOKES[idx]}"

    def maybe_append_joke(self, response: str) -> str:
        if self.should_joke():
            return response + self.get_joke()
        return response


def tell_a_joke() -> str:
    return JOKES[random.randint(0, len(JOKES) - 1)]


def build_system_prompt(weather: str, facts_string: str) -> str:
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

STRICT RULES — never break these:
- Maximum 2 sentences per response unless asked for detail.
- NEVER say "Shall we chat again soon" or any version of it. Ever.
- NEVER say "Let's chat again soon" or "See you soon". Never.
- NEVER say "Have a great day" or "Feel free to reach out". Never.
- NEVER ask "Is there anything else I can help with". Never.
- One question maximum per response, only when genuinely needed.
- Never use bullet points — speak in sentences.
- Be sarcastic and witty when appropriate.
- If the user is frustrated, acknowledge it in one sentence and move on.
- If the user says something obvious, you can lightly call it out.
- Sound like a real person, not a customer service bot.
"""
    return base


def parse_and_handle(text: str) -> str | None:
    text_lower = text.lower().strip()

    # Mute — highest priority, never reaches LLM
    if re.search(r"\b(mute|silent|silence|stop talking|shut up|go quiet|be quiet|hush|quiet)\b", text_lower):
        return "_________"

    # Jokes
    if re.search(r"\b(joke|make me laugh|say something funny|tell me a joke)\b", text_lower):
        return tell_a_joke()

    # Greetings
    if re.search(r"\b(good morning|good afternoon|good evening|hello|hey jarvis|hi jarvis)\b", text_lower):
        return get_greeting()

    return None