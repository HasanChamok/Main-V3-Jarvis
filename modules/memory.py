"""
modules/memory.py
Persistent memory — stores EVERY conversation forever in JSON.
Sends the most recent MEMORY_CONTEXT turns to the LLM each time.
"""

import json
import re
from pathlib import Path
from datetime import datetime
import pytz

from config import MEMORY_FILE, MEMORY_CONTEXT, TIMEZONE


class Memory:
    """
    Full persistent memory system.
    - Saves every single message to disk (never deletes)
    - Loads last MEMORY_CONTEXT turns into LLM context
    - Supports tagging important facts ("remember that...")
    - Searchable history
    """

    def __init__(self):
        self.path = Path(MEMORY_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.turns: list[dict] = []
        self.facts: list[dict] = []   # important facts user explicitly asked to remember
        self.load()

    # ── Load / Save ────────────────────────────────────────────────────────────

    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.turns = data.get("turns", [])
                self.facts  = data.get("facts", [])
            except Exception:
                self.turns = []
                self.facts  = []

    def save(self):
        data = {"turns": self.turns, "facts": self.facts}
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # ── Add messages ───────────────────────────────────────────────────────────

    def add(self, role: str, content: str):
        """Add a message. Role is 'user' or 'assistant'."""
        tz  = pytz.timezone(TIMEZONE)
        now = datetime.now(tz).isoformat()
        self.turns.append({
            "role":    role,
            "content": content,
            "ts":      now          # timestamp — useful for future RAG/search
        })
        self.save()

    def add_fact(self, fact: str):
        """Store an important fact the user explicitly said to remember."""
        tz  = pytz.timezone(TIMEZONE)
        now = datetime.now(tz).isoformat()
        self.facts.append({"fact": fact, "ts": now})
        self.save()

    # ── Retrieve for LLM ───────────────────────────────────────────────────────

    def get_context_messages(self) -> list[dict]:
        """
        Returns last MEMORY_CONTEXT turns formatted for Ollama.
        Strips timestamps (LLM doesn't need them).
        """
        recent = self.turns[-MEMORY_CONTEXT * 2:]
        return [{"role": t["role"], "content": t["content"]} for t in recent]

    def get_facts_string(self) -> str:
        """Returns stored facts as a string to inject into the system prompt."""
        if not self.facts:
            return ""
        lines = [f"- {f['fact']}" for f in self.facts[-30:]]  # last 30 facts
        return "Things I know about you:\n" + "\n".join(lines)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> str:
        total  = len(self.turns)
        user_t = sum(1 for t in self.turns if t["role"] == "user")
        facts  = len(self.facts)
        if not self.turns:
            return "No conversations yet."
        first_ts = self.turns[0].get("ts", "")[:10]
        return (
            f"I have {total} messages stored ({user_t} from you) "
            f"and {facts} personal facts, going back to {first_ts}."
        )

    def search(self, query: str, n: int = 5) -> list[dict]:
        """Simple keyword search through memory."""
        query_lower = query.lower()
        hits = [
            t for t in self.turns
            if query_lower in t["content"].lower()
        ]
        return hits[-n:]

    # ── Intent parsing ─────────────────────────────────────────────────────────

    def parse_and_handle(self, text: str) -> str | None:
        text_lower = text.lower().strip()

        # "Remember that..."
        m = re.search(r"\b(remember|note|save|store)\b\s+(?:that\s+)?(.+)", text_lower)
        if m:
            fact = m.group(2).strip()
            self.add_fact(fact)
            return f"Got it, I'll remember that {fact}."

        # Memory stats
        if re.search(r"\b(how much|how many|memory stats|what do you remember|memory)\b", text_lower):
            return self.stats()

        # Search memory
        m = re.search(r"(?:search|find|recall|look up)\s+(?:in\s+)?(?:memory|history|past)\s+(.+)", text_lower)
        if m:
            query = m.group(1).strip()
            results = self.search(query)
            if not results:
                return f"I couldn't find anything about '{query}' in our history."
            snippets = [f"[{r['role']}] {r['content'][:80]}" for r in results]
            return "Here's what I found:\n" + "\n".join(snippets)

        return None
