"""
modules/todo.py
Full voice-controlled todo list.
Add, complete, delete, update, and list tasks by voice.
"""

import json
import re
from pathlib import Path
from datetime import datetime
import pytz

from config import TODO_FILE, TIMEZONE

# ── Word to number conversion ──────────────────────────────────────────────────
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10,
}

def normalise_identifier(text: str) -> str:
    """
    Convert spoken numbers to digits.
    'task one'  → '1'
    'task 1'    → '1'
    'walk'      → 'walk'   (keyword search)
    """
    text = text.strip().lower()

    # Remove filler words
    text = re.sub(r"\b(number|num|task|item|the|my)\b", "", text).strip()

    # Replace word numbers → digits
    for word, num in WORD_TO_NUM.items():
        text = re.sub(rf"\b{word}\b", str(num), text)

    return text.strip()


class TodoManager:

    def __init__(self):
        self.path = Path(TODO_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.todos: list[dict] = []
        self.load()

    # ── Storage ────────────────────────────────────────────────────────────────

    def load(self):
        if self.path.exists():
            try:
                self.todos = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.todos = []

    def save(self):
        self.path.write_text(
            json.dumps(self.todos, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _now(self) -> str:
        tz = pytz.timezone(TIMEZONE)
        return datetime.now(tz).isoformat()

    def _next_id(self) -> int:
        return max((t["id"] for t in self.todos), default=0) + 1

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def add(self, task: str, priority: str = "normal") -> str:
        # Clean up junk that Whisper adds
        task = task.strip().strip(".")
        if len(task) < 3:
            return "That task was too short to save — could you say it again?"

        todo = {
            "id":        self._next_id(),
            "task":      task,
            "done":      False,
            "priority":  priority,
            "created":   self._now(),
            "completed": None,
        }
        self.todos.append(todo)
        self.save()
        count = len([t for t in self.todos if not t["done"]])
        return f"Added '{task}' to your list. You now have {count} pending task{'s' if count != 1 else ''}."

    def complete(self, identifier: str) -> str:
        todo = self._find(identifier)
        if not todo:
            return self._not_found_message(identifier)
        if todo["done"]:
            return f"'{todo['task']}' is already marked as done."
        todo["done"]      = True
        todo["completed"] = self._now()
        self.save()
        remaining = len([t for t in self.todos if not t["done"]])
        return f"Marked '{todo['task']}' as done. {remaining} task{'s' if remaining != 1 else ''} remaining."

    def delete(self, identifier: str) -> str:
        todo = self._find(identifier)
        if not todo:
            return self._not_found_message(identifier)
        task_name = todo["task"]
        self.todos = [t for t in self.todos if t["id"] != todo["id"]]
        self.save()
        return f"Deleted '{task_name}' from your list."

    def delete_all(self) -> str:
        count = len(self.todos)
        self.todos = []
        self.save()
        return f"Deleted all {count} tasks. Your list is now empty."

    def delete_done(self) -> str:
        done = [t for t in self.todos if t["done"]]
        self.todos = [t for t in self.todos if not t["done"]]
        self.save()
        return f"Cleared {len(done)} completed task{'s' if len(done) != 1 else ''} from your list."

    def update(self, identifier: str, new_text: str) -> str:
        todo = self._find(identifier)
        if not todo:
            return self._not_found_message(identifier)
        old = todo["task"]
        todo["task"] = new_text.strip()
        self.save()
        return f"Updated '{old}' to '{new_text}'."

    def list_todos(self, show_done: bool = False) -> str:
        items = self.todos if show_done else [t for t in self.todos if not t["done"]]
        if not items:
            return "Your todo list is empty — you're all caught up!" if not show_done else "No tasks at all."

        lines = []
        for t in items:
            status = "✓" if t["done"] else "○"
            pri    = " [HIGH]" if t["priority"] == "high" else ""
            lines.append(f"  {status} {t['id']}. {t['task']}{pri}")

        header = f"You have {len(items)} task{'s' if len(items) != 1 else ''}:"
        return header + "\n" + "\n".join(lines)

    def morning_summary(self) -> str:
        pending = [t for t in self.todos if not t["done"]]
        high    = [t for t in pending if t.get("priority") == "high"]

        if not pending:
            return "Your todo list is clear — great start to the day!"

        summary = f"You've got {len(pending)} thing{'s' if len(pending) != 1 else ''} on your list"
        if high:
            summary += f", including {len(high)} high-priority item{'s' if len(high) != 1 else ''}"
        summary += ". " + ", ".join(t["task"] for t in pending[:3])
        if len(pending) > 3:
            summary += f" and {len(pending) - 3} more."
        else:
            summary += "."
        return summary

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _find(self, identifier: str) -> dict | None:
        """
        Find task by:
        1. Spoken number → digit  ('one' → 1)
        2. Digit ID               ('1'   → id 1)
        3. Position in list       ('1'   → first item if id 1 missing)
        4. Keyword match          ('walk' → task containing 'walk')
        """
        norm = normalise_identifier(identifier)

        # Try numeric match first
        if norm.isdigit():
            num = int(norm)

            # Match by ID
            for t in self.todos:
                if t["id"] == num:
                    return t

            # Match by position (1-indexed) — handles gaps in IDs
            active = [t for t in self.todos if not t["done"]]
            if 1 <= num <= len(active):
                return active[num - 1]

        # Keyword match
        matches = [t for t in self.todos if norm in t["task"].lower()]
        return matches[-1] if matches else None

    def _not_found_message(self, identifier: str) -> str:
        """Helpful error that lists available tasks."""
        pending = [t for t in self.todos if not t["done"]]
        if not pending:
            return "Your todo list is empty."
        options = ", ".join(f"{t['id']}. {t['task'][:30]}" for t in pending)
        return f"I couldn't find that task. Your current tasks are: {options}"

    # ── Intent parsing ─────────────────────────────────────────────────────────

    def parse_and_handle(self, text: str) -> str | None:
        text_lower = text.lower().strip()

        # ── List todos ─────────────────────────────────────────────────────────
        if re.search(r"\b(list|show|what'?s? on|read|tell me)\b.+\b(todo|to-do|to do|tasks?|list)\b", text_lower):
            show_done = "done" in text_lower or "completed" in text_lower
            return self.list_todos(show_done)

        if re.search(r"\bmy (todo|to-do|to do|tasks?|list)\b", text_lower):
            return self.list_todos()

        # ── Delete ALL tasks ───────────────────────────────────────────────────
        if re.search(r"\b(delete|clear|remove|wipe)\b.+\b(all|every|everything)\b", text_lower):
            return self.delete_all()

        # ── Delete DONE tasks ──────────────────────────────────────────────────
        if re.search(r"\b(delete|clear|remove)\b.+\b(done|completed|finished)\b", text_lower):
            return self.delete_done()

        # ── Delete specific task ───────────────────────────────────────────────
        m = re.search(r"\b(delete|remove|cancel|drop)\b\s+(?:task\s+)?(.+)", text_lower)
        if m:
            return self.delete(m.group(2).strip())

        # ── Add task ───────────────────────────────────────────────────────────
        m = re.search(
            r"\b(add|create|new|put|remind me to|note down)\b\s+(?:a\s+)?(?:task\s+)?(?:to\s+)?(.+?)(?:\s+to\s+(?:my\s+)?(?:list|todos?))?$",
            text_lower
        )
        if m:
            task = m.group(2).strip()
            priority = "normal"
            if re.search(r"\b(urgent|asap|high priority|important)\b", task):
                priority = "high"
                task = re.sub(r"\b(urgent|asap|high priority|important)\b", "", task).strip()
            return self.add(task, priority)

        # ── Complete task ──────────────────────────────────────────────────────
        m = re.search(
            r"\b(done|complete|finish|tick off|mark as done|completed)\b\s+(?:task\s+)?(.+)",
            text_lower
        )
        if m:
            return self.complete(m.group(2).strip())

        # ── Update task ────────────────────────────────────────────────────────
        m = re.search(r"\b(update|change|rename|edit)\b\s+(?:task\s+)?(.+?)\s+to\s+(.+)", text_lower)
        if m:
            return self.update(m.group(2).strip(), m.group(3).strip())

        return None