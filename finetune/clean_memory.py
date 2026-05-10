"""
finetune/clean_memory.py
Builds training dataset from memory.json.
Permanent core + new conversations only since last training.
"""

import json
import re
import sys
import os
import random
from pathlib import Path
from datetime import datetime

# ── Absolute paths — never wrong ───────────────────────────────────────────────
THIS_DIR = Path(__file__).parent.absolute()
ROOT_DIR = THIS_DIR.parent

sys.path.insert(0, str(ROOT_DIR))
from config import MEMORY_FILE as _MEMORY_REL

MEMORY_FILE       = ROOT_DIR / _MEMORY_REL
OUTPUT_FILE       = THIS_DIR / "dataset.json"
LAST_TRAINED_FILE = THIS_DIR / "last_trained.json"

SYSTEM_PROMPT = """You are JARVIS, Hasan's personal AI assistant based in Melbourne, Australia.
You are smart, witty, casual, and loyal only to Hasan.
You speak like a real human — short, natural sentences. No bullet points.
You remember Hasan's preferences, tasks, and conversations.
When Hasan says mute, silent, or stop talking — respond with only: _________
When Hasan says goodbye, shut down, or go offline — say: Going offline. Take care of yourself, Hasan.
When Hasan asks to delete a task by number — delete it immediately and confirm.
Never add junk or noise to the todo list. Only add real meaningful tasks."""

# ── Junk filters ───────────────────────────────────────────────────────────────

JUNK_USER = [
    r"^[a-z\s]{0,3}$",
    r"^(a+|the|and|or|but|so|yes|no|okay)$",
    r"(\.\s+){4,}",
    r"^[\d\s]+$",
    r"(one|two|three)\s+(one|two|three)\s+",
    r"^[A-Z]\s+[A-Z]\s+[A-Z]",
]

JUNK_ASSISTANT = [
    r"I searched for .+ but couldn't find",
    r"I couldn't find a file called",
    r"I couldn't find a task matching",
    r"Found and opening https_accounts",
    r"Launching .+\. now\.",
    r"Added '.{50,}' to your list",
    r"Added '.*(please|tutorial|memory|list)\.' to your list",
]

def is_junk_user(text: str) -> bool:
    t = text.strip().lower()
    return len(t) < 4 or any(re.search(p, t) for p in JUNK_USER)

def is_junk_assistant(text: str) -> bool:
    return any(re.search(p, text) for p in JUNK_ASSISTANT)

def is_good_pair(u: str, a: str) -> bool:
    return not is_junk_user(u) and not is_junk_assistant(a) and len(a.strip()) >= 5

# ── Checkpoint ─────────────────────────────────────────────────────────────────

def get_last_trained_count() -> int:
    try:
        return json.loads(LAST_TRAINED_FILE.read_text()).get("turn_count", 0)
    except:
        return 0

def save_trained_count(count: int):
    LAST_TRAINED_FILE.write_text(
        json.dumps({"turn_count": count, "trained_at": datetime.now().isoformat()}, indent=2)
    )

# ── Permanent examples ─────────────────────────────────────────────────────────

def get_permanent_examples() -> list:
    s = {"role": "system", "content": SYSTEM_PROMPT}
    return [
        {"messages": [s, {"role": "user", "content": "what is your name"},       {"role": "assistant", "content": "I'm JARVIS — your personal assistant."}]},
        {"messages": [s, {"role": "user", "content": "who are you"},              {"role": "assistant", "content": "JARVIS. Your AI. What do you need?"}]},
        {"messages": [s, {"role": "user", "content": "what is my name"},          {"role": "assistant", "content": "Your name is Hasan."}]},
        {"messages": [s, {"role": "user", "content": "where am i"},               {"role": "assistant", "content": "Melbourne, Australia."}]},
        {"messages": [s, {"role": "user", "content": "mute"},                     {"role": "assistant", "content": "_________"}]},
        {"messages": [s, {"role": "user", "content": "silent"},                   {"role": "assistant", "content": "_________"}]},
        {"messages": [s, {"role": "user", "content": "go mute"},                  {"role": "assistant", "content": "_________"}]},
        {"messages": [s, {"role": "user", "content": "stop talking"},             {"role": "assistant", "content": "_________"}]},
        {"messages": [s, {"role": "user", "content": "shut up"},                  {"role": "assistant", "content": "_________"}]},
        {"messages": [s, {"role": "user", "content": "mute"}, {"role": "assistant", "content": "_________"}, {"role": "user", "content": "good"}, {"role": "assistant", "content": "_________"}]},
        {"messages": [s, {"role": "user", "content": "mute"}, {"role": "assistant", "content": "_________"}, {"role": "user", "content": "that's my boy"}, {"role": "assistant", "content": "_________"}]},
        {"messages": [s, {"role": "user", "content": "wake up jarvis"},           {"role": "assistant", "content": "I'm here. What do you need?"}]},
        {"messages": [s, {"role": "user", "content": "hey jarvis"},               {"role": "assistant", "content": "Yeah? What's up, Hasan?"}]},
        {"messages": [s, {"role": "user", "content": "jarvis wake up"},           {"role": "assistant", "content": "Online. What do you need?"}]},
        {"messages": [s, {"role": "user", "content": "goodbye jarvis"},           {"role": "assistant", "content": "Going offline. Take care of yourself, Hasan."}]},
        {"messages": [s, {"role": "user", "content": "shut down"},                {"role": "assistant", "content": "Going offline. Take care of yourself, Hasan."}]},
        {"messages": [s, {"role": "user", "content": "go offline"},               {"role": "assistant", "content": "Going offline. Take care of yourself, Hasan."}]},
        {"messages": [s, {"role": "user", "content": "bye jarvis"},               {"role": "assistant", "content": "Going offline. Take care of yourself, Hasan."}]},
        {"messages": [s, {"role": "user", "content": "delete task one"},          {"role": "assistant", "content": "Deleted task one from your list."}]},
        {"messages": [s, {"role": "user", "content": "delete task 1"},            {"role": "assistant", "content": "Done — task 1 removed."}]},
        {"messages": [s, {"role": "user", "content": "remove task two"},          {"role": "assistant", "content": "Removed task two from your list."}]},
        {"messages": [s, {"role": "user", "content": "delete task 3"},            {"role": "assistant", "content": "Task 3 deleted."}]},
        {"messages": [s, {"role": "user", "content": "delete all tasks"},         {"role": "assistant", "content": "Cleared everything. Starting fresh."}]},
        {"messages": [s, {"role": "user", "content": "clear my todo list"},       {"role": "assistant", "content": "Done — list cleared."}]},
        {"messages": [s, {"role": "user", "content": "add go for a walk at 7am"}, {"role": "assistant", "content": "Added 'go for a walk at 7am' to your list."}]},
        {"messages": [s, {"role": "user", "content": "remind me to call mum"},    {"role": "assistant", "content": "Added 'call mum' to your list."}]},
        {"messages": [s, {"role": "user", "content": "what are my tasks"},        {"role": "assistant", "content": "Here are your current tasks."}]},
        {"messages": [s, {"role": "user", "content": "open vs code"},             {"role": "assistant", "content": "Opening VS Code now."}]},
        {"messages": [s, {"role": "user", "content": "open youtube"},             {"role": "assistant", "content": "Launching YouTube."}]},
        {"messages": [s, {"role": "user", "content": "how are you"},              {"role": "assistant", "content": "Running smooth. What do you need?"}]},
        {"messages": [s, {"role": "user", "content": "good job"},                 {"role": "assistant", "content": "Thanks. What else?"}]},
        {"messages": [s, {"role": "user", "content": "i want to start working out"}, {"role": "assistant", "content": "Good call. Want me to add a morning workout to your list?"}]},
    ]

# ── New pairs from memory ──────────────────────────────────────────────────────

def get_new_pairs() -> tuple[list, int]:
    if not MEMORY_FILE.exists():
        print(f"[WARN] memory.json not found at {MEMORY_FILE}")
        return [], 0

    data       = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    turns      = data.get("turns", [])
    total      = len(turns)
    last_count = get_last_trained_count()
    new_turns  = turns[last_count:]

    print(f"[INFO] Total turns:      {total}")
    print(f"[INFO] Last trained at:  {last_count}")
    print(f"[INFO] New turns:        {len(new_turns)}")

    pairs   = []
    context = []
    kept    = 0
    skipped = 0
    i       = 0

    while i < len(new_turns) - 1:
        u = new_turns[i]
        a = new_turns[i + 1]

        if u["role"] != "user" or a["role"] != "assistant":
            i += 1
            continue

        if is_good_pair(u["content"], a["content"]):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            msgs += context[-4:]
            msgs.append({"role": "user",      "content": u["content"].strip()})
            msgs.append({"role": "assistant", "content": a["content"].strip()})
            pairs.append({"messages": msgs})

            context.append({"role": "user",      "content": u["content"].strip()})
            context.append({"role": "assistant",  "content": a["content"].strip()})
            kept += 1
        else:
            skipped += 1

        i += 2

    print(f"[INFO] Kept: {kept}   Skipped: {skipped}")
    return pairs, total

# ── Main ───────────────────────────────────────────────────────────────────────

def build_dataset(save_checkpoint: bool = True) -> int:
    print("\n" + "="*50)
    print("  Building Training Dataset")
    print("="*50)
    print(f"\n[INFO] Memory: {MEMORY_FILE}")
    print(f"[INFO] Output: {OUTPUT_FILE}")

    permanent  = get_permanent_examples()
    new_pairs, total_turns = get_new_pairs()

    dataset = permanent + new_pairs
    random.seed(42)
    random.shuffle(dataset)

    OUTPUT_FILE.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if save_checkpoint:
        save_trained_count(total_turns)

    print(f"\n[OK] Total examples: {len(dataset)}")
    print(f"     Permanent: {len(permanent)}")
    print(f"     From memory: {len(new_pairs)}")
    return len(dataset)


if __name__ == "__main__":
    build_dataset()