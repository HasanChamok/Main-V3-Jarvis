"""
finetune/auto_train.py
Background retraining watcher for JARVIS V3.
"""
from config import MEMORY_FILE as _MEMORY_REL
import json
import os
import sys
import time
import subprocess
import schedule
from pathlib import Path
from datetime import datetime

THIS_DIR = Path(__file__).parent.absolute()
ROOT_DIR = THIS_DIR.parent

sys.path.insert(0, str(ROOT_DIR))


MEMORY_FILE       = ROOT_DIR / _MEMORY_REL
LAST_TRAINED_FILE = THIS_DIR / "last_trained.json"
LOG_FILE          = THIS_DIR / "auto_train.log"
MODELFILE         = ROOT_DIR / "Modelfile"
CLEAN_SCRIPT      = THIS_DIR / "clean_memory.py"
TRAIN_SCRIPT      = THIS_DIR / "train.py"

MIN_NEW_TURNS  = 20
CHECK_INTERVAL = 30
PYTHON         = sys.executable


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_current_turns() -> int:
    try:
        return len(json.loads(MEMORY_FILE.read_text(encoding="utf-8")).get("turns", []))
    except:
        return 0

def get_last_trained() -> int:
    try:
        return json.loads(LAST_TRAINED_FILE.read_text()).get("turn_count", 0)
    except:
        return 0

def new_turns() -> int:
    return get_current_turns() - get_last_trained()


def run_step(cmd: list, name: str) -> bool:
    log(f"Starting: {name}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
    if r.returncode != 0:
        log(f"FAILED: {name}")
        log(f"Error: {r.stderr[-500:]}")
        return False
    log(f"Done: {name}")
    return True


def run_pipeline():
    n = new_turns()
    log("=" * 38)
    log(f"TRAINING — {n} new turns")
    log("=" * 38)

    t = time.time()

    if not run_step([PYTHON, str(CLEAN_SCRIPT)], "Building dataset"):
        return
    if not run_step([PYTHON, str(TRAIN_SCRIPT)], "Training model"):
        return
    if not run_step(["ollama", "create", "jarvis", "-f", str(MODELFILE)], "Loading into Ollama"):
        return

    log(f"COMPLETE in {round((time.time()-t)/60, 1)} mins — restart main.py to use new model")

    try:
        import winsound
        winsound.Beep(1000, 500)
    except:
        pass


def check():
    n = new_turns()
    if n == 0:
        return
    if n < MIN_NEW_TURNS:
        log(f"Waiting: {n}/{MIN_NEW_TURNS} new turns")
        return
    log(f"Threshold reached: {n} turns")
    run_pipeline()


def midnight():
    n = new_turns()
    if n == 0:
        log("Midnight: nothing new")
        return
    log(f"Midnight: {n} new turns")
    run_pipeline()


if __name__ == "__main__":
    log("AUTO-TRAIN V3 started")
    log(f"Watching: {MEMORY_FILE}")
    log(f"Python:   {PYTHON}")

    n = new_turns()
    if n >= MIN_NEW_TURNS:
        log(f"Startup: {n} turns ready — training now")
        run_pipeline()
    else:
        log(f"Startup: {n} turns — waiting for {MIN_NEW_TURNS}")

    schedule.every().day.at("00:00").do(midnight)
    schedule.every(CHECK_INTERVAL).minutes.do(check)

    log("Scheduler running. Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(60)