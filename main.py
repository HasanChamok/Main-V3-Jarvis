"""
main.py — Entry point for JARVIS V3.

    python main.py        # headless
    python main.py --gui  # with GUI (future)
"""

import sys
import os
import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
from voice_auth import is_my_voice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_voice() -> bool:
    """Record 5 seconds and verify it is Hasan."""
    print("🎤 Say something to verify your voice...")
    sample_rate = 16000
    duration    = 5

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate = sample_rate,
            channels   = 1,
            dtype      = "int16"
        )
        sd.wait()

        temp_path = "temp_verify.wav"
        sf.write(temp_path, audio, sample_rate)
        result = is_my_voice(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except Exception as e:
        print(f"❌ Recording error: {e}")
        return False


def run_headless():
    from jarvis import JARVIS
    j = JARVIS()
    j.start()


def run_gui():
    # Placeholder for future GUI
    print("[INFO] GUI not implemented in V3 yet. Running headless.")
    run_headless()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS V3")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    print("🔐 JARVIS V3 — Voice Authentication")
    print("=" * 38)

    authenticated = False
    for attempt in range(1, 4):
        print(f"Attempt {attempt} of 3...")
        if verify_voice():
            print("✅ Voice verified! Welcome back, Hasan.")
            authenticated = True
            break
        else:
            remaining = 3 - attempt
            if remaining > 0:
                print(f"❌ Not recognised. {remaining} attempt(s) left.")
            else:
                print("🚫 Access denied.")

    if not authenticated:
        print("🔒 JARVIS will not start. Goodbye.")
        sys.exit(1)

    if args.gui:
        run_gui()
    else:
        run_headless()