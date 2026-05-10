"""
voice_auth.py
Verifies speaker identity using resemblyzer.
"""

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

encoder = VoiceEncoder()

# Load your voice sample once on import
_sample_path = Path("your_voice_sample.wav")
if not _sample_path.exists():
    raise FileNotFoundError(
        "your_voice_sample.wav not found in project root.\n"
        "Copy it from V2 or record a new one."
    )

your_voice     = preprocess_wav(_sample_path)
your_embedding = encoder.embed_utterance(your_voice)


def is_my_voice(audio_input, threshold: float = 0.60) -> bool:
    """
    Returns True if audio_input matches Hasan's voice.
    audio_input can be a file path (str) or numpy array.
    """
    try:
        processed  = preprocess_wav(audio_input)
        embedding  = encoder.embed_utterance(processed)
        similarity = np.dot(your_embedding, embedding)
        print(f"Voice similarity: {similarity:.2f}")
        return float(similarity) >= threshold
    except Exception as e:
        print(f"[voice_auth] Error: {e}")
        return False