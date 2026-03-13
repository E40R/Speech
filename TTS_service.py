import sounddevice as sd
import numpy as np
import httpx
import threading
from dotenv import load_dotenv
import os

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Audio config
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_BYTES = 4096

# Voice config
VOICE = "aura-asteria-en"

DEEPGRAM_TTS_URL = (
    f"https://api.deepgram.com/v1/speak"
    f"?model={VOICE}&encoding=linear16&sample_rate={SAMPLE_RATE}"
)

# ── Barge-in stop flag ────────────────────────────────────────────────────────
# This is a simple threading.Event:
#   stop_event.set()   → signals speak() to stop
#   stop_event.clear() → resets before each speak() call
#   stop_event.is_set() → speak() checks this between chunks

stop_event = threading.Event()


# ── Internal ──────────────────────────────────────────────────────────────────

def _bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
    return audio_array.astype(np.float32) / 32768.0 


def _fetch_audio(text: str) -> bytes:
    """Fetch full audio bytes from Deepgram Aura."""
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    audio_data = b""
    with httpx.Client() as client:
        with client.stream(
            "POST", DEEPGRAM_TTS_URL,
            headers=headers,
            json={"text": text},
            timeout=30
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=CHUNK_BYTES):
                if chunk:
                    audio_data += chunk
    return audio_data


# ── Public interface ──────────────────────────────────────────────────────────

"""
    Convert text to speech and play chunk by chunk.
    Checks stop_event between every chunk — if set, stops immediately.

    Returns:
        True  → played fully
        False → interrupted by barge-in

    Usage in main.py:
        from TTS_service import speak, stop_event
        stop_event.clear()
        completed = speak(response)
"""
def speak(text: str) -> bool:
    print("[TTS] Generating audio...")
    raw_bytes = _fetch_audio(text)

    if not raw_bytes:
        print("[TTS] No audio received.")
        return True

    audio_array = _bytes_to_numpy(raw_bytes)
    print("[TTS] Playing...")
    stop_event.clear()

    # play audio — non blocking
    sd.play(audio_array, samplerate=SAMPLE_RATE)
    stop_event.clear()
    # block manually but check stop_event every 50ms
    while sd.get_stream().active:
        if stop_event.is_set():
            sd.stop()
            print("[TTS] Interrupted.")
            return False
        import time
        time.sleep(0.05)  # check every 50ms

    print("[TTS] Done.")
    return True
