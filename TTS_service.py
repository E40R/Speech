import sounddevice as sd
import numpy as np
import httpx
import io
import wave
from dotenv import load_dotenv
import os

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Audio config
SAMPLE_RATE = 24000        # Deepgram Aura outputs 24kHz
CHANNELS = 1
CHUNK_BYTES = 4096         # how many bytes to read per chunk during playback

# Voice config — options: aura-asteria-en, aura-luna-en, aura-stella-en, aura-athena-en
VOICE = "aura-asteria-en"

DEEPGRAM_TTS_URL = f"https://api.deepgram.com/v1/speak?model={VOICE}&encoding=linear16&sample_rate={SAMPLE_RATE}"


# ── Internal ──────────────────────────────────────────────────────────────────

def _bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    """Convert raw PCM bytes from Deepgram into a numpy float32 array for sounddevice."""
    audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
    return audio_array.astype(np.float32) / 32767.0  # normalize to -1.0 to 1.0


def _stream_audio(text: str) -> bytes:
    """
    Send text to Deepgram Aura, get back full audio bytes.
    Uses httpx streaming so chunks arrive as they're generated.
    """
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    audio_data = b""

    with httpx.Client() as client:
        with client.stream("POST", DEEPGRAM_TTS_URL, headers=headers, json=payload, timeout=30) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=CHUNK_BYTES):
                if chunk:
                    audio_data += chunk

    return audio_data


# ── Public interface ──────────────────────────────────────────────────────────

def speak(text: str) -> None:
    """
    Convert text to speech and play it through speakers.
    Blocks until audio finishes playing.

    Usage in model_test.py:
        from tts_service import speak
        speak(response)
    """
    print("[TTS] Generating audio...")
    raw_bytes = _stream_audio(text)

    if not raw_bytes:
        print("[TTS] No audio received.")
        return

    print("[TTS] Playing...")
    audio_array = _bytes_to_numpy(raw_bytes)

    sd.play(audio_array, samplerate=SAMPLE_RATE)
    sd.wait()  # blocks until playback finishes
    print("[TTS] Done.")