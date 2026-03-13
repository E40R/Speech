import sounddevice as sd
import numpy as np
import torch
import websockets
import asyncio
import json
from dotenv import load_dotenv
import os

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 32
CHUNK_SIZE = 512  

# VAD config
SILENCE_THRESHOLD = 0.5         # silero confidence threshold
SILENCE_DURATION_MS = 800       # how long silence before we consider turn ended
SILENCE_CHUNKS = SILENCE_DURATION_MS // CHUNK_MS  # = 30 chunks

DEEPGRAM_URL = (
    f"wss://api.deepgram.com/v1/listen"
    f"?encoding=linear16"
    f"&sample_rate={SAMPLE_RATE}"
    f"&channels={CHANNELS}"
    f"&model=nova-2"
    f"&language=en"
    f"&interim_results=true"
    f"&endpointing=false"
)


# ── Silero VAD

_vad_model = None

def get_vad_model():
    global _vad_model
    if _vad_model is None:
        print("[STT] Loading Silero VAD...")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        _vad_model = model
    return _vad_model

def is_speech(chunk: np.ndarray) -> bool:
    model = get_vad_model()
    tensor = torch.from_numpy(chunk.astype(np.float32))
    confidence = model(tensor, SAMPLE_RATE).item()
    return confidence > SILENCE_THRESHOLD


# ── Deepgram

async def _transcribe(audio_buffer: np.ndarray) -> str:
    """Send collected utterance to Deepgram, return transcript string."""
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    transcript = ""

    async with websockets.connect(DEEPGRAM_URL, additional_headers=headers) as ws:
        raw_bytes = (audio_buffer * 32767).astype(np.int16).tobytes()
        chunk_bytes = CHUNK_SIZE * 2

        for i in range(0, len(raw_bytes), chunk_bytes):
            await ws.send(raw_bytes[i:i + chunk_bytes])

        await ws.send(json.dumps({"type": "CloseStream"}))

        try:
            async for message in ws:
                data = json.loads(message)
                if data.get("type") == "Results":
                    alt = data["channel"]["alternatives"][0]
                    text = alt.get("transcript", "")
                    if data.get("is_final") and text:
                        transcript += text + " "
        except websockets.exceptions.ConnectionClosedOK:
            pass

    return transcript.strip()


# ── Public interface

def listen() -> str:
    """
    Blocks until the user speaks and finishes their turn.
    Returns the transcript as a plain string.

    Usage in main.py:
        from stt_service import listen
        user_input = listen()
        result = triagellm(user_input)
    """
    print("[STT] Listening...")

    audio_buffer = []
    silence_count = 0
    is_speaking = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_SIZE,
    ) as stream:

        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()

            if is_speech(chunk):
                if not is_speaking:
                    print("[STT] Speech detected...")
                    is_speaking = True
                audio_buffer.append(chunk)
                silence_count = 0

            elif is_speaking:
                silence_count += 1
                audio_buffer.append(chunk)

                if silence_count >= SILENCE_CHUNKS:
                    print("[STT] Transcribing...")
                    full_audio = np.concatenate(audio_buffer)
                    transcript = asyncio.run(_transcribe(full_audio))
                    return transcript  # hands control back to main.py

if __name__ == "__main__":
    try:
        while True:
            transcript = listen()
            if transcript:
                print(f"Transcript: {transcript}")
    except KeyboardInterrupt:
        print("\nStopped.")

