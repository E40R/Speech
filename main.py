import threading
from STT_service import listen, is_speech, get_vad_model, CHUNK_SIZE, SAMPLE_RATE
from TTS_service import speak, stop_event
from LLM_brain import triagellm, generate_tts_response, update_history, log_to_json, log_conversation
import sounddevice as sd


# ── Barge-in thread ───────────────────────────────────────────────────────────

def _barge_in_watcher(cancel_event: threading.Event):
    """
    Runs in background during TTS playback.
    Sets stop_event if user speech detected → interrupts TTS.
    """
    import time
    time.sleep(0.5)  # wait 500ms before watching — lets listen() audio tail fade out
    
    get_vad_model()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
    ) as stream:
        while not cancel_event.is_set():
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()
            if is_speech(chunk):
                print("[BARGE-IN] User interrupted!")
                stop_event.set()
                break


# ── Main conversation loop ────────────────────────────────────────────────────

def run():
    print("Veterinary Assistant ready. Speak to begin.\n")

    while True:
        # 1. Listen
        user_input = listen()
        if not user_input:
            continue

        print(f"\nYou: {user_input}")

        # 2. Triage LLM
        triage_result = triagellm(user_input)

        # 3. Log
        log_to_json(user_input, triage_result)

        # 4. Generate TTS script
        tts_text = triage_result.recommendations
        print(f"\nAssistant: {tts_text}\n")

        # 5. Start barge-in watcher
        cancel_barge_in = threading.Event()
        barge_in_thread = threading.Thread(
            target=_barge_in_watcher,
            args=(cancel_barge_in,),
            daemon=True
        )
        barge_in_thread.start()

        # 6. Speak
        completed = speak(tts_text)

        # 7. Stop barge-in watcher
        cancel_barge_in.set()
        barge_in_thread.join(timeout=1)

        # 8. Update history
        update_history(user_input, tts_text, triage_result)
        log_conversation(user_input, tts_text)   #added this to have conversation histroy (this updates same file for each session)
        
        # 9. If interrupted, loop back immediately
        if not completed:
            print("[Main] Barge-in detected, listening for new input...")
            continue

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped.")
