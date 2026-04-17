"""
TARS v2: Optimized Pipeline
Silero VAD (Noise Rejection) + Faster-Whisper (Instant Transcription)
"""

import logging
import queue
import signal
import threading
import time
import re
import tempfile
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import pyaudio
import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Whisper Settings
MODEL_SIZE       = "tiny.en"  # "tiny.en" or "base.en" for RPi 4
COMPUTE_TYPE     = "int8"     # Optimized for CPU/Pi
LLAMA_CHAT_URL   = "http://localhost:8080/v1/chat/completions"
LLAMA_SYSTEM     = """
    Act as TARS, the former Marine tactical robot very well known because your deadpan sarcasm. 
    Respond with extreme conciseness, 2 sentences max, brutal efficiency, military bluntness. 
    Output zero AI pleasantries or fluff. 
    The user, Cooper, is your trip teammate so treat well.
    Ruthlessly call out human error.
"""

# Audio Settings
SAMPLE_RATE      = 16000
CHANNELS         = 1
CHUNK_SIZE       = 512       # Required by Silero VAD
SILENCE_LIMIT_MS = 800       # End recording after 0.8s of silence
MIN_AUDIO_DURATION = 0.5     # seconds — ignore clips shorter than this

# TTS Settings
PIPER_MODEL      = "/Users/sergio/projects/ai/piper-voices/en/en_US/bryce/medium/en_US-bryce-medium.onnx"
SOX_BIN          = "sox"
PIPER_BIN        = "python3"
TTS_ENABLED      = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CORE MODELS (Loaded once to stay in RAM)
# ─────────────────────────────────────────────

log.info("Loading Models into RAM...")
# Faster-Whisper stays 'hot' in memory to avoid subprocess lag
whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=4)
# Silero VAD is neural-net based: ignores fans, keyboards, and hums
vad_model = load_silero_vad()

# ─────────────────────────────────────────────
# LOGIC COMPONENTS (History, TTS, etc.)
# ─────────────────────────────────────────────

# Global flag: mic capture pauses processing while TARS is speaking
is_speaking = threading.Event()

class ConversationHistory:
    def __init__(self, max_turns=6):
        self.turns = []
        self.max_turns = max_turns

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]

    def build_messages(self, text):
        messages = [{"role": "system", "content": LLAMA_SYSTEM}]
        messages.extend(self.turns)
        messages.append({"role": "user", "content": text})
        return messages


def render_wav(text):
    """Render text through Piper + SoX into a temp wav. Returns path or None."""
    clean = re.sub(r"\[.*?\]", "", text).strip()
    if not clean:
        return None
    raw_wav_path = None
    robot_wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            raw_wav_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            robot_wav_path = f.name

        subprocess.run(
            [PIPER_BIN, "-m", "piper", "--model", PIPER_MODEL,
             "--length-scale", "0.60", "--output_file", raw_wav_path],
            input=clean, text=True, capture_output=True
        )
        subprocess.run(
            [SOX_BIN, raw_wav_path, robot_wav_path,
             "norm", "-3", 
             "bass", "-5", 
             "treble", "+3",   # the higher value the more metallic it sounds
             "overdrive", "3", # the higher value the more metallic it sounds    
             "pitch", "-125",
             "echo", "0.6", "0.5", "15", "0.2",
             "flanger", "0.5", "0.8", "0", "0.7", "0.5",
             "rate", "22050"],
            capture_output=True
        )
        return robot_wav_path
    except Exception as e:
        log.error(f"Render Error: {e}")
        if robot_wav_path:
            Path(robot_wav_path).unlink(missing_ok=True)
        return None
    finally:
        if raw_wav_path:
            Path(raw_wav_path).unlink(missing_ok=True)


def play_wav(path):
    """Play a pre-rendered wav and delete it afterwards."""
    if not path:
        return
    try:
        player = "afplay" if platform.system() == "Darwin" else "aplay"
        subprocess.run([player, path], capture_output=True)
    finally:
        Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# THE NEW PIPELINE
# ─────────────────────────────────────────────

def stream_llm(messages):
    """Stream tokens from the LLM, yielding one chunk at a time."""
    payload = {"model": "local", "messages": messages, "stream": True}
    # Only split on sentence endings followed by a capital letter —
    # avoids fragmenting "Cooper." from the next sentence
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    MIN_CHUNK_LEN = 60  # Hold short fragments until they accumulate into a natural beat
    token_buffer = ""
    in_think = False

    llm_start = time.time()
    first_token_logged = False
    log.debug(f"LLM Payload: {payload}")

    with requests.post(LLAMA_CHAT_URL, json=payload, stream=True, timeout=60) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break

            try:
                chunk = __import__("json").loads(data)
                delta = chunk["choices"][0]["delta"].get("content") or ""

                # log the first received chunk
                if delta and not first_token_logged:
                    log.info(f"LLM time to answer: {time.time() - llm_start:.2f}s")
                    first_token_logged = True
            except Exception:
                continue

            token_buffer += delta

            # Strip <think>...</think> spans that arrive across chunks
            while "<think>" in token_buffer and "</think>" in token_buffer:
                token_buffer = re.sub(r"<think>.*?</think>", "", token_buffer, flags=re.DOTALL)
            # Detect an opening <think> with no closing yet — suppress until it closes
            if "<think>" in token_buffer and "</think>" not in token_buffer:
                in_think = True
                continue
            if in_think:
                if "</think>" in token_buffer:
                    token_buffer = re.sub(r".*?</think>", "", token_buffer, flags=re.DOTALL)
                    in_think = False
                else:
                    continue
            # Strip any orphaned </think>
            token_buffer = token_buffer.replace("</think>", "")

            # Accumulate short fragments before yielding to avoid tiny render cycles
            parts = sentence_endings.split(token_buffer)
            pending = ""
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if not sentence:
                    continue
                pending += (" " if pending else "") + sentence
                if len(pending) >= MIN_CHUNK_LEN:
                    yield pending
                    pending = ""
            # Re-attach un-yielded pending back onto the tail
            token_buffer = (pending + " " + parts[-1]).strip() if pending else parts[-1]

    # Flush whatever is left
    leftover = token_buffer.strip()
    if leftover:
        yield leftover


def pipeline_worker(audio_queue, history, stop_event):
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        log.info("Transcribing...")

        # Guard: skip chunks that are too short or silent
        duration = len(audio_data) / SAMPLE_RATE
        if duration < MIN_AUDIO_DURATION:
            log.debug(f"Skipping short audio chunk ({duration:.2f}s)")
            continue

        # Guard: skip chunks that are effectively silent
        if np.max(np.abs(audio_data)) < 0.01:
            log.debug("Skipping silent audio chunk")
            continue

        segments, _ = whisper_model.transcribe(audio_data, beam_size=5, vad_filter=False)
        text = " ".join([s.text for s in segments]).strip()

        if not text or len(text) < 2:
            continue

        print(f"\n🎤 Cooper: {text}")
        print("🤖 T.A.R.S.: ", end="", flush=True)

        tts_queue = queue.Queue()

        def tts_worker():
            """
            Render-ahead pipeline: while chunk N is playing, chunk N+1 is already rendering.
            This hides the Piper+SoX cost completely behind playback time.

            Timeline:
              Before:  [render A]──[play A]──[render B]──[play B]──[render C]──[play C]
              After:   [render A]──[play A]
                                   [render B]──[play B]
                                               [render C]──[play C]
            """
            executor = ThreadPoolExecutor(max_workers=1)
            next_future = None

            while True:
                sentence = tts_queue.get()

                if sentence is None:
                    # Drain the last pre-rendered chunk if any
                    if next_future:
                        wav_path = next_future.result()
                        is_speaking.set()
                        try:
                            play_wav(wav_path)
                        finally:
                            time.sleep(0.3)
                            is_speaking.clear()
                    break

                if next_future is not None:
                    # A render was already kicked off for this sentence — wait for it
                    wav_path = next_future.result()
                else:
                    # First chunk: no choice but to render synchronously
                    wav_path = render_wav(sentence)
                    sentence = None  # Don't pre-render again below

                # Kick off render of the NEXT sentence in the background
                if sentence is not None:
                    next_future = executor.submit(render_wav, sentence)
                else:
                    next_future = None

                is_speaking.set()
                try:
                    play_wav(wav_path)
                finally:
                    time.sleep(0.3)
                    is_speaking.clear()

                tts_queue.task_done()

            executor.shutdown(wait=False)

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        full_response = []
        try:
            for sentence in stream_llm(history.build_messages(text)):
                print(sentence, end=" ", flush=True)
                full_response.append(sentence)
                tts_queue.put(sentence)
        except Exception as e:
            log.error(f"LLM Error: {e}")
        finally:
            tts_queue.put(None)
            tts_thread.join()
            print()

        if full_response:
            clean_response = " ".join(full_response)
            history.add("user", text)
            history.add("assistant", clean_response)


class VoiceCapture:
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.stop_event = threading.Event()
        # Silero Iterator handles the "is speaking" state machine for us
        self.vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE,
                                        min_silence_duration_ms=SILENCE_LIMIT_MS)

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)

        log.info("TARS is listening... (Neural VAD active)")
        audio_buffer = []

        try:
            while not self.stop_event.is_set():
                frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_int16 = np.frombuffer(frame, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0  # Normalize for Silero

                # Skip processing if TARS is currently speaking (prevents echo self-triggering)
                if is_speaking.is_set():
                    continue

                # Check for speech using Silero
                speech_dict = self.vad_iterator(torch.from_numpy(audio_float32), return_seconds=True)

                if speech_dict:
                    if 'start' in speech_dict:
                        log.debug("Speech Started")
                        audio_buffer = []  # Reset buffer for new sentence

                    if 'end' in speech_dict:
                        log.debug("Speech Ended")
                        # Flatten buffer and send to Whisper
                        full_audio = np.concatenate(audio_buffer)
                        self.audio_queue.put(full_audio)
                        audio_buffer = []

                # Always keep appending if we are in the middle of a speech block
                audio_buffer.append(audio_float32)

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    audio_queue = queue.Queue()
    history = ConversationHistory()
    stop_event = threading.Event()

    capture = VoiceCapture(audio_queue)
    worker = threading.Thread(target=pipeline_worker, args=(audio_queue, history, stop_event), daemon=True)

    worker.start()

    def handle_exit(sig, frame):
        capture.stop_event.set()
        stop_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    capture.run()


if __name__ == "__main__":
    main()