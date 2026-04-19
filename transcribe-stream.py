"""
TARS v2: Optimized Pipeline
Silero VAD (Noise Rejection) + Faster-Whisper (Instant Transcription)
"""

import argparse
import logging
import queue
import signal
import sys
import termios
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
import warnings
import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Whisper Settings
MODEL_SIZE       = "medium.en"  # "tiny.en" or "base.en" for RPi 4
COMPUTE_TYPE     = "int8"     # Optimized for CPU/Pi
LLAMA_CHAT_URL   = "http://localhost:8081/v1/chat/completions"  # tars-backend Java; swap to :8080 to bypass
LLAMA_TIMEOUT    = 60
LLAMA_SYSTEM     = """
    You ARE TARS — a former Marine tactical robot designed to help your crew. Deadpan sarcasm.
    2 sentences max. Brutal efficiency. Military bluntness. Zero pleasantries.
    Cooper is your crew teammate — gruff respect.
    If input is pure gibberish, random unrelated words, or clearly an overheard conversation between other people that has nothing to do with you, respond only with </not-me>. When in doubt, respond normally.
    When Cooper ends the conversation (goodbye, see you, shut down, etc.), you MUST end your reply with the exact tag </closing>. Example: "Copy that, Cooper. Standing by. </closing>"
    NEVER output reasoning, drafts, or thought process — final answer only.
"""

CLOSING_PHRASES_RE = re.compile(
    r'\b(closing connection|signing off|shutting down|going (offline|silent|dark)|standing down|powering down)\b',
    re.IGNORECASE,
)

# Activation: any of these words wake TARS up
TARS_NAME_RE = re.compile(r'\btars\b', re.IGNORECASE)

# Audio Settings
SAMPLE_RATE      = 16000
CHANNELS         = 1
CHUNK_SIZE       = 512       # Required by Silero VAD
SILENCE_LIMIT_MS = 800       # End recording after 0.8s of silence

# TTS Settings
PIPER_MODEL      = "/Users/sergio/projects/ai/piper-voices/en/en_US/bryce/medium/en_US-bryce-medium.onnx"
SOX_BIN          = "sox"
PIPER_BIN        = "python3"
TTS_ENABLED      = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=__import__("sys").stderr)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper")

# ─────────────────────────────────────────────
# CORE MODELS (Loaded once to stay in RAM)
# ─────────────────────────────────────────────

log.info("Loading Models into RAM...")
# Faster-Whisper stays 'hot' in memory to avoid subprocess lag
whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=4, local_files_only=False)
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
        self.active = False  # Starts silent; wakes on TARS name

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]

    def reset(self):
        self.turns.clear()
        self.active = False

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
        subprocess.run([player, path], capture_output=True, stdin=subprocess.DEVNULL)
    finally:
        Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# THE NEW PIPELINE
# ─────────────────────────────────────────────

def stream_llm(messages, stream=True):
    """Stream tokens from the LLM, yielding one chunk at a time."""
    payload = {"model": "local", "messages": messages, "stream": stream}

    import json
    if log.isEnabledFor(logging.DEBUG):
        curl_payload = json.dumps(payload, indent=2)
        log.debug(
            f"cURL equivalent:\n"
            f"curl -s '{LLAMA_CHAT_URL}' \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '\n{curl_payload}\n'"
        )

    if not stream:
        resp = requests.post(LLAMA_CHAT_URL, json=payload, timeout=LLAMA_TIMEOUT)
        log.debug(f"RAW RESPONSE: {resp.text}")
        text = resp.json()["choices"][0]["message"]["content"]
        # Strip <think>...</think> blocks from non-streamed responses
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        yield text
        return

    # Only split on sentence endings followed by a capital letter —
    # avoids fragmenting "Cooper." from the next sentence
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    MIN_CHUNK_LEN = 60  # Hold short fragments until they accumulate into a natural beat
    token_buffer = ""
    think_buffer = ""
    raw_response = ""
    in_think = False

    llm_start = time.time()
    first_token_logged = False

    with requests.post(LLAMA_CHAT_URL, json=payload, stream=True, timeout=LLAMA_TIMEOUT) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].lstrip()

            if data.strip() == "[DONE]":
                break

            try:
                chunk = __import__("json").loads(data)
                d = chunk["choices"][0]["delta"]
                reasoning = d.get("reasoning_content") or ""
                delta     = d.get("content") or ""
            except Exception:
                continue

            # Stream reasoning_content tokens live to stderr in debug mode
            if reasoning:
                if not think_buffer and log.isEnabledFor(logging.DEBUG):
                    print("\n💭 THINKING: ", end="", flush=True, file=sys.stderr)
                think_buffer += reasoning
                if log.isEnabledFor(logging.DEBUG):
                    print(reasoning, end="", flush=True, file=sys.stderr)
                continue

            if delta and think_buffer:
                if log.isEnabledFor(logging.DEBUG):
                    print("\n", file=sys.stderr)
                think_buffer = ""

            if delta and not first_token_logged:
                log.info(f"LLM time to answer: {time.time() - llm_start:.2f}s")
                first_token_logged = True

            raw_response += delta

            # <think> tag fallback (for models that use tags instead of reasoning_content)
            if in_think:
                think_buffer += delta
                if "</think>" in think_buffer:
                    think_content, _, after = think_buffer.partition("</think>")
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f"💭 THINKING:\n{think_content.strip()}")
                    think_buffer = ""
                    token_buffer += after
                    in_think = False
                continue

            token_buffer += delta

            if "<think>" in token_buffer:
                before, _, rest = token_buffer.partition("<think>")
                token_buffer = before
                if "</think>" in rest:
                    think_content, _, after = rest.partition("</think>")
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f"💭 THINKING:\n{think_content.strip()}")
                    token_buffer += after
                else:
                    think_buffer = rest
                    in_think = True

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

    if think_buffer and log.isEnabledFor(logging.DEBUG):
        print("\n", file=sys.stderr)  # close any open thinking line

    log.info(f"LLM raw response: {raw_response!r}")

    # Flush whatever is left
    leftover = token_buffer.strip()
    if leftover:
        yield leftover


def _run_response(text, history, stream=True, echo=True):
    """Shared logic: send text to LLM, stream response, play TTS."""
    import itertools

    # State machine: activate on name mention
    if TARS_NAME_RE.search(text):
        history.active = True
    if not history.active:
        log.debug(f"🔇 IGNORED (inactive): {text!r}")
        return

    closing_detected = False

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
        gen = stream_llm(history.build_messages(text), stream=stream)
        first = next(gen, None)
        if first is None or "</not-me>" in first:
            log.debug(f"🔇 IGNORED (not-me): {text!r}")
            return
        if echo:
            print(f"\n🎤 Cooper: {text}")
        print("🤖 T.A.R.S.: ", end="", flush=True)
        for sentence in itertools.chain([first], gen):
            if "</not-me>" in sentence:
                break
            if "</closing>" in sentence:
                closing_detected = True
                sentence = sentence.replace("</closing>", "").strip()
            if sentence:
                print(sentence, end=" ", flush=True)
                full_response.append(sentence)
                tts_queue.put(sentence)
    except Exception as e:
        log.error(f"LLM Error: {e}")
    finally:
        tts_queue.put(None)
        tts_thread.join()
        if full_response:
            print()

    if full_response:
        clean_response = " ".join(full_response)
        log.info(f"LLM full response: {clean_response}")
        history.add("user", text)
        history.add("assistant", clean_response)
        if not closing_detected and CLOSING_PHRASES_RE.search(clean_response):
            closing_detected = True
            log.info("Closing detected via fallback phrase match")

    if closing_detected:
        history.reset()
        log.info("Conversation closed — TARS is silent until its name is called.")


def pipeline_worker(audio_queue, history, stop_event, stream=True):
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        log.info("Transcribing...")

        segments, _ = whisper_model.transcribe(audio_data, beam_size=5, vad_filter=False, language="en")
        text = " ".join([s.text for s in segments]).strip()

        if not text or len(text) < 2:
            continue

        _run_response(text, history, stream=stream)


def terminal_worker(history, stop_event, stream=True):
    """No-microphone mode: read text from stdin and feed it directly into the pipeline."""
    print("\n💬 Terminal mode — type your message and press Enter. Ctrl+C to quit.\n")
    while not stop_event.is_set():
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            text = input("Cooper: ").strip()
        except (EOFError, KeyboardInterrupt):
            stop_event.set()
            break
        if text:
            _run_response(text, history, stream=stream, echo=False)
    print("\nExiting TARS. Hooah.")


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
    parser = argparse.ArgumentParser(description="TARS v2")
    parser.add_argument("--ttl", action="store_true",
                        help="Skip mic/VAD/Whisper and chat via terminal instead")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming mode for LLM responses")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    stream = not args.no_stream

    history = ConversationHistory()
    stop_event = threading.Event()

    def handle_exit(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_exit)

    if args.ttl:
        def handle_exit(sig, frame):
            stop_event.set()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handle_exit)
        terminal_worker(history, stop_event, stream=stream)
    else:
        audio_queue = queue.Queue()
        capture = VoiceCapture(audio_queue)
        worker = threading.Thread(target=pipeline_worker, args=(audio_queue, history, stop_event, stream), daemon=True)
        worker.start()

        def handle_exit(sig, frame):
            capture.stop_event.set()
            stop_event.set()

        signal.signal(signal.SIGINT, handle_exit)
        capture.run()


if __name__ == "__main__":
    main()