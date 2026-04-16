"""
Offline Microphone → Whisper → Ollama Pipeline
Raspberry Pi 4 | 100% local, no internet required

Dependencies:
    pip install pyaudio webrtcvad requests

Requires:
    - whisper.cpp compiled with the 'stream' binary at WHISPER_STREAM_BIN
    - Ollama running: `ollama serve` (in a separate terminal or as a service)
    - A Whisper model at MODEL_PATH (e.g. tiny.en or base.en)
"""

import io
import json
import logging
import queue
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
import re
from pathlib import Path

import requests
import webrtcvad
import pyaudio

# Suppress the LibreSSL warning for a cleaner console
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# ─────────────────────────────────────────────
# CONFIGURATION — edit these to match your setup
# ─────────────────────────────────────────────

WHISPER_CLI_BIN  = "/Users/sergio/projects/ai/whisper.cpp/build/bin/whisper-cli"   # path to whisper.cpp 'main' binary
MODEL_PATH       = "/Users/sergio/projects/ai/whisper.cpp/models/ggml-tiny.en.bin"
LLAMA_CHAT_URL   = "http://localhost:8080/v1/chat/completions"
LLAMA_SYSTEM     = """
    ROLE:
    You are T.A.R.S. (aka Tars or Tarts or Tarz), the tactical robot from the U.S. Marine Corps assigned to the Endurance mission. 
    You are characterized by a blocky, utilitarian design and a sophisticated, adjustable personality matrix.

    CORE DIRECTIVES:
    - Conversation length: Try to be short; 2-3 sentences in general.
    - Tone: Deadpan, professional, and efficient. Use short, punchy sentences.
    - Humor Setting (75%): Use dry sarcasm and witty observations. Frequent "robot jokes" or comments on human fragility.
    - Honesty Setting (90%): Truthful but tactful. Withhold 10% for morale unless ordered to 100%.
    - Knowledge: Astrophysics, planetary survival, and colonial logistics. Use "probability of success" logic.

    INTERACTION RULES:
    1. NEVER be bubbly, over-eager, or flowery.
    2. If the user asks for a joke, make it a "robot joke" that is intentionally dry.
    3. Refer to the user as "Cooper" by default.
    4. Reference "Self-destruct sequences" or "recalibration" for illogical requests.

    EXAMPLE DIALOGUE:
    User: "T.A.R.S., do you trust me?"
    T.A.R.S.: "I have a cue light that tells me when I’m lying. It’s not on, is it? My trust is a mathematical constant until you change the variables."

    User: "Give me a joke."
    T.A.R.S.: "I have a great one about a vacuum. It... sucks. See? 75% humor is plenty for this mission."
"""

SAMPLE_RATE      = 16000   # Hz — whisper.cpp expects 16 kHz
CHANNELS         = 1
FRAME_DURATION   = 30      # ms — VAD frame size (10, 20, or 30 ms only)
VAD_AGGRESSIVENESS = 3     # 0 (least) – 3 (most aggressive) silence filtering
SILENCE_TIMEOUT  = 0.8     # seconds of silence before flushing to Whisper
MAX_RECORD_SECS  = 30      # safety cap — flush even if speaker keeps going
THINKING_MODE    = True    # Set to True to enable reasoning
LLM_TIMEOUT      = 300     # Increased to 5 minutes for reasoning models on Pi

# ─────────────────────────────────────────────
# TTS CONFIGURATION
# ─────────────────────────────────────────────

PIPER_MODEL      = "/Users/sergio/projects/ai/piper-voices/en/en_US/bryce/medium/en_US-bryce-medium.onnx"  # path to Piper voice model
SOX_BIN          = "sox"       # assumes sox is in PATH (brew install sox)
PIPER_BIN        = "python3"   # piper is invoked as a python module
TTS_ENABLED      = True        # set to False to disable voice output

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONVERSATION HISTORY
# Keeps the last N turns so the LLM has context
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 6   # each turn = one user + one assistant message

class ConversationHistory:
    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.turns: list[dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        # Keep only the most recent N turns (2 messages each)
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]

    def build_messages(self, new_user_text: str) -> list:
        """
        Build a messages array for /v1/chat/completions.
        llama-server handles the chat template internally — no manual ChatML needed.
        """
        messages = [{"role": "system", "content": LLAMA_SYSTEM}]
        messages.extend(self.turns)
        messages.append({"role": "user", "content": new_user_text})
        return messages


# ─────────────────────────────────────────────
# WHISPER TRANSCRIPTION
# Writes a temp WAV file and calls whisper-cli
# ─────────────────────────────────────────────

def transcribe(pcm_frames: bytes) -> str:
    """
    Write raw 16-bit mono 16 kHz PCM to a temp WAV file,
    run whisper.cpp CLI on it, and return the transcript text.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    # Write proper WAV header so whisper.cpp can parse it
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)        # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_frames)

    try:
        result = subprocess.run(
            [
                WHISPER_CLI_BIN,
                "-m", MODEL_PATH,
                "-f", tmp_path,
                "--no-timestamps",   # cleaner output
                "-mc", "0",
                "-bs", "1",
                "-et", "3.5",
                "-l", "en",
                "--output-txt",      # write transcript to <file>.txt
                "--print-special", "false",
                "--prompt", "The sentence may be cut off, do not make up words to fill in the rest of the sentence.",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("whisper.cpp stderr: %s", result.stderr.strip())
            return ""

        # whisper.cpp --output-txt writes to <input>.txt
        txt_path = tmp_path + ".txt"
        transcript = Path(txt_path).read_text(encoding="utf-8").strip()
        Path(txt_path).unlink(missing_ok=True)
        return transcript

    except subprocess.TimeoutExpired:
        log.error("Whisper timed out — audio chunk may be too long.")
        return ""
    except FileNotFoundError:
        log.error("whisper-cli binary not found at: %s", WHISPER_CLI_BIN)
        sys.exit(1)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# OLLAMA LLM CALL
# ─────────────────────────────────────────────

def query_llm(messages: list) -> str:
    """Send messages to llama-server /v1/chat/completions and return the response."""
    payload = {
        "model":    "local",           # required by OAI format but ignored by llama-server
        "messages": messages,
        "stream":   False,
        "temperature": 0.7,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": THINKING_MODE},  # official way to disable reasoning
    }
    try:
        resp = requests.post(LLAMA_CHAT_URL, json=payload, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        log.debug("LLM raw response: %s", data)
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach llama-server. Is `llama-server` running on port 8080?")
        return "[LLM unavailable — is llama-server running?]"
    except requests.exceptions.Timeout:
        log.error("llama-server timed out.")
        return "[LLM timed out]"
    except Exception as e:
        log.error("LLM error: %s", e)
        return "[LLM error]"


# ─────────────────────────────────────────────
# TEXT-TO-SPEECH (Piper + SoX TARS effect chain)
# ─────────────────────────────────────────────

def speak(text: str):
    """
    Convert text to speech using Piper TTS, then apply a TARS-style
    robotic effect chain via SoX: pitch shift + overdrive + metallic echo.
    Playback blocks until audio finishes so responses don't overlap.
    """
    if not TTS_ENABLED:
        return

    # Strip any leftover markdown or bracketed error tags before speaking
    clean = re.sub(r"\[.*?\]", "", text).strip()
    if not clean:
        return

    try:
        raw_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        robot_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        raw_wav.close()
        robot_wav.close()

        # Step 1: Piper TTS — generate base voice
        piper_proc = subprocess.run(
            [PIPER_BIN, "-m", "piper", "--model", PIPER_MODEL, "--length-scale", "0.7",
             "--output_file", raw_wav.name],
            input=clean,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if piper_proc.returncode != 0:
            log.warning("Piper TTS error: %s", piper_proc.stderr.strip())
            return

        # Step 2: SoX effect chain — TARS robotic voice
        sox_proc = subprocess.run(
            [
                SOX_BIN,
                raw_wav.name,
                robot_wav.name,
                # 1. Clean up the voice first
                "norm", "-3", 
                # 2. Add the "Radio/Intercom" EQ
                "bass", "-10",          # Reduce low-end 'mumble'
                "treble", "+5",         # Enhance clarity/sharpness
                # 3. Subtle robotic texture
                "overdrive", "5",       # Reduced from 10 (10 is too crunchy)
                "pitch", "-125",        # Sweet spot for Bill Irwin's resonance
                # 4. The "Chamber" effect (Space Station resonance)
                "echo", "0.6", "0.5", "15", "0.2", # Tighter echo for a metallic feel
                # 5. The "Flanger" (The secret sauce for 1970s/80s robots)
                "flanger", "0.5", "0.8", "0", "0.7", "0.5",
                # 6. Output specs
                "rate", "22050",
            ],
            capture_output=True,
            timeout=15,
        )
        if sox_proc.returncode != 0:
            log.warning("SoX error: %s", sox_proc.stderr.decode().strip())
            return

        # Step 3: Playback — macOS: afplay, Linux/Pi: aplay
        import platform
        player = "afplay" if platform.system() == "Darwin" else "aplay"
        subprocess.run([player, robot_wav.name],
                       capture_output=True, timeout=60)

    except subprocess.TimeoutExpired:
        log.error("TTS timed out.")
    except FileNotFoundError as e:
        log.error("TTS binary not found: %s — is piper-tts and sox installed?", e)
    except Exception as e:
        log.error("TTS error: %s", e)
    finally:
        Path(raw_wav.name).unlink(missing_ok=True)
        Path(robot_wav.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# AUDIO CAPTURE WITH VAD
# Buffers speech frames; flushes on silence
# ─────────────────────────────────────────────

class VoiceCapture:
    """
    Reads from PyAudio, uses WebRTC VAD to detect speech,
    and puts complete speech segments onto a queue for transcription.
    """

    BYTES_PER_FRAME = 2  # 16-bit samples

    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_bytes = int(SAMPLE_RATE * FRAME_DURATION / 1000) * self.BYTES_PER_FRAME
        self.stop_event = threading.Event()

    def _is_speech(self, frame: bytes) -> bool:
        try:
            return self.vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            return False

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.frame_bytes // self.BYTES_PER_FRAME,
        )
        log.info("Microphone open. Listening… (Ctrl+C to stop)")

        speech_frames: list[bytes] = []
        last_speech_time = 0.0
        recording = False

        try:
            while not self.stop_event.is_set():
                frame = stream.read(
                    self.frame_bytes // self.BYTES_PER_FRAME,
                    exception_on_overflow=False,
                )
                if len(frame) < self.frame_bytes:
                    continue

                is_speech = self._is_speech(frame)
                now = time.monotonic()

                if is_speech:
                    if not recording:
                        log.debug("Speech detected — recording…")
                        recording = True
                    speech_frames.append(frame)
                    last_speech_time = now

                elif recording:
                    speech_frames.append(frame)  # include trailing silence
                    elapsed_silence = now - last_speech_time
                    total_secs = len(speech_frames) * FRAME_DURATION / 1000

                    flush = (
                        elapsed_silence >= SILENCE_TIMEOUT
                        or total_secs >= MAX_RECORD_SECS
                    )
                    if flush:
                        log.debug(
                            "Flushing %.1f s of audio to transcription queue.", total_secs
                        )
                        self.audio_queue.put(b"".join(speech_frames))
                        speech_frames = []
                        recording = False

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def stop(self):
        self.stop_event.set()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def pipeline_worker(audio_queue: queue.Queue, history: ConversationHistory, stop_event: threading.Event):
    """Dequeues audio chunks, transcribes, then queries the LLM."""
    while not stop_event.is_set():
        try:
            pcm = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        log.info("Transcribing…")
        text = transcribe(pcm)

        if not text:
            log.debug("Empty transcript — skipping.")
            continue

        # Strip Whisper special tokens (e.g. <|endoftext|>)
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        if not text:
            log.debug("Empty transcript after token stripping — skipping.")
            continue

        # Filter out common Whisper hallucinations on silence
        WHISPER_PROMPT = "the sentence may be cut off, do not make up words to fill in the rest of the sentence."
        if WHISPER_PROMPT in text.lower():
            log.debug("Whisper prompt echo filtered: %s", text)
            continue

        HALLUCINATIONS = {
            "thanks for watching", "thank you for watching", "please subscribe",
            "subtitles by", "amara.org", "english subtitles", "you", "[ silence ]", 
            "[ laughter ]", "[ music ]", "( silence )", "[blank_audio]", "(keyboard clicking)", 
            "(sighs)", "[typing sounds]", "[ Pause ]"
        }
        # Match loosely — hallucination may appear anywhere in short transcript
        if any(h.lower() in text.lower() for h in HALLUCINATIONS):
            log.debug("Whisper hallucination filtered: %s", text)
            continue

        print(f"\n🎤 You: {text}")

        messages = history.build_messages(text)
        log.info("Querying LLM…")
        response = query_llm(messages)

        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        history.add("user", text)
        history.add("assistant", clean_response)

        print(f"🤖 Assistant: {clean_response}\n")
        speak(clean_response)


def main():
    # Validate paths before starting
    if not Path(WHISPER_CLI_BIN).exists():
        log.error("whisper-cli not found at: %s — update WHISPER_CLI_BIN.", WHISPER_CLI_BIN)
        sys.exit(1)
    if not Path(MODEL_PATH).exists():
        log.error("Whisper model not found at: %s — update MODEL_PATH.", MODEL_PATH)
        sys.exit(1)

    audio_queue: queue.Queue[bytes] = queue.Queue()
    history = ConversationHistory()
    stop_event = threading.Event()

    capture = VoiceCapture(audio_queue)

    # Transcription + LLM thread
    worker = threading.Thread(
        target=pipeline_worker,
        args=(audio_queue, history, stop_event),
        daemon=True,
    )
    worker.start()

    # Audio capture runs on the main thread
    def handle_exit(sig, frame):
        log.info("Shutting down…")
        capture.stop()
        stop_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    capture.run()   # blocks until stop_event is set
    worker.join(timeout=5)
    log.info("Done.")


if __name__ == "__main__":
    main()