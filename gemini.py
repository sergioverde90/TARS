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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CORE MODELS (Loaded once to stay in RAM)
# ─────────────────────────────────────────────

log.info("Loading Models into RAM...")
# Faster-Whisper stays 'hot' in memory to avoid subprocess lag
whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
# Silero VAD is neural-net based: ignores fans, keyboards, and hums
vad_model = load_silero_vad()

# ─────────────────────────────────────────────
# LOGIC COMPONENTS (History, TTS, etc.)
# ─────────────────────────────────────────────

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

def speak(text):
    if not TTS_ENABLED: return
    clean = re.sub(r"\[.*?\]", "", text).strip()
    if not clean: return
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw_wav, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as robot_wav:
            
            # Piper TTS
            subprocess.run([PIPER_BIN, "-m", "piper", "--model", PIPER_MODEL, "--length-scale", "0.7",
                             "--output_file", raw_wav.name], input=clean, text=True, capture_output=True)

            # TARS Robotic SoX Chain
            subprocess.run([SOX_BIN, raw_wav.name, robot_wav.name, "norm", "-3", "bass", "-10", "treble", "+5",
                            "overdrive", "5", "pitch", "-125", "echo", "0.6", "0.5", "15", "0.2",
                            "flanger", "0.5", "0.8", "0", "0.7", "0.5", "rate", "22050"], capture_output=True)

            player = "afplay" if platform.system() == "Darwin" else "aplay"
            subprocess.run([player, robot_wav.name], capture_output=True)
            
            Path(raw_wav.name).unlink(missing_ok=True)
            Path(robot_wav.name).unlink(missing_ok=True)
    except Exception as e:
        log.error(f"TTS Error: {e}")

# ─────────────────────────────────────────────
# THE NEW PIPELINE
# ─────────────────────────────────────────────

def pipeline_worker(audio_queue, history, stop_event):
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        log.info("Transcribing...")
        # Direct inference from RAM - no subprocess!
        segments, _ = whisper_model.transcribe(audio_data, beam_size=5, vad_filter=False)
        text = " ".join([s.text for s in segments]).strip()

        if not text or len(text) < 2:
            continue

        print(f"\n🎤 Cooper: {text}")
        
        # LLM Call
        payload = {"model": "local", "messages": history.build_messages(text), "stream": False}
        try:
            resp = requests.post(LLAMA_CHAT_URL, json=payload, timeout=60)
            response_text = resp.json()["choices"][0]["message"]["content"].strip()
            clean_response = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            
            history.add("user", text)
            history.add("assistant", clean_response)
            
            print(f"🤖 T.A.R.S.: {clean_response}\n")
            speak(clean_response)
        except Exception as e:
            log.error(f"LLM Error: {e}")

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
                audio_float32 = audio_int16.astype(np.float32) / 32768.0 # Normalize for Silero
                
                # Check for speech using Silero
                speech_dict = self.vad_iterator(torch.from_numpy(audio_float32), return_seconds=True)
                
                if speech_dict:
                    if 'start' in speech_dict:
                        log.debug("Speech Started")
                        audio_buffer = [] # Reset buffer for new sentence
                    
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