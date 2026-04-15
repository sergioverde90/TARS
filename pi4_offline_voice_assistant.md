# Offline Voice Assistant — Raspberry Pi 4
### Microphone → Whisper → Qwen3.5-0.8B via llama-server

> **Goal:** A fully local, internet-independent voice assistant. No cloud, no API keys,
> no data leaving the device. All processing runs on the Pi 4 after a one-time setup.

---

## Hardware Requirements

| Component | Recommendation | Notes |
|-----------|----------------|-------|
| Raspberry Pi | Pi 4 — **4GB or 8GB RAM** | 4GB works; 8GB gives more headroom |
| Microphone | USB microphone (e.g. Blue Yeti, HyperX SoloCast) | Better quality & lower latency than 3.5mm |
| Storage | SD card ≥ 16GB or USB drive | Models alone need ~2–3 GB |
| Power | Official Pi 4 USB-C PSU (5.1V / 3A) | Underpowering causes random crashes |

---

## Technology Stack

| Layer | Tool | Why |
|-------|------|-----|
| Audio capture + VAD | `PyAudio` + `webrtcvad` | Captures mic input; only processes frames where speech is detected |
| Speech-to-text | `whisper.cpp` | Fast, offline Whisper inference on ARM64; no Python overhead |
| LLM inference server | `llama-server` (llama.cpp) | Leaner than Ollama on Pi 4; correct API for Qwen3.5 GGUFs |
| LLM model | **Qwen3.5-0.8B-Instruct Q4_K_M** | Best quality/RAM ratio for Pi 4; ~600MB; built for edge devices |
| Orchestration script | Python (`transcribe.py`) | Ties everything together; manages conversation history |

---

## Model Choice — Qwen3.5-0.8B

This model was released March 2, 2026 and is purpose-built for edge deployment.

**Why it fits the Pi 4:**
- Q4_K_M quantized weight is ~600 MB — leaves ~2.5 GB free for OS + whisper.cpp
- Hybrid Gated DeltaNet architecture: uses 3 linear attention layers per 1 full attention layer,
  which keeps memory usage flat and throughput high on CPU
- Explicitly designed for "tiny, fast, edge device" use (Alibaba's own description for the 0.8B/2B tier)
- 262K native context window — overkill for voice, but means no context-related bugs

**Known limitation:** Accuracy degrades on code generation tasks with few-shot prompts.
For a conversational voice assistant this is irrelevant.

**If 0.8B feels too limited:** step up to `Qwen3.5-2B-Instruct Q4_K_M` (~1.4 GB) —
it fits comfortably on a 4GB Pi 4 and gives noticeably better reasoning.

> ⚠️ Qwen3.5 GGUFs do **not** work with Ollama as of early 2026 due to separate
> multimodal projection files. Use `llama-server` directly.

---

## One-Time Setup

### 1. System dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    cmake \
    build-essential \
    git \
    python3-pip
```

### 2. Build whisper.cpp

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make -j$(nproc)                         # builds 'main' and other binaries
sudo cp main /usr/local/bin/whisper-cli
sudo chmod +x /usr/local/bin/whisper-cli
cd ~
```

Download the model (do this on a machine with internet, then copy via USB or SCP):

```bash
# On the Pi or via USB transfer — tiny.en is fastest, base.en is more accurate
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin \
     -O /home/pi/models/ggml-base.en.bin
```

> Use `ggml-tiny.en.bin` if transcription latency is too high.
> Tiny: ~50–100 wpm real-time; Base: ~20–30 wpm real-time on Pi 4.

### 3. Build llama.cpp and llama-server

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF         # CPU-only for Pi 4
cmake --build build --config Release -j$(nproc)
sudo cp build/bin/llama-server /usr/local/bin/llama-server
cd ~
```

### 4. Download the Qwen3.5-0.8B model

Download on a machine with internet access, then copy to the Pi:

```bash
# Find the GGUF on HuggingFace:
# https://huggingface.co/collections/Qwen/qwen35-...
# Search for: Qwen3.5-0.8B-Instruct-GGUF  →  pick Q4_K_M variant

# Place it at:
mkdir -p /home/pi/models
# scp or copy via USB to:
# /home/pi/models/qwen3.5-0.8b-instruct-q4_k_m.gguf
```

### 5. Python dependencies

```bash
pip3 install pyaudio webrtcvad requests --break-system-packages
```

---

## Running the System

Two processes must be running simultaneously. Use two terminal sessions (or SSH tabs),
or configure both as systemd services (see below).

### Terminal 1 — start llama-server

```bash
llama-server \
    -m /home/pi/models/qwen3.5-0.8b-instruct-q4_k_m.gguf \
    --port 8080 \
    --ctx-size 1024 \
    --threads 4
```

| Flag | Value | Why |
|------|-------|-----|
| `--ctx-size` | 1024 | Saves RAM; enough for voice assistant turns |
| `--threads` | 4 | Pi 4 has 4 cores — use all of them |

Wait for the log line: `llama server listening at http://0.0.0.0:8080`

### Terminal 2 — start the voice pipeline

```bash
python3 transcribe.py
```

You should see:
```
Microphone open. Listening… (Ctrl+C to stop)
```

Speak naturally. After a ~1.2 s pause the audio is sent to Whisper, then the transcript
goes to the LLM. Responses appear in the terminal:

```
🎤 You: what's the capital of Portugal
🤖 Assistant: The capital of Portugal is Lisbon.
```

Press `Ctrl+C` to stop gracefully.

---

## The Pipeline Script (`transcribe.py`)

Full script is provided separately. Key design decisions explained:

**Voice Activity Detection (VAD)**
Uses `webrtcvad` at aggressiveness level 2. Only frames where speech is detected are
buffered. After 1.2 seconds of silence the buffer is flushed to Whisper. This avoids
sending silence to Whisper on every chunk (which would be very slow on Pi 4).

**Whisper integration**
Writes buffered PCM frames to a temp WAV file with a proper header, then calls
`whisper-cli` via subprocess with `--output-txt`. The CLI writes a `.txt` sidecar file
which is read back and cleaned up. This is the correct way to use the whisper.cpp CLI —
it does not accept raw PCM on stdin.

**Conversation history**
A `ConversationHistory` class keeps the last 6 turns (12 messages) and builds a full
context prompt for each LLM call. This gives the assistant memory across turns without
exceeding the 1024-token context window.

**Hallucination filtering**
Whisper occasionally outputs `[BLANK_AUDIO]`, `(silence)`, or `Thanks for watching.`
on silent input. These are filtered before hitting the LLM.

**llama-server API**
`POST http://localhost:8080/completion` with flat JSON body:
```json
{
  "prompt": "...",
  "stream": false,
  "temperature": 0.7,
  "n_predict": 200
}
```
Response text is in the `"content"` key. This differs from Ollama (`"response"` key,
nested `"options"`, required `"model"` field).

---

## Performance Expectations (Pi 4, 4GB RAM)

| Component | Expected performance |
|-----------|---------------------|
| Whisper tiny.en | ~1–2 s latency for a short sentence |
| Whisper base.en | ~3–5 s latency for a short sentence |
| Qwen3.5-0.8B Q4_K_M | ~6–10 tokens/second on Pi 4 CPU |
| End-to-end (tiny + 0.8B) | ~5–8 s from end of speech to first word of reply |

For a voice assistant, 5–8 s is acceptable. If it feels too slow:
- Switch to `ggml-tiny.en.bin` for Whisper
- Reduce `--ctx-size` to 512
- Reduce `n_predict` to 100 in `transcribe.py`

---

## Optional: Run as systemd Services

To have both processes start automatically on boot:

**`/etc/systemd/system/llama-server.service`**
```ini
[Unit]
Description=llama-server LLM inference
After=network.target

[Service]
ExecStart=/usr/local/bin/llama-server \
    -m /home/pi/models/qwen3.5-0.8b-instruct-q4_k_m.gguf \
    --port 8080 --ctx-size 1024 --threads 4
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/voice-assistant.service`**
```ini
[Unit]
Description=Voice Assistant Pipeline
After=llama-server.service
Requires=llama-server.service

[Service]
ExecStart=/usr/bin/python3 /home/pi/transcribe.py
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

Enable both:
```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-server voice-assistant
sudo systemctl start llama-server voice-assistant
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `whisper-cli: not found` | Binary not in PATH | Run `sudo cp main /usr/local/bin/whisper-cli` |
| `OSError: [Errno -9996]` from PyAudio | Wrong audio device | Run `python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"` to list devices |
| Empty transcripts every time | Mic not recording / wrong sample rate | Check mic with `arecord -d 3 test.wav && aplay test.wav` |
| `Cannot reach llama-server` | Server not started or still loading | Wait for `listening at http://0.0.0.0:8080` in server log |
| LLM very slow (< 3 tok/s) | Thermal throttling | Add heatsink/fan; check `vcgencmd measure_temp` |
| `[BLANK_AUDIO]` printed frequently | VAD aggressiveness too low | Increase `VAD_AGGRESSIVENESS` to 3 in `transcribe.py` |

---

## Upgrade Path

| Upgrade | Benefit |
|---------|---------|
| Switch to Qwen3.5-2B Q4_K_M | Better reasoning, ~1.4 GB, still fits 4GB Pi 4 |
| Raspberry Pi 5 (8GB) | ~2× faster inference; unlocks Qwen3.5-4B comfortably |
| Add a coral TPU or Hailo-8 HAT | Offload Whisper to dedicated hardware; cuts STT latency to < 500 ms |
| Text-to-speech output | Add `piper-tts` (fast, offline, runs well on Pi 4) to speak responses aloud |

---

*All components are open-source and Apache 2.0 licensed. No data leaves the device.*
