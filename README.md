> pip install faster-whisper silero-vad torch numpy onnxruntime

Python version: 3.10 or 3.11 (faster-whisper has issues with 3.12+)

pip install faster-whisper silero-vad pyaudio numpy torch requests

brew install sox portaudio

sudo apt install sox portaudio19-dev

pip install piper-tts

Quick sanity check — after installing, verify the key ones:
bashpython -c "import faster_whisper, silero_vad, pyaudio, torch; print('OK')"
sox --version

spin up llama-server

```bash
llama-server -m ../ai/models/Qwen3.5-9B-Q4_K_M.gguf --reasoning-budget 0 -c 1024 --cache-reuse 256 --reasoning-format none
```

# How to start

```bash
llama-server -m /path/to/model.gguf --reasoning-budget 0 -c 100000 --jinja
```

# TODO

- [x] make sure the script is cleaning all the *.wav files when finish

- [ ] create a "proxy" to evaluate if enable or not Thinking process.
    - if so, tell to the user "I'm thinking, be patient..."

- [ ] use `--reasoning-format none` when `llama-server` spin up instead of stripping `<thinking>` annotation    

- [ ] the ninja template is not working properly with /no-thinking