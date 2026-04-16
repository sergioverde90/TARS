> pip install faster-whisper silero-vad torch numpy onnxruntime

Python version: 3.10 or 3.11 (faster-whisper has issues with 3.12+)

pip install faster-whisper silero-vad pyaudio numpy torch requests

brew install sox portaudio

sudo apt install sox portaudio19-dev

pip install piper-tts

Quick sanity check — after installing, verify the key ones:
bashpython -c "import faster_whisper, silero_vad, pyaudio, torch; print('OK')"
sox --version