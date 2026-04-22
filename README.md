> pip install faster-whisper silero-vad torch numpy onnxruntime

Python version: 3.10 or 3.11 (faster-whisper has issues with 3.12+)

pip install faster-whisper silero-vad pyaudio numpy torch requests

brew install sox portaudio

sudo apt install sox portaudio19-dev

pip install piper-tts

Quick sanity check — after installing, verify the key ones:
bashpython -c "import faster_whisper, silero_vad, pyaudio, torch; print('OK')"
sox --version

## Related doc

* https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
* https://huggingface.co/unsloth/Qwen3.5-9B-GGUF

## SPIN UP LLAMA SERVER - ALL PARAMETERS INDICATED

```bash
llama-server \
  -m /Users/sergio/projects/ai/models/Qwen3.5-9B-Q4_K_M.gguf \
  -c 100000 \
  --jinja \
  --threads 10 \
  --temp 0.8 \
  --min-p 0.06 \
  --presence-penalty 1.2 \
  --repeat-penalty 1.05 \
  --reasoning-budget-message "[Reasoning budget exhausted. Getting the final answer...]"
```

## SPIN UP LLAMA SERVER - MINIMAL PARAMETERS + COMPLETION API

```bash
llama-server \
  -m /Users/sergio/projects/ai/models/Qwen3.5-9B-Q4_K_M.gguf \
  -c 100000 \
  --jinja \
  --threads 10 \
  --reasoning-budget-message "[Reasoning budget exhausted. Getting the final answer...]"
```

```bash
{
  "messages": [
    {
      "role": "system",
      "content": "<your-system-prompt>"
    },
    {
      "role": "user",
      "content": "solve this summation: $$ \\sum\\limits_{i=0}^{n} \\sum\\limits_{j=i}^{n} c $$"
    }
  ],
  "stream": false,
  "temperature": 0.8,
  "min_p": 0.06,
  "presence_penalty": 1.2,
  "repeat_penalty": 1.05,
  "thinking_budget_tokens": 512,
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

## How to extract GGUF system template from models

```bash
python3 -c 'from gguf import GGUFReader; reader = GGUFReader("Qwen3.5-9B-Q4_K_M.gguf"); print(reader.get_field("tokenizer.chat_template").parts[-1].tobytes().decode())'
```

# TODO

- [x] make sure the script is cleaning all the *.wav files when finish

- [ ] create a "proxy" to evaluate if enable or not Thinking process.
    - if so, tell to the user "I'm thinking, be patient..."
