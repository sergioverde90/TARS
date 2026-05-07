# TARS — Agent Quick Reference

## Architecture

Two services + one voice pipeline:

| Service | Tech | Port | Role |
|---|---|---|---|
| **mlx-openai-server** | Unsloth MLX binary | 8080 | OpenAI-compatible LLM server (`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`, ~263k context). Binary at `~/.unsloth/unsloth_qwen3_6_mlx/bin/mlx-openai-server` |
| **tars-backend** | Spring Boot 3.5 / Java 25 + langchain4j | 8000 | `/v1/chat/completions` SSE endpoint with tool calling (5 tools). Entry: `TarsApplication.java` |
| **transcribe-stream.py** | Python 3.10–3.11 | — | Voice pipeline: Silero VAD → Faster-Whisper → LLM → Piper TTS + SoX effects |

The voice pipeline calls `localhost:8081/v1/chat/completions` for the Java backend (tool calling). Swap to `:8080` to bypass Java and hit mlx-openai-server directly.

## Setup

```bash
# Python voice pipeline (3.10 or 3.11 ONLY — faster-whisper broken on 3.12+)
pip install faster-whisper silero-vad pyaudio numpy torch requests
brew install sox portaudio          # macOS
# sudo apt install sox portaudio19-dev   # Linux

# Java backend
cd tars-backend && mvn spring-boot:run
# Requires .env with ORS_API_KEY and TAVILY_API_KEY (copy from .env.example)
```

## Running

```bash
# 1. Start mlx-openai-server (downloads model from HuggingFace on first run)
~/.unsloth/unsloth_qwen3_6_mlx/bin/mlx-openai-server

# 2. Start Java backend (provides tool calling via langchain4j)
cd tars-backend && mvn spring-boot:run

# 3. Run voice pipeline
python3 transcribe-stream.py              # mic mode (default)
python3 transcribe-stream.py --ttl        # terminal mode (no mic, type to chat)
python3 transcribe-stream.py --no-stream  # non-streaming LLM responses
python3 transcribe-stream.py --debug      # verbose logging
```

## Key Constraints

- **Python version**: 3.10 or 3.11. `faster-whisper` fails on 3.12+.
- **HuggingFace auth**: mlx-openai-server needs a valid HF token to pull `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`.
- **MLX model path**: `~/.unsloth/unsloth_qwen3_6_mlx/` (venv + model cache).
- **Conversation state**: `transcribe-stream.py` uses `ConversationHistory` (max 6 turns). `</closing>` tag in LLM response resets conversation. `</not-me>` filters out irrelevant LLM output.
- **TTS output rules** (enforced in `LLAMA_SYSTEM` prompt): no Markdown, no special chars, spell out dates/units, 2 sentences max, no pleasantries.
- **Echo prevention**: `is_speaking` threading event blocks mic capture while TARS is speaking.
- **TTS render-ahead**: Piper + SoX rendering happens in background while previous chunk plays. First chunk renders synchronously (visible latency).
- **Java tools**: `TimeTool`, `DistanceTool` (OpenRouteService geocode + directions), `SearchTool` (Tavily), `WebsiteScrappingTool`, `YahooFinanceTool`. Max 5 tool-call iterations per request.
- **Java config**: `TarsProperties` loads `tars.openrouteservice.*` and `tars.tavily.*` from Spring config / env.

## File Map

```
transcribe-stream.py          # Voice pipeline (main entrypoint)
tars-backend/                 # Spring Boot + langchain4j
  src/main/java/com/tars/
    TarsApplication.java      # Entry
    api/ChatController.java   # /v1/chat/completions, /health
    chat/StreamChatCompletionService.java  # SSE streaming + tool loop
    config/TarsConfiguration.java          # Tool bean wiring
    config/TarsProperties.java             # External API config
    tools/                          # 5 tool implementations
cache/cache.safetensors           # MLX model cache
logs/app.log                      # mlx-openai-server logs
```

## Gotchas

- mlx-openai-server has a 300s RPC timeout — long reasoning prompts can trigger `TimeoutError`.
- The mlx-openai-server ignores `repeat_penalty`, `thinking_budget_tokens`, `preserve_thinking` fields (logged as warnings).
- `transcribe-stream.py` system prompt is embedded in-code (not externalized). Editing TARS personality requires modifying `transcribe-stream.py` directly.
- `cache/` and `logs/` are NOT in `.gitignore` (only `.qwen` and `.DS_Store`).
- The `distance-tool-guide.md` and `pi4_offline_voice_assistant.md` are planning/prototype docs — not current implementation.
