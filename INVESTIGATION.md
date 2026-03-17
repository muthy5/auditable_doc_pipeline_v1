# Investigation: API Issues and Alternatives

## Problem

The Claude API backend may fail for several reasons. This document investigates
root causes and documents available alternatives.

## Root Causes for Claude API Failures

### 1. Missing API Key (most common)

The `ClaudeAPIBackend` requires an Anthropic API key via:
- `--claude-api-key` CLI argument, or
- `ANTHROPIC_API_KEY` environment variable

Without it, the backend raises `ValueError` at initialization
(`src/claude_backend.py:37-40`).

**Fix:** Set the environment variable before running:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Missing `anthropic` Python Package

The package `anthropic>=0.39.0` is listed in `requirements.txt` but marked as
optional. If not installed, `ClaudeAPIBackend.__init__` raises `BackendError`.

**Fix:**
```bash
pip install anthropic>=0.39.0
```

### 3. API Rate Limits or Network Errors

The backend retries up to 3 attempts (1 initial + 2 retries) for transient
errors like JSON parse failures or API errors (`src/claude_backend.py:68-127`).
However, it does **not** implement exponential backoff between retries for API
rate limit errors (HTTP 429) — it only retries on response parsing failures.

### 4. Model Availability

The default model is `claude-sonnet-4-20250514` (`src/claude_backend.py:21`).
If this model is deprecated or unavailable on your API plan, calls will fail.

## Existing Alternatives (Built In)

The pipeline supports three backends via `--backend {demo|ollama|claude}`:

| Backend  | API Key Required | Internet Required | Notes                          |
|----------|-----------------|-------------------|--------------------------------|
| `demo`   | No              | No                | Rule-based heuristics; no LLM  |
| `ollama` | No              | No                | Local LLM server required       |
| `claude` | Yes             | Yes               | Best quality; requires API key  |

### Using Ollama (Recommended Local Alternative)

1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull llama3.1` or `ollama pull mistral`
3. Run the pipeline:
   ```bash
   python -m src --backend ollama --ollama-model llama3.1 --input doc.txt
   ```

### Using Demo Backend (No Setup Required)

```bash
python -m src --backend demo --input doc.txt
```

This uses rule-based heuristics and requires no external dependencies, but
produces lower-quality analysis results.

## Potential New API Alternatives

### OpenAI-Compatible Backend

Many providers expose OpenAI-compatible APIs. Adding an OpenAI backend would
unlock:

- **OpenAI GPT-4o** — Comparable quality to Claude
- **Azure OpenAI** — Enterprise-grade, same API shape
- **OpenRouter** — Single API key for 100+ models (Claude, GPT-4, Gemini, Llama, etc.)
- **Local servers** — vLLM, llama.cpp, LM Studio all expose OpenAI-compatible endpoints

Implementation would follow the same `LocalLLMBackend` interface pattern already
used by `ClaudeAPIBackend` and `OllamaLocalBackend`.

### Google Gemini

The `google-generativeai` package provides access to Gemini models. Competitive
with Claude for structured JSON generation tasks.

## Recommendation

1. **Short-term:** Use `ollama` backend with a capable local model (llama3.1,
   mistral, or codellama) for zero-cost, no-API-key operation.
2. **Medium-term:** Add an OpenAI-compatible backend to support the broadest
   range of providers with a single implementation.
3. **Long-term:** Consider OpenRouter integration for maximum model flexibility
   behind a single API key.
