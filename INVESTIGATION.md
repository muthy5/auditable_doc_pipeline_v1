# Investigation: Why the Pipeline "Is Not Working"

This repository now includes preflight diagnostics that catch most setup/runtime
issues before execution. When the pipeline appears "not working," it is usually
one (or more) of the checks below.

## 1) Backend not configured correctly

### Claude backend
- Missing `anthropic` package.
- Missing `ANTHROPIC_API_KEY` (or `--claude-api-key`).
- Model unavailable for your account/region.

### OpenAI-compatible backend
- Missing OpenAI API key (`OPENAI_API_KEY` or `--openai-api-key`).
- Endpoint/model mismatch (provider does not serve the requested model).

### Ollama backend
- Ollama server is not reachable at `--ollama-base-url`.
- Requested `--ollama-model` is not installed locally.

## 2) Optional features enabled without required credentials

- `--enable-search` set without `BRAVE_API_KEY` (or `--brave-api-key`).

## 3) Input format/parsing problems

- PDF input without `pypdf` installed.
- DOCX input without `python-docx` installed.
- Image-only/scanned PDFs (OCR not included by default).
- Empty or unreadable input files.

## 4) Environment/runtime constraints

- No network access for cloud APIs (Claude/OpenAI/Brave).
- API rate limiting or transient provider errors.
- Local runtime memory/timeout limits for large docs or local models.

## 5) Strict-mode behavior looks like "failure"

With `--strict`, any schema-validation failure stops execution immediately. In
non-strict mode, the run can continue with fallback artifacts, which may look
partial but is expected behavior.

## 6) Quick recovery checklist

1. Run tests (`pytest -q`) to confirm local code health.
2. Use `--backend demo` first to verify core pipeline wiring.
3. If using Claude/OpenAI, verify package + API key + model name.
4. If using Ollama, confirm server reachability and model installation.
5. If enabling search, set `BRAVE_API_KEY`.
