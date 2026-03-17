# Deep Investigation: Why the Pipeline Fails and How to Fix It

This guide is intentionally detailed. It explains **failure modes**, **how to identify each one**, and the **exact fix path**.

---

## Failure taxonomy (what "not working" usually means)

Most reports map to one of these buckets:

1. **Startup/preflight failures**: missing packages, missing keys, or missing backend prerequisites.
2. **Transport failures**: DNS, TLS, firewall, proxy, or endpoint connectivity issues.
3. **Provider/API failures**: auth errors, quota/rate limits, model not found, payload too large.
4. **Parsing failures**: LLM output is non-JSON/partial JSON, extraction errors.
5. **Input extraction failures**: unsupported file type, parser missing, scanned PDF, empty text.
6. **Validation/workflow failures**: schema errors, strict-mode stop, or fallback confusion.
7. **Performance failures**: timeouts, context overflow, local model memory pressure.

The sections below break these down by symptom.

---

## 1) Backend and credential configuration failures

### Claude backend (`--backend claude`)

#### Typical root causes
- `anthropic` package is not installed.
- `ANTHROPIC_API_KEY` / `--claude-api-key` missing.
- API key present but invalid/expired.
- requested model not available on account/region.

#### How it manifests
- Immediate startup failure before passes run.
- API errors during first model call.

#### What to check
- `pip show anthropic`
- `echo "$ANTHROPIC_API_KEY" | wc -c` (non-zero)
- provider dashboard for model entitlement

#### Fix
- install dependency, set valid key, and pick a model your plan supports.

### OpenAI-compatible backend (`--backend openai`)

#### Typical root causes
- missing `OPENAI_API_KEY` / `--openai-api-key`
- wrong `OPENAI_BASE_URL` (missing `/v1`, wrong host, stale local URL)
- provider/model mismatch (endpoint alive but model unavailable)

#### How it manifests
- HTTP 401/403 (auth), 404 (wrong route/model), 429 (rate limit), 5xx (provider issues)

#### Fix
- validate key + base URL + model trio together (must all match same provider).

### Ollama backend (`--backend ollama`)

#### Typical root causes
- server not running / unreachable
- model not pulled locally
- remote/container network cannot reach your host Ollama instance

#### How it manifests
- reachability error on `/api/tags`
- model-not-found with available model list

#### Fix
- start Ollama, `ollama pull <model>`, and ensure URL is reachable from runtime.

---

## 2) Optional feature misconfiguration

### Web search enrichment (`--enable-search`)

#### Root cause
- search enabled without `BRAVE_API_KEY`.

#### Fix
- set key or disable search.

---

## 3) Input/document extraction failures

### Unsupported or partially supported content

#### Root causes
- PDF parser package (`pypdf`) missing.
- DOCX parser package (`python-docx`) missing.
- scanned/image-only PDF (OCR not bundled).
- valid file but text extraction returns empty/near-empty content.

#### Why this is common
Users often assume "PDF support" implies OCR support. Here it means text-layer PDFs only unless OCR is added.

#### Fix
- install parser dependencies.
- run OCR externally for image-only documents before ingestion.
- verify extracted text is non-empty prior to pipeline run.

---

## 4) Runtime/network failures

### Transport level

#### Root causes
- DNS resolution failure
- outbound firewall or proxy restrictions
- TLS/cert trust problems
- ephemeral network instability

#### Symptoms
- `URLError`, socket timeout, connection reset, intermittent pass failures.

#### Fix
- verify outbound HTTPS from runtime host/container to provider endpoints.
- configure proxy/CA chain correctly if corporate network is enforced.

### Provider stability and quota

#### Root causes
- 429 rate limit / quota exceeded
- provider transient 5xx

#### Fix
- retry with exponential backoff, reduce concurrency, or upgrade quota.

---

## 5) LLM output and schema validation failures

### Non-JSON or malformed JSON output

#### Root causes
- model drifts from strict JSON instructions
- truncated responses from token limits
- provider returns wrapped markdown/code fences

#### Fix
- strict JSON prompts + extraction heuristics + retry.
- keep schema/payload concise enough to avoid truncation.

### Strict mode confusion (`--strict`)

#### Root cause
- strict mode intentionally halts on first schema validation error.

#### Why it appears "broken"
- run stops early even though fallback behavior exists in non-strict mode.

#### Fix
- use non-strict during diagnostics; enable strict in enforcement workflows.

---

## 6) Performance and scale issues

### Large payloads / long documents

#### Root causes
- token/context overflow
- local model memory constraints
- timeout configuration too aggressive

#### Fix
- reduce input size, chunk more aggressively, or use a larger-context model.
- adjust timeout/retry where provider latency is high.

---

## 7) What was fixed in code now

To reduce real-world failures (especially "works sometimes" behavior), backend retry behavior was hardened:

1. **Claude backend now uses exponential backoff retries** instead of immediate tight-loop retries.
2. **Claude retry classification improved** for rate-limit/timeout/transient transport errors.
3. **OpenAI-compatible backend now preserves HTTP status metadata** in API exceptions.
4. **OpenAI-compatible retry logic now correctly retries transient HTTP codes** (e.g., 429/5xx) because those status codes are no longer hidden inside generic exceptions.
5. Added tests that prove retry-then-success behavior for both Claude and OpenAI-compatible backends.

---

## 8) End-to-end recovery playbook

1. `pytest -q` (confirm local code baseline).
2. Run with `--backend demo` and a known-good text sample.
3. Add backend complexity one layer at a time:
   - first keys/packages
   - then model/base URL
   - then optional search
4. If failures are intermittent, inspect rate-limit/network behavior and rely on built-in backoff.
5. Only use `--strict` once outputs are consistently schema-clean.

