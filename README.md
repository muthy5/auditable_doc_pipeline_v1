# auditable_doc_pipeline_v1

Auditable, chunked document-analysis pipeline with schema-validated multi-pass outputs.

## Repository layout

```text
auditable_doc_pipeline_v1/
  README.md
  CONTRIBUTING.md
  Makefile
  requirements.txt
  ruff.toml
  examples/
  prompts/
  schemas/
  runs/
  src/
    __main__.py
    cli.py
    chunker.py
    claude_backend.py
    config.py
    exceptions.py
    llm_interface.py
    markdown_writer.py
    merge_engine.py
    ollama_backend.py
    pass_runner.py
    pipeline.py
    prompts.py
    report.py
    retry.py
    run_inspector.py
    schemas.py
    validators.py
  tests/
```

## Quick start

```bash
make setup
make test
python -m src --input examples/lemonade_plan_missing_juicing.txt --backend demo --runs-dir runs
```

## Supported document types

The pipeline now supports schema-auditing against multiple document templates:

- `procedural_plan` (default fallback)
- `business_plan`
- `legal_contract`
- `project_proposal`
- `medical_protocol`
- `technical_spec`

Use CLI `--document-type auto` (default) to classify automatically, or set an explicit type.

To add a new type:

1. Add `schemas/document_types/<new_type>.json` with `document_type` and `expected_sections`.
2. Add the new type to `src/document_classifier.py` (`SUPPORTED_DOCUMENT_TYPES`).
3. Update `prompts/classify_document.txt` and `schemas/classify_document.schema.json` allowed values.
4. Add backend heuristics in `src/llm_interface.py` for the demo backend.

## CLI features

- `--strict`: stop on first schema-validation failure.
- `--dry-run`: print pass order/backend plan only; no execution.
- `--resume`: resume existing run (`--run-dir`) from first incomplete pass.
- `--verbose`: set logging level to DEBUG.
- `--quiet`: set logging level to ERROR.
- `--enable-search`: enable optional Brave web-search enrichment (off by default).
- `--brave-api-key`: Brave API key (falls back to `BRAVE_API_KEY`).
- `--reference-dir`: Local folder of `.txt`, `.md`, `.pdf`, and `.docx` files used for retrieval-augmented context.
- `--document-type`: Use `auto` (default) or force a specific supported document type.

## Backends

### ollama

```bash
make run-ollama
```

### claude

Use the Claude API backend with `--backend claude`. Provide credentials via `--claude-api-key` or the `ANTHROPIC_API_KEY` environment variable. The default model is `claude-sonnet-4-20250514` (override with `--claude-model`).


## Web search enrichment (Brave)

You can optionally augment all passes with current web context:

```bash
python -m src \
  --input examples/lemonade_plan_missing_juicing.txt \
  --backend claude \
  --enable-search \
  --brave-api-key "$BRAVE_API_KEY"
```

How it works:
- After `00_normalize_request`, the pipeline generates 3-5 targeted queries (`prompts/search_queries.txt`).
- It calls Brave Web Search and stores normalized results (`title`, `url`, `snippet`).
- Results are injected into subsequent pass payloads as `web_context`.

To get a free Brave API key, create an account in the Brave Search API dashboard and generate a subscription token, then pass it via `--brave-api-key` or export `BRAVE_API_KEY`.


## Local reference retrieval (RAG)

You can enrich the pipeline with local reference documents:

```bash
python -m src \
  --input examples/lemonade_plan_missing_juicing.txt \
  --backend demo \
  --reference-dir ./reference_docs
```

How it works:
- Loads `.txt`, `.md`, `.pdf`, and `.docx` files from `--reference-dir`.
- Reuses pipeline chunking logic to split reference documents into chunks.
- Builds an in-memory TF-IDF index and retrieves top-matching chunks.
- Injects retrieved context into downstream passes as `reference_context`.

Run artifacts include `passes/retrieval_context.json` when reference context is retrieved.

## Run outputs

Each run writes:

- `timing.json`: per-pass and total timing.
- `report.json`: run metadata, pass status, gap/claim counters, schema failures, and per-pass details (including fallback/skip reasons).
- `logs/run.log`: run logs.
- `final/final_answer.json` and `.md`.
- `final/plan.json` and `final/plan.md` (generated corrected execution plan).



### Pass status semantics (`report.json`)

`report.json.per_pass_status` and `report.json.per_pass_details` use explicit state values:

- `completed`: pass ran successfully in this run.
- `resumed`: pass output was loaded from an existing run directory during `--resume`.
- `skipped`: pass intentionally skipped (for example passes 05/06 in fast mode).
- `completed_with_fallback`: schema validation failed in non-strict mode and a fallback canonical artifact was written.
- `failed`: no successful/resumable output was recorded.

When fallback is used, the canonical `passes/<pass>.json` file includes metadata:
`_schema_validation_failed`, `_fallback_generated`, `_failed_output_path`, and `_validation_error`.

## Run inspector

```bash
python -m src.run_inspector --run-dir runs/<RUN_ID>
```

Prints pass status, blocking gaps, timing, and final-answer preview.

## Web Interface

Run the Streamlit interface from the repository root:

Main document uploads accept `.txt`, `.md`, `.pdf`, and `.docx` (text extraction is best-effort for PDF/DOCX based on installed optional dependencies). Reference uploads accept the same file types.

```bash
streamlit run app.py
```

The app supports demo and Claude in Streamlit Cloud deployments. Ollama remains available for local/self-hosted deployments only. The UI also supports strict mode toggling, optional web search, optional local reference documents, and renders run artifacts from the generated run directory.


## Deploy to Streamlit Cloud

1. Push your branch/repo to GitHub.
2. In Streamlit Community Cloud, create a new app and point it at this repository with `app.py` as the entry point.
3. In the app settings, add secrets from `.streamlit/secrets.toml.example` (for example `ANTHROPIC_API_KEY` and `BRAVE_API_KEY`).
4. On Streamlit Cloud, use `demo` or `claude` backend. Ollama is intentionally hidden unless you self-host the app with network access to your Ollama instance.
5. Deploy. Streamlit Cloud automatically reads `.streamlit/config.toml` for runtime settings and theme configuration.

Local development can still use environment variables; the app checks `st.secrets` first, then falls back to environment variables for API keys.




## Pipeline pass sequence

1. `00_normalize_request`
2. `01_extract_chunk`
3. `02_merge_global`
4. `03_schema_audit`
5. `04_dependency_audit`
6. `05_assumption_audit`
7. `06_evidence_audit`
8. `07_synthesize`
9. `09_generate_plan`
10. `08_validate_final`

The `09_generate_plan` pass creates a corrected, audit-informed plan with objective, materials/quantities, equipment, prerequisites, ordered steps (with `original`/`added`/`reordered` status), time estimates, warnings, quality checkpoints, blockers, assumptions, cost indicators, and contingencies.
