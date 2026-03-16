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

## CLI features

- `--strict`: stop on first schema-validation failure.
- `--dry-run`: print pass order/backend plan only; no execution.
- `--resume`: resume existing run (`--run-dir`) from first incomplete pass.
- `--verbose`: set logging level to DEBUG.
- `--quiet`: set logging level to ERROR.

## Backends

### ollama

```bash
make run-ollama
```

### claude

Use the Claude API backend with `--backend claude`. Provide credentials via `--claude-api-key` or the `ANTHROPIC_API_KEY` environment variable. The default model is `claude-sonnet-4-20250514` (override with `--claude-model`).

## Run outputs

Each run writes:

- `timing.json`: per-pass and total timing.
- `report.json`: run metadata, pass status, gap/claim counters, schema failures.
- `logs/run.log`: run logs.
- `final/final_answer.json` and `.md`.
- `final/plan.json` and `final/plan.md` (generated corrected execution plan).

## Run inspector

```bash
python -m src.run_inspector --run-dir runs/<RUN_ID>
```

Prints pass status, blocking gaps, timing, and final-answer preview.

## Web Interface

Run the Streamlit interface from the repository root:

```bash
streamlit run app.py
```

The app supports the demo, Ollama, and Claude backends, allows strict mode toggling, and renders run artifacts from the generated run directory.



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
