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

## Ollama usage

```bash
make run-ollama
```

## Run outputs

Each run writes:

- `timing.json`: per-pass and total timing.
- `report.json`: run metadata, pass status, gap/claim counters, schema failures.
- `logs/run.log`: run logs.
- `final/final_answer.json` and `.md`.

## Run inspector

```bash
python -m src.run_inspector --run-dir runs/<RUN_ID>
```

Prints pass status, blocking gaps, timing, and final-answer preview.
