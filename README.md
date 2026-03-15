# auditable_doc_pipeline_v1

A concrete v1 repository skeleton for an auditable, chunked document-analysis pipeline.

The pipeline is designed for long documents (for example, ~9,000 words) where you want to:
- surface missing information
- expose hidden assumptions
- detect dependency gaps
- organize the material into a structured final answer
- prevent unsupported claims from appearing in the output

## What this repo includes

- file layout for the pipeline
- Python skeleton with a runnable CLI
- pass runner
- JSON schemas
- mechanical validation logic
- prompt templates
- a rule-based demo backend
- an example document (`examples/lemonade_plan_missing_juicing.txt`)

## Important scope note

This is a **v1 implementation skeleton**, not a production-grade reasoning engine.

It is designed to:
1. make the architecture concrete
2. make every pass auditable
3. give you a local control layer you can later connect to a real local model

The included `RuleBasedDemoBackend` is intentionally simple. It can demonstrate the pipeline on basic procedural documents and can catch obvious omissions such as a lemonade plan that never includes a juicing step.

## Repository layout

```text
auditable_doc_pipeline_v1/
  README.md
  requirements.txt
  .gitignore
  examples/
    lemonade_plan_missing_juicing.txt
  prompts/
    00_normalize_request.txt
    01_extract_chunk.txt
    03_schema_audit.txt
    04_dependency_audit.txt
    05_assumption_audit.txt
    06_evidence_audit.txt
    07_synthesize.txt
  schemas/
    document.schema.json
    chunk.schema.json
    00_normalize_request.schema.json
    01_extract_chunk.schema.json
    02_merge_global.schema.json
    03_schema_audit.schema.json
    04_dependency_audit.schema.json
    05_assumption_audit.schema.json
    06_evidence_audit.schema.json
    07_synthesize.schema.json
    08_validate_final.schema.json
  runs/
  src/
    __init__.py
    cli.py
    config.py
    chunker.py
    llm_interface.py
    merge_engine.py
    pass_runner.py
    pipeline.py
    prompts.py
    schemas.py
    validators.py
    markdown_writer.py
```

## Quick start

Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the example:

```bash
python -m src.cli   --input examples/lemonade_plan_missing_juicing.txt   --runs-dir runs   --backend demo
```

To use Ollama instead, set `--backend ollama` and provide `--ollama-model`.
Optional flags: `--ollama-base-url`, `--ollama-timeout-s`, `--ollama-temperature`,
`--ollama-num-predict`, and `--ollama-max-retries`.

The command will create a timestamped run directory under `runs/` and write:

```text
runs/<RUN_ID>/
  input/
    document.json
    chunks.json
  passes/
    00_normalize_request.json
    01_extract_chunk/
      chunk_0001.json
      ...
    02_merge_global.json
    03_schema_audit.json
    04_dependency_audit.json
    05_assumption_audit.json
    06_evidence_audit.json
    07_synthesize.json
    08_validate_final.json
  final/
    final_answer.json
    final_answer.md
  logs/
    run.log
```

## Backends

### `demo`
A rule-based local demo backend. Useful for testing the controller architecture and basic omission detection.

### `ollama`
A real local model backend via an Ollama server. This backend preserves the same pass sequence,
schema validation, and run artifact layout as the demo backend.

Example:

```bash
python -m src.cli \
  --input examples/lemonade_plan_missing_juicing.txt \
  --runs-dir runs \
  --backend ollama \
  --ollama-model llama3.1:8b-instruct-q4_K_M \
  --ollama-base-url http://127.0.0.1:11434
```

### Future backends
Replace the demo backend with a real local model backend that implements the interface in `src/llm_interface.py`.

## Design invariants

- Every pass returns JSON only.
- Every pass output is validated against a JSON schema.
- Final synthesis may only use upstream artifacts.
- Every substantive output sentence carries support IDs.
- Blocking gaps must be surfaced explicitly.
- Unsupported claims fail validation.

## Example behavior

The example lemonade document intentionally omits the step where lemons are juiced.

The demo pipeline should flag that omission in:
- the schema audit
- the dependency audit
- the final answer

## Limitations

- Entity resolution is exact-string only in v1.
- Contradiction detection is limited.
- The demo backend uses heuristics, not deep reasoning.
- Some document types will require domain-specific schema templates and a stronger backend.

## Next upgrade path

1. Replace `RuleBasedDemoBackend` with a real local LLM backend.
2. Add retrieval over local files.
3. Add domain-specific schema libraries.
4. Add stronger contradiction checking.
5. Add selective re-generation on validation failure.
