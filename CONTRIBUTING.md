# Contributing

## Development setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
make setup
source .venv/bin/activate
```

## Running tests

```bash
make test
```

## Linting

```bash
make lint
```

## Adding a new pass

1. Add a prompt file under `prompts/`.
2. Add a JSON schema under `schemas/`.
3. Implement pass behavior in the backend (`src/llm_interface.py` for demo logic or backend-specific implementation).
4. Wire the pass into `src/pipeline.py` in execution order.
5. Add/update validations in `src/validators.py` as needed.

## Adding a new backend

1. Implement `LocalLLMBackend` from `src/llm_interface.py`.
2. Add backend configuration to `PipelineConfig` in `src/config.py`.
3. Add backend construction logic in `AuditablePipeline` (`src/pipeline.py`).
4. Expose backend option in CLI (`src/cli.py`).
5. Validate behavior with `make test` and a dry run (`python -m src.cli ... --dry-run`).

## Commit message format

Use imperative, scoped commit messages:

```text
<type>: <short summary>
```

Recommended types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

Example:

```text
feat: add dry-run and logging controls to cli
```
