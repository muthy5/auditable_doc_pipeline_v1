# Contributing

## Dev environment setup

```bash
make setup
```

## Running tests

```bash
make test
python -m compileall src tests
```

## Adding a new pass

1. Add prompt and schema files.
2. Add the pass to `AuditablePipeline.PASS_SEQUENCE`.
3. Add execution wiring in `src/pipeline.py`.
4. Add tests for pass behavior and artifacts.

## Adding a new backend

1. Implement `LocalLLMBackend` in `src/llm_interface.py` or new module.
2. Add backend config fields.
3. Wire backend creation in `src/pipeline.py` and CLI args in `src/cli.py`.
4. Add backend-specific tests.

## Commit message format

Use Conventional Commits, e.g.:

- `feat: add run inspector summary output`
- `fix: handle missing schema files`
