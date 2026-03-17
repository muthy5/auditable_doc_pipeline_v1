from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import AuditablePipeline


class _StopAfterDependency(RuntimeError):
    pass


class _FakePassRunner:
    def __init__(self, calls: list[tuple[str, dict]], submit_trace: list[tuple[str, str]]) -> None:
        self.calls = calls
        self.submit_trace = submit_trace
        self.validation_failures: list[dict] = []
        self.pass_outcomes: dict[str, dict] = {}

    def validate_with_schema(self, _schema_name: str, _payload: dict) -> None:
        return None

    def write_validated_json(self, _schema: str, payload: dict, _output_path: Path, _pass_name: str, _strict: bool) -> dict:
        return payload

    def mark_pass_status(self, _pass_name: str, _status: str, **_kwargs: object) -> None:
        return None

    def write_timings(self, _path: Path, _seconds: float) -> None:
        return None

    def run_model_pass(
        self,
        pass_name: str,
        _prompt: str,
        _schema: str,
        payload: dict,
        _output_path: Path,
        _strict: bool,
    ) -> dict:
        self.calls.append((pass_name, payload))
        if pass_name == "00_normalize_request":
            return {"task": {"primary_goal": "x"}, "questions_to_answer": []}
        if pass_name == "01_extract_chunk":
            return {
                "chunk_id": "c1",
                "section_role": "body",
                "entities": [],
                "facts": [],
                "events": [],
                "requirements": [],
                "metrics": [],
                "risks": [],
                "open_questions": [],
                "evidence_quotes": [],
            }
        if pass_name == "03_schema_audit":
            return {"expected_sections": [], "missing_sections": [], "inconsistencies": [], "blocking_gaps": []}
        if pass_name == "04_dependency_audit":
            return {"missing_prerequisites": [], "ordering_issues": [], "dependency_graph": []}
        if pass_name == "05_assumption_audit":
            raise _StopAfterDependency("stop")
        return {}


class _FakeFuture:
    def __init__(self, name: str, value: dict, trace: list[tuple[str, str]]) -> None:
        self._name = name
        self._value = value
        self._trace = trace

    def result(self) -> dict:
        self._trace.append(("result", self._name))
        return self._value


class _FakeExecutor:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._trace = _kwargs.pop("trace")

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def submit(self, fn, pass_name: str, *args):
        self._trace.append(("submit", pass_name))
        value = fn(pass_name, *args)
        return _FakeFuture(pass_name, value, self._trace)


def _build_pipeline(monkeypatch: pytest.MonkeyPatch, backend_name: str) -> tuple[AuditablePipeline, list[tuple[str, dict]], list[tuple[str, str]]]:
    repo_root = Path(__file__).resolve().parents[1]

    if backend_name == "openai":
        class _DummyOpenAIBackend:
            def __init__(self, config):
                self.config = config

        monkeypatch.setattr("src.pipeline.OpenAICompatibleBackend", _DummyOpenAIBackend)
    if backend_name == "ollama":
        class _DummyOllamaBackend:
            def __init__(self, config):
                self.config = config

            def health_check(self) -> None:
                return None

        monkeypatch.setattr("src.pipeline.OllamaLocalBackend", _DummyOllamaBackend)

    pipeline = AuditablePipeline(repo_root=repo_root, backend_name=backend_name)
    calls: list[tuple[str, dict]] = []
    trace: list[tuple[str, str]] = []
    pipeline.pass_runner = _FakePassRunner(calls, trace)

    monkeypatch.setattr(
        "src.pipeline.extract_text_from_path",
        lambda _path: type("Extract", (), {"ok": True, "text": "alpha beta gamma", "error_code": "", "error_message": ""})(),
    )
    monkeypatch.setattr("src.pipeline.chunk_document", lambda *_args, **_kwargs: [{"chunk_id": "c1", "text": "alpha beta", "span": [0, 2]}])
    monkeypatch.setattr("src.pipeline.validate_chunks", lambda _chunks: [])
    monkeypatch.setattr(
        "src.pipeline.merge_chunk_extractions",
        lambda _doc_id, _items: {"sections": [], "entities": [], "facts": [], "open_questions": []},
    )
    monkeypatch.setattr("src.pipeline.classify_document_with_metadata", lambda *_args, **_kwargs: {"document_type": "procedural_plan", "confidence": 0.9, "reason": "test"})
    monkeypatch.setattr("src.pipeline.load_schema", lambda *_args, **_kwargs: {})

    return pipeline, calls, trace


def test_passes_03_04_run_concurrently(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline, calls, trace = _build_pipeline(monkeypatch, backend_name="openai")

    monkeypatch.setattr("src.pipeline.ThreadPoolExecutor", lambda *args, **kwargs: _FakeExecutor(*args, trace=trace, **kwargs))

    with pytest.raises(_StopAfterDependency):
        pipeline.run(
            input_path=Path(__file__).resolve().parents[1] / "examples" / "lemonade_plan_missing_juicing.txt",
            runs_dir=tmp_path / "runs",
            parallel_chunks=1,
        )

    assert trace[:4] == [
        ("submit", "03_schema_audit"),
        ("submit", "04_dependency_audit"),
        ("result", "03_schema_audit"),
        ("result", "04_dependency_audit"),
    ]
    dependency_payload = [payload for pass_name, payload in calls if pass_name == "04_dependency_audit"][0]
    assert "schema_audit" not in dependency_payload


def test_passes_03_04_sequential_on_ollama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline, calls, _trace = _build_pipeline(monkeypatch, backend_name="ollama")

    with pytest.raises(_StopAfterDependency):
        pipeline.run(
            input_path=Path(__file__).resolve().parents[1] / "examples" / "lemonade_plan_missing_juicing.txt",
            runs_dir=tmp_path / "runs",
            parallel_chunks=1,
        )

    pass_names = [name for name, _payload in calls]
    assert pass_names.index("03_schema_audit") < pass_names.index("04_dependency_audit")
    dependency_payload = [payload for pass_name, payload in calls if pass_name == "04_dependency_audit"][0]
    assert "schema_audit" not in dependency_payload
