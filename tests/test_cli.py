from __future__ import annotations

from src.pipeline import AuditablePipeline


def test_describe_execution_plan_contains_backend() -> None:
    pipeline = AuditablePipeline(repo_root=__import__('pathlib').Path(__file__).resolve().parents[1], backend_name='demo')
    plan = pipeline.describe_execution_plan(backend_name='demo')
    assert plan[0] == 'backend=demo'
    assert any('07_synthesize' in item for item in plan)
