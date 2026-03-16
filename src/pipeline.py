from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunker import chunk_document
from .config import PipelineConfig, RepoPaths
from .exceptions import PipelineError
from .llm_interface import RuleBasedDemoBackend
from .markdown_writer import render_final_answer_markdown
from .merge_engine import merge_chunk_extractions
from .ollama_backend import OllamaBackendConfig, OllamaLocalBackend
from .pass_runner import PassRunner
from .report import write_run_report
from .schemas import load_schema
from .validators import validate_chunks, validate_final_output

LOGGER = logging.getLogger(__name__)


def utc_run_id() -> str:
    """Return a UTC timestamp run identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class AuditablePipeline:
    """Main orchestrator for the auditable document pipeline."""

    PASS_SEQUENCE = [
        ("00_normalize_request", "00_normalize_request.txt", "00_normalize_request.schema.json"),
        ("01_extract_chunk", "01_extract_chunk.txt", "01_extract_chunk.schema.json"),
        ("02_merge_global", None, "02_merge_global.schema.json"),
        ("03_schema_audit", "03_schema_audit.txt", "03_schema_audit.schema.json"),
        ("04_dependency_audit", "04_dependency_audit.txt", "04_dependency_audit.schema.json"),
        ("05_assumption_audit", "05_assumption_audit.txt", "05_assumption_audit.schema.json"),
        ("06_evidence_audit", "06_evidence_audit.txt", "06_evidence_audit.schema.json"),
        ("07_synthesize", "07_synthesize.txt", "07_synthesize.schema.json"),
        ("08_validate_final", None, "08_validate_final.schema.json"),
    ]

    def __init__(self, repo_root: Path, backend_name: str = "demo", config: PipelineConfig | None = None) -> None:
        """Initialize pipeline dependencies and backend."""
        self.repo_paths = RepoPaths.from_root(repo_root)
        self.config = config or PipelineConfig()
        self._validate_required_files(self.repo_paths)
        self.backend_name = backend_name
        if backend_name == "demo":
            self.backend = RuleBasedDemoBackend()
        elif backend_name == "ollama":
            ollama_config = OllamaBackendConfig(
                base_url=self.config.ollama_base_url,
                model=self.config.ollama_model,
                timeout_s=self.config.ollama_timeout_s,
                temperature=self.config.ollama_temperature,
                num_predict=self.config.ollama_num_predict,
                max_retries=self.config.ollama_max_retries,
            )
            self.backend = OllamaLocalBackend(config=ollama_config)
            self.backend.health_check()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
        self.pass_runner = PassRunner(self.backend, self.repo_paths.prompts_dir, self.repo_paths.schemas_dir)

    def _validate_required_files(self, repo_paths: RepoPaths) -> None:
        """Ensure all expected prompt and schema files exist."""
        missing: list[str] = []
        for _pass_name, prompt, schema in self.PASS_SEQUENCE:
            if prompt and not (repo_paths.prompts_dir / prompt).exists():
                missing.append(str(repo_paths.prompts_dir / prompt))
            if schema and not (repo_paths.schemas_dir / schema).exists():
                missing.append(str(repo_paths.schemas_dir / schema))
        if not (repo_paths.schemas_dir / "document.schema.json").exists():
            missing.append(str(repo_paths.schemas_dir / "document.schema.json"))
        if not (repo_paths.schemas_dir / "chunk.schema.json").exists():
            missing.append(str(repo_paths.schemas_dir / "chunk.schema.json"))
        if missing:
            raise PipelineError("Missing required files:\n" + "\n".join(missing))

    def build_execution_plan(self, backend: str) -> list[dict[str, Any]]:
        """Return pass execution plan metadata."""
        return [{"pass": name, "order": i + 1, "backend": backend} for i, (name, _, _) in enumerate(self.PASS_SEQUENCE)]

    def run(
        self,
        input_path: Path,
        runs_dir: Path,
        doc_id: str = "doc_001",
        title: str | None = None,
        user_goal: str = "Identify missing information and organize the document into an actionable structure.",
        requested_deliverable: str = "structured_gap_analysis_and_plan",
        strict: bool = False,
        run_dir: Path | None = None,
        resume: bool = False,
    ) -> Path:
        """Execute the full pipeline and return the run directory."""
        start_total = time.perf_counter()
        run_id = run_dir.name if run_dir else utc_run_id()
        run_dir = run_dir or (runs_dir / run_id)
        input_dir, passes_dir, final_dir, logs_dir = run_dir / "input", run_dir / "passes", run_dir / "final", run_dir / "logs"
        for p in [input_dir, passes_dir, final_dir, logs_dir]:
            p.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        try:
            text = input_path.read_text(encoding=self.config.encoding)
            document = {
                "doc_id": doc_id,
                "title": title,
                "source_path": str(input_path),
                "content_type": "text/plain",
                "encoding": self.config.encoding,
                "text": text,
                "metadata": {"author": None, "created_at": None, "user_goal": user_goal},
            }
            self.pass_runner.validate_with_schema("document.schema.json", document)
            (input_dir / "document.json").write_text(json.dumps(document, indent=2), encoding="utf-8")

            chunks = chunk_document(document, self.config.chunk_target_min_words, self.config.chunk_target_max_words, self.config.chunk_hard_max_words, self.config.chunk_overlap_max_words)
            if errors := validate_chunks(chunks):
                raise PipelineError(f"Chunk validation failed: {errors}")
            (input_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

            def has_output(pass_name: str) -> bool:
                path_map = {
                    "00_normalize_request": passes_dir / "00_normalize_request.json",
                    "01_extract_chunk": passes_dir / "01_extract_chunk",
                    "02_merge_global": passes_dir / "02_merge_global.json",
                    "03_schema_audit": passes_dir / "03_schema_audit.json",
                    "04_dependency_audit": passes_dir / "04_dependency_audit.json",
                    "05_assumption_audit": passes_dir / "05_assumption_audit.json",
                    "06_evidence_audit": passes_dir / "06_evidence_audit.json",
                    "07_synthesize": passes_dir / "07_synthesize.json",
                    "08_validate_final": passes_dir / "08_validate_final.json",
                }
                target = path_map[pass_name]
                if target.is_dir():
                    return any(target.glob("*.json"))
                return target.exists()

            normalize_input = {
                "doc_manifest": {"doc_id": doc_id, "title": title},
                "user_goal": user_goal,
                "requested_deliverable": requested_deliverable,
                "user_constraints": ["Do not invent facts", "Surface uncertainty explicitly"],
            }

            if resume and has_output("00_normalize_request"):
                normalize = json.loads((passes_dir / "00_normalize_request.json").read_text())
            else:
                LOGGER.info("Starting pass 00_normalize_request")
                normalize = self.pass_runner.run_model_pass("00_normalize_request", "00_normalize_request.txt", "00_normalize_request.schema.json", normalize_input, passes_dir / "00_normalize_request.json", strict)

            extraction_dir = passes_dir / "01_extract_chunk"
            extraction_dir.mkdir(parents=True, exist_ok=True)
            chunk_extractions: list[dict[str, Any]] = []
            for chunk in chunks:
                out = extraction_dir / f"{chunk['chunk_id']}.json"
                if resume and out.exists():
                    extraction = json.loads(out.read_text())
                else:
                    LOGGER.info("Starting pass 01_extract_chunk for %s", chunk["chunk_id"])
                    extraction = self.pass_runner.run_model_pass("01_extract_chunk", "01_extract_chunk.txt", "01_extract_chunk.schema.json", {"task": normalize, "chunk": chunk}, out, strict)
                chunk_extractions.append(extraction)

            merge = merge_chunk_extractions(doc_id, chunk_extractions)
            if not (resume and has_output("02_merge_global")):
                LOGGER.info("Starting pass 02_merge_global")
                self.pass_runner.write_validated_json("02_merge_global.schema.json", merge, passes_dir / "02_merge_global.json", "02_merge_global", strict)

            chunk_summaries = [{"chunk_id": item["chunk_id"], "section_role": item["section_role"]} for item in chunk_extractions]

            def run_or_load(pass_name: str, prompt: str, schema: str, payload: dict[str, Any]) -> dict[str, Any]:
                output_path = passes_dir / f"{pass_name}.json"
                if resume and output_path.exists():
                    return json.loads(output_path.read_text())
                LOGGER.info("Starting pass %s", pass_name)
                return self.pass_runner.run_model_pass(pass_name, prompt, schema, payload, output_path, strict)

            schema_audit = run_or_load("03_schema_audit", "03_schema_audit.txt", "03_schema_audit.schema.json", {"task": normalize, "merge": merge, "chunk_summaries": chunk_summaries})
            dependency_audit = run_or_load("04_dependency_audit", "04_dependency_audit.txt", "04_dependency_audit.schema.json", {"task": normalize, "merge": merge, "schema_audit": schema_audit})
            assumption_audit = run_or_load("05_assumption_audit", "05_assumption_audit.txt", "05_assumption_audit.schema.json", {"task": normalize, "merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit})
            evidence_audit = run_or_load("06_evidence_audit", "06_evidence_audit.txt", "06_evidence_audit.schema.json", {"merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit, "assumption_audit": assumption_audit})
            synthesis = run_or_load("07_synthesize", "07_synthesize.txt", "07_synthesize.schema.json", {"task": normalize, "merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit, "assumption_audit": assumption_audit, "evidence_audit": evidence_audit})

            validation = validate_final_output(synthesis, normalize, schema_audit, dependency_audit, assumption_audit, evidence_audit, load_schema(self.repo_paths.schemas_dir, "07_synthesize.schema.json"))
            if not (resume and has_output("08_validate_final")):
                LOGGER.info("Starting pass 08_validate_final")
                self.pass_runner.write_validated_json("08_validate_final.schema.json", validation, passes_dir / "08_validate_final.json", "08_validate_final", strict)

            (final_dir / "final_answer.json").write_text(json.dumps(synthesis, indent=2), encoding="utf-8")
            (final_dir / "final_answer.md").write_text(render_final_answer_markdown(synthesis), encoding="utf-8")

            total_time = time.perf_counter() - start_total
            self.pass_runner.write_timings(run_dir / "timing.json", total_time)
            write_run_report(
                run_dir,
                {
                    "run_id": run_id,
                    "backend": self.backend_name,
                    "model_name": getattr(self.backend, "config", None).model if hasattr(self.backend, "config") else "demo",
                    "input_path": str(input_path),
                    "total_duration_seconds": total_time,
                    "per_pass_status": {name: "completed" for name, _, _ in self.PASS_SEQUENCE},
                    "blocking_gap_count": len(schema_audit.get("blocking_gaps", [])),
                    "unsupported_claim_count": len([e for e in validation.get("errors", []) if e.get("code") == "E_SYNTH_UNSUPPORTED_CLAIM"]),
                    "schema_validation_failure_list": self.pass_runner.validation_failures,
                },
            )
            return run_dir
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()
