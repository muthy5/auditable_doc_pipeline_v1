from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .claude_backend import ClaudeAPIBackend, ClaudeBackendConfig
from .chunker import chunk_document
from .config import PipelineConfig, RepoPaths
from .exceptions import PipelineError
from .document_classifier import DEFAULT_DOCUMENT_TYPE, SUPPORTED_DOCUMENT_TYPES, classify_document_with_metadata
from .llm_interface import RuleBasedDemoBackend
from .markdown_writer import render_final_answer_markdown, render_plan_markdown
from .openai_backend import OpenAIBackendConfig, OpenAICompatibleBackend
from .prompts import load_prompt
from .search import BraveSearchClient
from .merge_engine import merge_chunk_extractions
from .ollama_backend import OllamaBackendConfig, OllamaLocalBackend
from .pass_runner import PassRunner
from .report import write_run_report
from .text_extractor import extract_text_from_path
from .retriever import LocalFileRetriever
from .schemas import load_schema
from .validators import validate_chunks, validate_final_output

LOGGER = logging.getLogger(__name__)


def utc_run_id() -> str:
    """Return a UTC timestamp run identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


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
        ("09_generate_plan", "09_generate_plan.txt", "09_generate_plan.schema.json"),
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
        elif backend_name == "claude":
            claude_config = ClaudeBackendConfig(
                api_key=self.config.claude_api_key,
                model=self.config.claude_model,
            )
            self.backend = ClaudeAPIBackend(config=claude_config)
        elif backend_name == "openai":
            openai_config = OpenAIBackendConfig(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                base_url=self.config.openai_base_url,
            )
            self.backend = OpenAICompatibleBackend(config=openai_config)
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
        if not (repo_paths.prompts_dir / "search_queries.txt").exists():
            missing.append(str(repo_paths.prompts_dir / "search_queries.txt"))
        if not (repo_paths.schemas_dir / "search_queries.schema.json").exists():
            missing.append(str(repo_paths.schemas_dir / "search_queries.schema.json"))
        if not (repo_paths.prompts_dir / "classify_document.txt").exists():
            missing.append(str(repo_paths.prompts_dir / "classify_document.txt"))
        if not (repo_paths.schemas_dir / "classify_document.schema.json").exists():
            missing.append(str(repo_paths.schemas_dir / "classify_document.schema.json"))
        for document_type in SUPPORTED_DOCUMENT_TYPES:
            template_path = repo_paths.schemas_dir / "document_types" / f"{document_type}.json"
            if not template_path.exists():
                missing.append(str(template_path))
        if missing:
            raise PipelineError("Missing required files:\n" + "\n".join(missing))

    def build_execution_plan(self, backend: str) -> list[dict[str, Any]]:
        """Return pass execution plan metadata."""
        return [{"pass": name, "order": i + 1, "backend": backend} for i, (name, _, _) in enumerate(self.PASS_SEQUENCE)]

    def _generate_search_queries(self, normalize: dict[str, Any], document_text: str) -> list[str]:
        """Generate search queries from normalized task + source document."""
        payload = {"task": normalize, "document_text": document_text}
        query_output = self.backend.generate_json(
            pass_name="search_queries",
            prompt_text=load_prompt(self.repo_paths.prompts_dir, "search_queries.txt"),
            payload=payload,
            schema=load_schema(self.repo_paths.schemas_dir, "search_queries.schema.json"),
        )
        self.pass_runner.validate_with_schema("search_queries.schema.json", query_output)
        queries = query_output.get("queries", [])
        return [q for q in queries if isinstance(q, str) and q.strip()]

    def _build_web_context(self, normalize: dict[str, Any], document_text: str, strict: bool = False) -> list[dict[str, Any]]:
        """Generate queries and execute Brave searches for each query."""
        if not self.config.enable_search:
            return []
        if not self.config.brave_api_key:
            raise PipelineError(
                "Web search is enabled but no Brave API key was provided. Set --brave-api-key or BRAVE_API_KEY environment variable."
            )
        try:
            client = BraveSearchClient(api_key=self.config.brave_api_key)
            queries = self._generate_search_queries(normalize, document_text)
            web_context: list[dict[str, Any]] = []
            for query in queries:
                web_context.append({"query": query, "results": client.search(query)})
            return web_context
        except Exception as exc:  # noqa: BLE001 - strict mode controls whether search failures are fatal
            if strict:
                raise
            LOGGER.warning("Web search enrichment skipped due to error: %s", exc)
            return []


    def _generate_retrieval_queries(self, normalize: dict[str, Any], user_goal: str) -> list[str]:
        """Create retrieval queries from normalized task details."""
        task = normalize.get("task", {}) if isinstance(normalize, dict) else {}
        queries: list[str] = []
        for key in ["primary_goal", "deliverable_type", "domain", "audience", "jurisdiction", "timeframe"]:
            value = task.get(key) if isinstance(task, dict) else None
            if isinstance(value, str) and value.strip():
                queries.append(value.strip())
        for question in normalize.get("questions_to_answer", []):
            if isinstance(question, str) and question.strip():
                queries.append(question.strip())
        if user_goal.strip():
            queries.append(user_goal.strip())

        unique_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        return unique_queries[:8]

    def _build_reference_context(self, normalize: dict[str, Any], user_goal: str) -> list[dict[str, Any]]:
        """Retrieve local context chunks when a reference directory is configured."""
        if not self.config.reference_dir:
            return []
        retriever = LocalFileRetriever(
            self.config.reference_dir,
            chunk_target_min_words=self.config.chunk_target_min_words,
            chunk_target_max_words=self.config.chunk_target_max_words,
            chunk_hard_max_words=self.config.chunk_hard_max_words,
            chunk_overlap_max_words=self.config.chunk_overlap_max_words,
        )
        aggregated: list[dict[str, Any]] = []
        for query in self._generate_retrieval_queries(normalize, user_goal):
            for item in retriever.retrieve(query, top_k=5):
                aggregated.append({"query": query, **item})

        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in aggregated:
            key = (item["source_file"], item["text"])
            prev = deduped.get(key)
            if prev is None or item["similarity_score"] > prev["similarity_score"]:
                deduped[key] = item
        return sorted(deduped.values(), key=lambda x: x["similarity_score"], reverse=True)[:20]

    def _resolve_chunk_settings(self, fast: bool = False) -> tuple[int, int, int]:
        """Return backend-aware chunk sizing settings."""
        if self.backend_name in ("claude", "openai"):
            if fast:
                return (3500, 5500, 7000)
            return (3000, 5000, 6000)
        if fast:
            return (
                max(self.config.chunk_target_min_words, 2000),
                max(self.config.chunk_target_max_words, 3000),
                max(self.config.chunk_hard_max_words, 4000),
            )
        return (
            self.config.chunk_target_min_words,
            self.config.chunk_target_max_words,
            self.config.chunk_hard_max_words,
        )

    def _default_parallel_chunks(self) -> int:
        """Return backend-aware default for chunk parallelism."""
        return 4 if self.backend_name in ("claude", "openai") else 1

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
        document_type: str = "auto",
        parallel_chunks: int | None = None,
        fast: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Execute the full pipeline and return the run directory."""
        if "fast_mode" in kwargs:
            fast_mode_value = kwargs.pop("fast_mode")
            # Reject non-bool types: bool("false") is truthy and would silently skip audit passes 05 and 06.
            if not isinstance(fast_mode_value, bool):
                raise TypeError(
                    f"fast_mode must be a bool, got {type(fast_mode_value).__name__}: {fast_mode_value!r}"
                )
            fast = fast_mode_value
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"run() got unexpected keyword argument(s): {unexpected}")

        self.pass_runner.validation_failures.clear()
        self.pass_runner.pass_outcomes.clear()
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
            extraction_result = extract_text_from_path(input_path)
            if not extraction_result.ok:
                raise PipelineError(
                    f"Unable to extract text from input file ({extraction_result.error_code}): {extraction_result.error_message}"
                )
            text = extraction_result.text.strip()
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

            chunk_target_min_words, chunk_target_max_words, chunk_hard_max_words = self._resolve_chunk_settings(fast=fast)
            chunks = chunk_document(
                document,
                chunk_target_min_words,
                chunk_target_max_words,
                chunk_hard_max_words,
                self.config.chunk_overlap_max_words,
            )
            if errors := validate_chunks(chunks):
                raise PipelineError(f"Chunk validation failed: {errors}")
            total_words = len(text.split())
            avg_words_per_chunk = (sum(len(chunk.get("text", "").split()) for chunk in chunks) / len(chunks)) if chunks else 0.0
            LOGGER.info(
                "Chunking stats: chunks=%d avg_words_per_chunk=%.1f total_words=%d",
                len(chunks),
                avg_words_per_chunk,
                total_words,
            )
            (input_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

            def has_output(pass_name: str) -> bool:
                path_map = {
                    "00_normalize_request": passes_dir / "00_normalize_request.json",
                    "01_extract_chunk": passes_dir / "01_extract_chunk",
                    "02_merge_global": passes_dir / "02_merge_global.json",
                    "classify_document": passes_dir / "classify_document.json",
                    "03_schema_audit": passes_dir / "03_schema_audit.json",
                    "04_dependency_audit": passes_dir / "04_dependency_audit.json",
                    "05_assumption_audit": passes_dir / "05_assumption_audit.json",
                    "06_evidence_audit": passes_dir / "06_evidence_audit.json",
                    "07_synthesize": passes_dir / "07_synthesize.json",
                    "09_generate_plan": passes_dir / "09_generate_plan.json",
                    "08_validate_final": passes_dir / "08_validate_final.json",
                }
                target = path_map[pass_name]
                if target.is_dir():
                    return any(target.glob("*.json"))
                return target.exists()

            def _status_from_payload(payload: dict[str, Any], resumed: bool = False) -> str:
                if payload.get("_fallback_generated") or payload.get("_schema_validation_failed"):
                    return "completed_with_fallback"
                return "resumed" if resumed else "completed"

            def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
                details: dict[str, Any] = {}
                if payload.get("_fallback_generated"):
                    details["detail"] = "Schema validation failed; fallback payload used."
                    for key in ["_schema_validation_failed", "_fallback_generated", "_failed_output_path", "_validation_error"]:
                        if key in payload:
                            details[key] = payload[key]
                return details

            normalize_input = {
                "doc_manifest": {"doc_id": doc_id, "title": title},
                "user_goal": user_goal,
                "requested_deliverable": requested_deliverable,
                "user_constraints": ["Do not invent facts", "Surface uncertainty explicitly"],
            }

            if resume and has_output("00_normalize_request"):
                normalize = json.loads((passes_dir / "00_normalize_request.json").read_text())
                self.pass_runner.mark_pass_status("00_normalize_request", _status_from_payload(normalize, resumed=True), **_metadata_from_payload(normalize))
            else:
                LOGGER.info("Starting pass 00_normalize_request")
                normalize = self.pass_runner.run_model_pass("00_normalize_request", "00_normalize_request.txt", "00_normalize_request.schema.json", normalize_input, passes_dir / "00_normalize_request.json", strict)

            classification_output_path = passes_dir / "classify_document.json"
            if document_type == "auto":
                if resume and has_output("classify_document"):
                    classification = json.loads(classification_output_path.read_text())
                    selected_document_type = classification.get("selected_document_type", classification.get("document_type", DEFAULT_DOCUMENT_TYPE))
                else:
                    LOGGER.info("Starting pass classify_document")
                    classification = classify_document_with_metadata(text, self.backend, repo_root=self.repo_paths.root)
                    self.pass_runner.validate_with_schema("classify_document.schema.json", {
                        "document_type": classification["document_type"],
                        "confidence": classification["confidence"],
                        "reason": classification["reason"],
                    })
                    selected_document_type = str(classification.get("selected_document_type", DEFAULT_DOCUMENT_TYPE))
                    classification_output_path.write_text(json.dumps(classification, indent=2), encoding="utf-8")
            else:
                if document_type not in SUPPORTED_DOCUMENT_TYPES:
                    raise PipelineError(f"Unsupported document type '{document_type}'. Supported: {sorted(SUPPORTED_DOCUMENT_TYPES)}")
                selected_document_type = document_type
                classification = {
                    "document_type": document_type,
                    "confidence": "high",
                    "reason": "User-specified document type.",
                    "selected_document_type": document_type,
                }
                if not (resume and classification_output_path.exists()):
                    classification_output_path.write_text(json.dumps(classification, indent=2), encoding="utf-8")

            document_type_schema = load_schema(
                self.repo_paths.schemas_dir / "document_types",
                f"{selected_document_type}.json",
            )

            web_context: list[dict[str, Any]] = []
            web_context_output = passes_dir / "search_web_context.json"
            if resume and web_context_output.exists():
                web_context = json.loads(web_context_output.read_text(encoding="utf-8")).get("web_context", [])
            else:
                web_context = self._build_web_context(normalize=normalize, document_text=text, strict=strict)
                if web_context:
                    web_context_output.write_text(json.dumps({"web_context": web_context}, indent=2, ensure_ascii=False), encoding="utf-8")

            reference_context: list[dict[str, Any]] = []
            retrieval_context_output = passes_dir / "retrieval_context.json"
            if resume and retrieval_context_output.exists():
                reference_context = json.loads(retrieval_context_output.read_text(encoding="utf-8")).get("reference_context", [])
            else:
                reference_context = self._build_reference_context(normalize=normalize, user_goal=user_goal)
                if reference_context:
                    retrieval_context_output.write_text(json.dumps({"reference_context": reference_context}, indent=2, ensure_ascii=False), encoding="utf-8")

            extraction_dir = passes_dir / "01_extract_chunk"
            extraction_dir.mkdir(parents=True, exist_ok=True)
            total_chunks = len(chunks)
            chunk_extractions: list[dict[str, Any] | None] = [None] * total_chunks
            effective_parallel_chunks = max(1, parallel_chunks if parallel_chunks is not None else self._default_parallel_chunks())
            if fast and parallel_chunks is None:
                effective_parallel_chunks = max(effective_parallel_chunks, self._default_parallel_chunks())

            def _process_chunk(index: int, chunk: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                out = extraction_dir / f"{chunk['chunk_id']}.json"
                LOGGER.info("Processing chunk %d of %d (%s)", index + 1, total_chunks, chunk["chunk_id"])
                if progress_callback:
                    progress_callback(index + 1, total_chunks)
                if resume and out.exists():
                    return index, json.loads(out.read_text())
                extraction = self.pass_runner.run_model_pass(
                    "01_extract_chunk",
                    "01_extract_chunk.txt",
                    "01_extract_chunk.schema.json",
                    {"task": normalize, "chunk": chunk, "web_context": web_context, "reference_context": reference_context},
                    out,
                    strict,
                )
                return index, extraction

            if effective_parallel_chunks == 1:
                for idx, chunk in enumerate(chunks):
                    chunk_index, extraction = _process_chunk(idx, chunk)
                    chunk_extractions[chunk_index] = extraction
            else:
                with ThreadPoolExecutor(max_workers=effective_parallel_chunks) as executor:
                    futures = [executor.submit(_process_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
                    for future in as_completed(futures):
                        chunk_index, extraction = future.result()
                        chunk_extractions[chunk_index] = extraction

            ordered_chunk_extractions: list[dict[str, Any]] = [item for item in chunk_extractions if item is not None]
            resumed_chunks = 0
            fallback_chunks = 0
            for idx, chunk in enumerate(chunks):
                out = extraction_dir / f"{chunk['chunk_id']}.json"
                if resume and out.exists():
                    resumed_chunks += 1
                payload = ordered_chunk_extractions[idx] if idx < len(ordered_chunk_extractions) else {}
                if isinstance(payload, dict) and (payload.get("_fallback_generated") or payload.get("_schema_validation_failed")):
                    fallback_chunks += 1
            if fallback_chunks > 0:
                self.pass_runner.mark_pass_status("01_extract_chunk", "completed_with_fallback", detail=f"{fallback_chunks} chunk extraction(s) used fallback payloads")
            elif resumed_chunks == len(chunks) and len(chunks) > 0:
                self.pass_runner.mark_pass_status("01_extract_chunk", "resumed", detail="All chunk extractions loaded from existing artifacts")
            else:
                self.pass_runner.mark_pass_status("01_extract_chunk", "completed")

            merge = merge_chunk_extractions(doc_id, ordered_chunk_extractions)
            if resume and has_output("02_merge_global"):
                merge = json.loads((passes_dir / "02_merge_global.json").read_text())
                self.pass_runner.mark_pass_status("02_merge_global", _status_from_payload(merge, resumed=True), **_metadata_from_payload(merge))
            else:
                LOGGER.info("Starting pass 02_merge_global")
                merge = self.pass_runner.write_validated_json("02_merge_global.schema.json", merge, passes_dir / "02_merge_global.json", "02_merge_global", strict)

            chunk_summaries = [{"chunk_id": item["chunk_id"], "section_role": item["section_role"]} for item in ordered_chunk_extractions]

            def run_or_load(pass_name: str, prompt: str, schema: str, payload: dict[str, Any]) -> dict[str, Any]:
                output_path = passes_dir / f"{pass_name}.json"
                if resume and output_path.exists():
                    existing = json.loads(output_path.read_text())
                    self.pass_runner.mark_pass_status(pass_name, _status_from_payload(existing, resumed=True), **_metadata_from_payload(existing))
                    return existing
                LOGGER.info("Starting pass %s", pass_name)
                return self.pass_runner.run_model_pass(pass_name, prompt, schema, payload, output_path, strict)

            schema_audit = run_or_load("03_schema_audit", "03_schema_audit.txt", "03_schema_audit.schema.json", {"task": normalize, "merge": merge, "chunk_summaries": chunk_summaries, "document_type": selected_document_type, "document_type_schema": document_type_schema, "web_context": web_context, "reference_context": reference_context})
            dependency_audit = run_or_load("04_dependency_audit", "04_dependency_audit.txt", "04_dependency_audit.schema.json", {"task": normalize, "merge": merge, "schema_audit": schema_audit, "web_context": web_context, "reference_context": reference_context})
            if fast:
                LOGGER.info("Fast mode enabled: skipping passes 05_assumption_audit and 06_evidence_audit")
                assumption_audit = {"implicit_assumptions_found": [], "uncertainty_points": [], "blocking_assumptions": []}
                evidence_audit = {"claim_registry": []}
                self.pass_runner.mark_pass_status("05_assumption_audit", "skipped", detail="Skipped in fast mode")
                self.pass_runner.mark_pass_status("06_evidence_audit", "skipped", detail="Skipped in fast mode")
            else:
                assumption_audit = run_or_load("05_assumption_audit", "05_assumption_audit.txt", "05_assumption_audit.schema.json", {"task": normalize, "merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit, "web_context": web_context, "reference_context": reference_context})
                evidence_audit = run_or_load("06_evidence_audit", "06_evidence_audit.txt", "06_evidence_audit.schema.json", {"merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit, "assumption_audit": assumption_audit, "web_context": web_context, "reference_context": reference_context})
            synthesis = run_or_load("07_synthesize", "07_synthesize.txt", "07_synthesize.schema.json", {"task": normalize, "merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit, "assumption_audit": assumption_audit, "evidence_audit": evidence_audit, "web_context": web_context, "reference_context": reference_context})
            plan = run_or_load("09_generate_plan", "09_generate_plan.txt", "09_generate_plan.schema.json", {"task": normalize, "merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit, "assumption_audit": assumption_audit, "evidence_audit": evidence_audit, "synthesis": synthesis, "web_context": web_context, "reference_context": reference_context})

            validation = validate_final_output(synthesis, normalize, schema_audit, dependency_audit, assumption_audit, evidence_audit, load_schema(self.repo_paths.schemas_dir, "07_synthesize.schema.json"))
            if resume and has_output("08_validate_final"):
                existing_validation = json.loads((passes_dir / "08_validate_final.json").read_text())
                self.pass_runner.mark_pass_status("08_validate_final", _status_from_payload(existing_validation, resumed=True), **_metadata_from_payload(existing_validation))
            else:
                LOGGER.info("Starting pass 08_validate_final")
                self.pass_runner.write_validated_json("08_validate_final.schema.json", validation, passes_dir / "08_validate_final.json", "08_validate_final", strict)

            (final_dir / "final_answer.json").write_text(json.dumps(synthesis, indent=2), encoding="utf-8")
            (final_dir / "final_answer.md").write_text(render_final_answer_markdown(synthesis), encoding="utf-8")
            (final_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            (final_dir / "plan.md").write_text(render_plan_markdown(plan), encoding="utf-8")

            total_time = time.perf_counter() - start_total
            self.pass_runner.write_timings(run_dir / "timing.json", total_time)
            per_pass_status: dict[str, str] = {}
            per_pass_details: dict[str, dict[str, Any]] = {}
            for name, _, _ in self.PASS_SEQUENCE:
                outcome = self.pass_runner.pass_outcomes.get(name, {"status": "failed", "detail": "No output recorded"})
                per_pass_status[name] = str(outcome.get("status", "failed"))
                per_pass_details[name] = outcome

            write_run_report(
                run_dir,
                {
                    "run_id": run_id,
                    "backend": self.backend_name,
                    "model_name": getattr(self.backend, "config", None).model if hasattr(self.backend, "config") else "demo",
                    "input_path": str(input_path),
                    "total_duration_seconds": total_time,
                    "per_pass_status": per_pass_status,
                    "per_pass_details": per_pass_details,
                    "blocking_gap_count": len(schema_audit.get("blocking_gaps", [])),
                    "unsupported_claim_count": len([e for e in validation.get("errors", []) if e.get("code") == "E_SYNTH_UNSUPPORTED_CLAIM"]),
                    "schema_validation_failure_list": self.pass_runner.validation_failures,
                    "document_type": selected_document_type,
                },
            )
            return run_dir
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()
