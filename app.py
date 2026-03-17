from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

import app_utils
from src.config import PipelineConfig
from src.document_classifier import SUPPORTED_DOCUMENT_TYPES
from src.pipeline import AuditablePipeline
from src.preflight import CapabilityStatus, run_preflight
from src.text_extractor import TextExtractionResult, extract_text_from_path

PASS_SEQUENCE = app_utils.PASS_SEQUENCE
collect_pass_outputs = app_utils.collect_pass_outputs
collect_run_report = app_utils.collect_run_report
format_gap_plain_english = app_utils.format_gap_plain_english
format_plan_for_display = app_utils.format_plan_for_display
get_available_backends = app_utils.get_available_backends
get_status_color = app_utils.get_status_color
is_streamlit_cloud_environment = app_utils.is_streamlit_cloud_environment
parse_final_sections = app_utils.parse_final_sections
parse_plan_output = app_utils.parse_plan_output
read_final_markdown = app_utils.read_final_markdown


def _build_plan_request_document_fallback(plan_request: str) -> str:
    request = plan_request.strip()
    return "\n".join(
        [
            "Title: Requested Plan",
            f"Goal: {request}",
            "Materials:",
            "- TBD resources to be identified during planning",
            "Plan:",
            "1. Clarify the desired outcome and constraints.",
            "2. Gather required materials, tools, and dependencies.",
            "3. Execute the work in a safe, ordered sequence.",
            f"Expected output: Completed objective for '{request}'.",
        ]
    )


build_plan_request_document = getattr(app_utils, "build_plan_request_document", _build_plan_request_document_fallback)

st.set_page_config(page_title="Auditable Document Pipeline", page_icon="📄", layout="wide")
st.title("Auditable Document Pipeline")

def _secret_or_env(name: str) -> str:
    try:
        value = st.secrets.get(name, None)
        if value is not None:
            return str(value)
    except (StreamlitSecretNotFoundError, TypeError, KeyError, FileNotFoundError, Exception):  # noqa: BLE001
        pass
    return os.environ.get(name, "")








def _format_extraction_error(result: TextExtractionResult) -> str:
    if result.error_code == "missing_pdf_parser":
        return "PDF parser package missing: install optional dependency 'pypdf'."
    if result.error_code == "missing_docx_parser":
        return "DOCX parser package missing: install optional dependency 'python-docx'."
    if result.error_code == "unsupported_file_type":
        return "Unsupported file type. Please upload .txt, .md, .pdf, or .docx."
    if result.error_code == "corrupted_document":
        return f"Document corrupted or unreadable: {result.error_message}"
    if result.error_code == "image_only_pdf":
        return "PDF appears scanned/image-only and has no extractable text (OCR not configured)."
    if result.error_code == "empty_document":
        return "File contains no extractable text."
    return result.error_message or "Failed to parse uploaded document."


def _render_capability_status(statuses: dict[str, CapabilityStatus]) -> None:
    st.subheader("System Status")
    labels = {
        "demo_backend": "Demo backend",
        "claude_backend": "Claude backend",
        "ollama_backend": "Ollama backend",
        "openai_backend": "OpenAI-compatible backend",
        "pdf_parsing": "PDF parsing",
        "docx_parsing": "DOCX parsing",
        "web_search": "Web search",
    }
    for key, label in labels.items():
        status = statuses[key]
        prefix = "✅" if status.available else "❌"
        st.caption(f"{prefix} {label}: {status.message}")

def _render_status_banner(status_color: str) -> None:
    if status_color == "red":
        message = "🚨 This document has critical gaps"
        style = "background:#fee2e2;border-left:8px solid #dc2626;color:#7f1d1d;"
    elif status_color == "yellow":
        message = "⚠️ This document has notable gaps or uncertainty"
        style = "background:#fef3c7;border-left:8px solid #d97706;color:#7c2d12;"
    else:
        message = "✅ This document appears operationally complete"
        style = "background:#dcfce7;border-left:8px solid #16a34a;color:#14532d;"

    st.markdown(
        f"<div style='padding:1rem;border-radius:0.5rem;font-size:1.2rem;font-weight:700;{style}'>{message}</div>",
        unsafe_allow_html=True,
    )


def _render_gaps(sections: dict[str, Any], plan_display: dict[str, Any]) -> None:
    gap_candidates: list[dict[str, Any]] = [
        *sections.get("missing_information", []),
        *sections.get("dependencies", []),
    ]
    for text in plan_display.get("blocking_items", []):
        gap_candidates.append({"text": text})

    if not gap_candidates:
        st.success("No major missing pieces were detected.")
        return

    for gap in gap_candidates:
        st.markdown(f"- {format_gap_plain_english(gap)}")


def _render_plan(plan_display: dict[str, Any]) -> None:
    if not plan_display:
        st.info("No generated plan available.")
        return

    if plan_display.get("time_estimate"):
        confidence = plan_display.get("time_confidence") or "unknown"
        st.markdown(f"**Estimated total time:** {plan_display['time_estimate']} _(confidence: {confidence})_")

    if plan_display.get("objective"):
        st.markdown("#### Objective")
        st.write(plan_display["objective"])

    st.markdown("#### Materials")
    materials = plan_display.get("materials", [])
    if materials:
        st.table(materials)
    else:
        st.caption("No materials listed.")

    st.markdown("#### Steps")
    checkpoint_map = plan_display.get("quality_checkpoints", {})
    global_warnings = plan_display.get("warnings", [])

    if not plan_display.get("steps"):
        st.caption("No steps generated.")
    else:
        for index, step in enumerate(plan_display["steps"], start=1):
            st.markdown(f"{index}. {step['text']}")
            if step.get("warning"):
                st.error(f"⚠️ Warning: {step['warning']}")
            step_number = step.get("number")
            if isinstance(step_number, int):
                for check in checkpoint_map.get(step_number, []):
                    st.warning(f"🧪 Quality checkpoint after step {step_number}: {check}")

    if global_warnings:
        with st.expander("Additional Safety Warnings"):
            for warning in global_warnings:
                st.error(f"⚠️ {warning}")


def _render_assumptions_and_bottom_line(sections: dict[str, Any], plan_display: dict[str, Any]) -> None:
    st.markdown("### What you need to know")
    assumptions = sections.get("assumptions", [])
    uncertainties = sections.get("uncertainties", [])
    plan_assumptions = [{"text": text} for text in plan_display.get("assumptions", [])]

    items = [*assumptions, *plan_assumptions, *uncertainties]
    if not items:
        st.caption("No major assumptions or uncertainties were highlighted.")
    else:
        for item in items:
            st.markdown(f"- {format_gap_plain_english(item)}")

    bottom_line = sections.get("bottom_line") or "No bottom line generated."
    st.markdown("### Bottom line")
    st.markdown(
        f"<div style='font-size:1.2rem;padding:1rem;border:2px solid #1d4ed8;border-radius:0.5rem;background:#eef6ff;'><strong>{bottom_line}</strong></div>",
        unsafe_allow_html=True,
    )


def _render_evidence_trail(pass_outputs: dict[str, Any]) -> None:
    evidence = pass_outputs.get("06_evidence_audit", {})
    claim_registry = evidence.get("claim_registry", []) if isinstance(evidence, dict) else []
    if not claim_registry:
        st.caption("No evidence trail available.")
        return

    for claim in claim_registry:
        claim_text = claim.get("text", "")
        claim_id = claim.get("claim_id", "unknown")
        status = claim.get("status", "unknown")
        basis = claim.get("basis", [])
        refs: list[str] = []
        for basis_item in basis:
            for source_ref in basis_item.get("source_refs", []):
                chunk_id = source_ref.get("source_chunk_id")
                if chunk_id:
                    refs.append(str(chunk_id))
        refs_text = ", ".join(sorted(set(refs))) if refs else "no direct source refs"
        st.markdown(f"- **{claim_id}** ({status}): {claim_text} — _Sources: {refs_text}_")


def _render_detailed_audit(pass_outputs: dict[str, Any], run_report: dict[str, Any], web_context_payload: dict[str, Any], retrieval_context_payload: dict[str, Any]) -> None:
    st.markdown("### Evidence trail")
    _render_evidence_trail(pass_outputs)

    st.markdown("### Pass-by-pass outputs")
    for pass_name in [*PASS_SEQUENCE, "classify_document", "search_web_context", "retrieval_context"]:
        payload = pass_outputs.get(pass_name)
        if payload is None:
            continue
        with st.expander(f"{pass_name}"):
            st.json(payload)

    with st.expander("Run report"):
        st.json(run_report)

    with st.expander("Web search results"):
        if not web_context_payload.get("web_context"):
            st.info("No web search context captured for this run.")
        else:
            st.json(web_context_payload)

    with st.expander("Retrieved reference context"):
        if not retrieval_context_payload.get("reference_context"):
            st.info("No local reference context retrieved for this run.")
        else:
            st.json(retrieval_context_payload)


def _load_plan_from_final(run_dir: Path) -> dict[str, Any]:
    plan_path = run_dir / "final" / "plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8")).get("plan", {})
    return parse_plan_output(run_dir)


def _ensure_plan_generated(pass_outputs: dict[str, Any], run_dir: Path) -> None:
    if pass_outputs.get("09_generate_plan"):
        return
    fallback = run_dir / "passes" / "09_generate_plan.json"
    if fallback.exists():
        pass_outputs["09_generate_plan"] = json.loads(fallback.read_text(encoding="utf-8"))


def _render_results(run_dir: Path) -> None:
    sections = parse_final_sections(run_dir)
    synthesis_path = run_dir / "passes" / "07_synthesize.json"
    if synthesis_path.exists():
        synthesis_payload = json.loads(synthesis_path.read_text(encoding="utf-8"))
    else:
        synthesis_payload = json.loads((run_dir / "final" / "final_answer.json").read_text(encoding="utf-8"))
    plan = _load_plan_from_final(run_dir)
    plan_display = format_plan_for_display(plan) if plan else {}
    pass_outputs = collect_pass_outputs(run_dir)
    _ensure_plan_generated(pass_outputs, run_dir)
    run_report = collect_run_report(run_dir)
    final_markdown = read_final_markdown(run_dir)

    web_context_path = run_dir / "passes" / "search_web_context.json"
    retrieval_context_path = run_dir / "passes" / "retrieval_context.json"
    web_context_payload = json.loads(web_context_path.read_text(encoding="utf-8")) if web_context_path.exists() else {}
    retrieval_context_payload = json.loads(retrieval_context_path.read_text(encoding="utf-8")) if retrieval_context_path.exists() else {}

    detected_type = "unknown"
    classification_path = run_dir / "passes" / "classify_document.json"
    if classification_path.exists():
        classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
        detected_type = str(classification_payload.get("selected_document_type") or classification_payload.get("document_type") or "unknown")

    st.subheader("Results")
    st.caption(f"Detected document type: `{detected_type}`")

    tabs = st.tabs(["Executive Summary", "Detailed Audit", "Generated Plan"])

    with tabs[0]:
        objective = plan_display.get("objective") or sections.get("goal") or "Objective unavailable"
        st.markdown(f"## {detected_type.replace('_', ' ').title()} — {objective}")

        _render_status_banner(get_status_color(synthesis_payload))

        st.markdown("### Here's what's missing")
        _render_gaps(sections, plan_display)

        st.markdown("### Corrected Plan")
        _render_plan(plan_display)

        _render_assumptions_and_bottom_line(sections, plan_display)

    with tabs[1]:
        _render_detailed_audit(pass_outputs, run_report, web_context_payload, retrieval_context_payload)

    with tabs[2]:
        if not plan:
            st.info("No generated plan available.")
        else:
            st.json(plan)

    st.download_button(
        "Download final_answer.md",
        data=final_markdown,
        file_name="final_answer.md",
        mime="text/markdown",
    )


def _run_pipeline(pipeline: AuditablePipeline, input_path: Path, temp_root: Path, user_goal: str, strict_mode: bool, document_type_choice: str, fast_mode: bool = False, parallel_chunks: int | None = None) -> tuple[Path | None, Exception | None]:
    result: dict[str, Path | Exception | None] = {"run_dir": None, "error": None}
    chunk_progress: dict[str, int] = {"current": 0, "total": 0}

    def _execute_pipeline() -> None:
        try:
            run_dir = pipeline.run(
                input_path=input_path,
                runs_dir=temp_root / "runs",
                user_goal=user_goal,
                strict=strict_mode,
                document_type=document_type_choice,
                fast=fast_mode,
                parallel_chunks=parallel_chunks,
                progress_callback=lambda current, total: chunk_progress.update({"current": current, "total": total}),
            )
            result["run_dir"] = run_dir
        except Exception as exc:  # noqa: BLE001 - surfaced to Streamlit users
            result["error"] = exc

    thread = threading.Thread(target=_execute_pipeline, daemon=True)
    thread.start()

    progress = st.progress(0, text="Starting pipeline...")
    status_placeholder = st.empty()
    pass_completion = {name: False for name in PASS_SEQUENCE}
    passes_dir = temp_root / "runs"

    while thread.is_alive():
        run_dirs = sorted(passes_dir.glob("*/passes")) if passes_dir.exists() else []
        if run_dirs:
            active_passes_dir = run_dirs[-1]
            for pass_name in PASS_SEQUENCE:
                if pass_name == "01_extract_chunk":
                    pass_completion[pass_name] = (active_passes_dir / "01_extract_chunk").exists() and any(
                        (active_passes_dir / "01_extract_chunk").glob("*.json")
                    )
                else:
                    pass_completion[pass_name] = (active_passes_dir / f"{pass_name}.json").exists()

        completed = sum(1 for done in pass_completion.values() if done)
        current = PASS_SEQUENCE[min(completed, len(PASS_SEQUENCE) - 1)]
        current_pass_index = PASS_SEQUENCE.index("01_extract_chunk") if "01_extract_chunk" in PASS_SEQUENCE else 0
        overall_progress = completed / len(PASS_SEQUENCE)
        chunk_total = chunk_progress.get("total", 0)
        chunk_current = chunk_progress.get("current", 0)
        if chunk_total > 0:
            overall_progress = max(overall_progress, (current_pass_index + (chunk_current / max(chunk_total, 1))) / len(PASS_SEQUENCE))
            progress.progress(overall_progress, text=f"Running {current}... Processing chunk {chunk_current} of {chunk_total}")
            status_placeholder.info(f"Completed {completed}/{len(PASS_SEQUENCE)} passes • Processing chunk {chunk_current} of {chunk_total}")
        else:
            # fallback to filesystem-based chunk progress when callback has not fired yet
            chunk_files = 0
            chunk_estimated_total = 0
            if run_dirs:
                chunks_path = run_dirs[-1].parent / "input" / "chunks.json"
                if chunks_path.exists():
                    try:
                        chunk_estimated_total = len(json.loads(chunks_path.read_text(encoding="utf-8")))
                    except json.JSONDecodeError:
                        chunk_estimated_total = 0
                extract_dir = run_dirs[-1] / "01_extract_chunk"
                if extract_dir.exists():
                    chunk_files = len(list(extract_dir.glob("*.json")))
            if chunk_estimated_total > 0 and chunk_files < chunk_estimated_total:
                overall_progress = max(overall_progress, (current_pass_index + (chunk_files / max(chunk_estimated_total, 1))) / len(PASS_SEQUENCE))
                progress.progress(overall_progress, text=f"Running {current}... Processing chunk {chunk_files} of {chunk_estimated_total}")
                status_placeholder.info(f"Completed {completed}/{len(PASS_SEQUENCE)} passes • Processing chunk {chunk_files} of {chunk_estimated_total}")
            else:
                progress.progress(overall_progress, text=f"Running {current}...")
                status_placeholder.info(f"Completed {completed}/{len(PASS_SEQUENCE)} passes")
        time.sleep(0.25)

    thread.join()

    if result["error"] is not None:
        st.error(f"Pipeline failed: {result['error']}")
        return None, result["error"] if isinstance(result["error"], Exception) else Exception("Unknown error")

    run_dir = result["run_dir"]
    if run_dir is None:
        st.error("Pipeline did not produce a run directory.")
        return None, Exception("missing run_dir")

    progress.progress(1.0, text="Pipeline complete")
    status_placeholder.success("Pipeline completed successfully")
    return run_dir, None


def main() -> None:
    default_claude_api_key = _secret_or_env("ANTHROPIC_API_KEY")
    default_brave_api_key = _secret_or_env("BRAVE_API_KEY")
    default_openai_api_key = _secret_or_env("OPENAI_API_KEY")

    cloud_mode = is_streamlit_cloud_environment()
    with st.sidebar:
        st.header("Run Settings")
        backend_options = get_available_backends(cloud_mode)
        backend = st.selectbox("Backend", backend_options, index=0)
        if cloud_mode:
            st.caption("Ollama is hidden in cloud mode. Use demo/Claude here, or self-host locally to use Ollama.")

        claude_api_key = ""
        claude_model = "claude-sonnet-4-20250514"
        if backend == "claude":
            claude_api_key = st.text_input("Claude API Key", type="password", value=default_claude_api_key)
            claude_model = st.text_input("Claude model", value="claude-sonnet-4-20250514")

        openai_api_key = ""
        openai_model = "gpt-4o"
        openai_base_url = "https://api.openai.com/v1"
        if backend == "openai":
            openai_api_key = st.text_input("OpenAI-compatible API Key", type="password", value=default_openai_api_key)
            openai_model = st.text_input("OpenAI-compatible model", value="gpt-4o")
            openai_base_url = st.text_input("API base URL", value="https://api.openai.com/v1")

        ollama_base_url = "http://127.0.0.1:11434"
        ollama_model = ""
        if backend == "ollama":
            ollama_base_url = st.text_input("Ollama base URL", value="http://127.0.0.1:11434")
            ollama_model = st.text_input("Ollama model name", value="")

        strict_mode = st.toggle("Strict mode", value=False)
        fast_mode = st.toggle("Fast mode", value=(backend in ("claude", "openai")), help="Use larger chunks, process chunks in parallel, and skip passes 05/06.")
        parallel_chunks = st.number_input("Parallel chunks", min_value=1, max_value=16, value=4 if backend in ("claude", "openai") else 1, step=1)
        enable_search = st.toggle("Web Search", value=False)
        brave_api_key = ""
        if enable_search:
            brave_api_key = st.text_input("Brave API Key", type="password", value=default_brave_api_key)
        st.markdown("---")
        st.subheader("Reference Documents")
        reference_dir_input = st.text_input("Reference directory path", value="")
        uploaded_references = st.file_uploader("Or upload reference files", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)

        user_goal = st.text_area(
            "User goal",
            value="Identify missing information and organize the document into an actionable structure.",
            height=140,
        )

        document_type_options = ["auto", *sorted(SUPPORTED_DOCUMENT_TYPES)]
        document_type_choice = st.selectbox("Document type", document_type_options, index=0)

        st.markdown("---")
        preflight_statuses = run_preflight(
            backend=backend,
            enable_search=enable_search,
            claude_api_key=claude_api_key or default_claude_api_key,
            brave_api_key=brave_api_key or default_brave_api_key,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            openai_api_key=openai_api_key or default_openai_api_key,
            openai_base_url=openai_base_url,
        )
        _render_capability_status(preflight_statuses)

    st.markdown("### Plan request")
    plan_request = st.text_area("Ask for a plan (optional)", value="", height=100, help="If provided without an uploaded document, the app will generate a starter planning document from your request and run the full pipeline on it.")

    uploaded_file = st.file_uploader("Upload main document", type=["txt", "md", "pdf", "docx"])
    run_clicked = st.button("Run Pipeline", type="primary", disabled=(uploaded_file is None and not plan_request.strip()))

    if not run_clicked:
        return

    use_plan_request = uploaded_file is None and bool(plan_request.strip())
    if uploaded_file is None and not use_plan_request:
        st.error("Please upload a main document or enter a plan request.")
        return
    if backend == "claude" and not preflight_statuses["claude_backend"].available:
        st.error(preflight_statuses["claude_backend"].message)
        return
    if backend == "ollama" and not preflight_statuses["ollama_backend"].available:
        st.error(preflight_statuses["ollama_backend"].message)
        return
    if backend == "openai" and not preflight_statuses["openai_backend"].available:
        st.error(preflight_statuses["openai_backend"].message)
        return
    if enable_search and not preflight_statuses["web_search"].available:
        st.error(preflight_statuses["web_search"].message)
        return

    try:
        with tempfile.TemporaryDirectory(prefix="auditable_pipeline_") as temp_dir:
            temp_root = Path(temp_dir)
            if use_plan_request:
                input_path = temp_root / "plan_request.txt"
                input_path.write_text(build_plan_request_document(plan_request), encoding="utf-8")
                st.info("Running pipeline from your plan request prompt.")
            else:
                input_path = temp_root / uploaded_file.name
                input_path.write_bytes(uploaded_file.getvalue())

            extraction_result = extract_text_from_path(input_path)
            if not extraction_result.ok:
                st.error(_format_extraction_error(extraction_result))
                return

            resolved_reference_dir = reference_dir_input.strip()
            if uploaded_references:
                local_reference_dir = temp_root / "reference_docs"
                local_reference_dir.mkdir(parents=True, exist_ok=True)
                for uploaded_reference in uploaded_references:
                    (local_reference_dir / uploaded_reference.name).write_bytes(uploaded_reference.getvalue())
                if not resolved_reference_dir:
                    resolved_reference_dir = str(local_reference_dir)

            config = PipelineConfig(
                claude_api_key=claude_api_key or default_claude_api_key,
                claude_model=claude_model,
                openai_api_key=openai_api_key or default_openai_api_key,
                openai_model=openai_model,
                openai_base_url=openai_base_url,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_model,
                enable_search=enable_search,
                brave_api_key=brave_api_key,
                reference_dir=resolved_reference_dir,
            )

            pipeline = AuditablePipeline(repo_root=Path(__file__).parent, backend_name=backend, config=config)
            run_dir, error = _run_pipeline(pipeline, input_path, temp_root, user_goal, strict_mode, document_type_choice, fast_mode=fast_mode, parallel_chunks=int(parallel_chunks))
            if error or run_dir is None:
                return
            _render_results(run_dir)
    except Exception as exc:  # noqa: BLE001 - catch top-level app exceptions
        st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
