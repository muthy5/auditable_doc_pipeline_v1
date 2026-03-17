# UX + Materials Improvement Code Change Plan

This plan translates product/UX recommendations into concrete code changes for the Streamlit app and supporting utilities.

## 1) Add Simple vs Advanced run configuration mode

### Files
- `app.py`

### Changes
- Add a top-level sidebar toggle/select for `Configuration mode` with values:
  - `Simple`
  - `Advanced`
- In `Simple` mode, hide advanced controls and expose only:
  - backend
  - user goal
  - optional document type
  - one "Quality profile" selector with presets:
    - `Quick check` (fast on, strict off, parallel chunks low)
    - `Deep audit` (fast off, strict on)
- In `Advanced` mode, keep current controls (`strict_mode`, `fast_mode`, `parallel_chunks`, search toggles, etc.).
- Add helper copy under each profile describing speed/completeness tradeoff.

## 2) Add "What happens next" execution stepper and ETA

### Files
- `app.py`
- `app_utils.py`

### Changes
- Add a display component before run start that shows plain-language stage mapping for `PASS_SEQUENCE`.
- Create helper in `app_utils.py`:
  - `get_pipeline_stage_descriptions() -> list[dict[str, str]]`
- Add lightweight ETA estimate function:
  - `estimate_runtime_seconds(chunk_count: int, fast_mode: bool, backend: str) -> int`
- During active run (`_wait_for_pipeline`), show:
  - current stage description
  - estimated remaining time

## 3) Make Executive Summary more action-oriented

### Files
- `app.py`
- `app_utils.py`

### Changes
- Add `_render_top_actions(...)` above "Here’s what’s missing".
- Build top actions from:
  - `missing_information`
  - `dependencies`
  - `plan_display.blocking_items`
- Add readiness score helper in `app_utils.py`:
  - `compute_readiness_score(synthesis_payload: dict[str, Any]) -> dict[str, Any]`
  - Returns score (0-100), level label, and rationale counts.
- Show readiness score card next to existing red/yellow/green banner.

## 4) Upgrade Materials section to procurement checklist

### Files
- `app_utils.py`
- `app.py`
- `schemas/09_generate_plan.schema.json` (optional schema extension)
- `prompts/09_generate_plan.txt` (prompt extension)

### Changes
- Extend `format_plan_for_display()` material mapping with optional fields:
  - `required` (bool)
  - `have_it` (default false UI-side)
  - `estimated_cost`
  - `vendor_or_source`
  - `substitutes` (list[str])
  - `used_in_steps` (list[int])
  - `notes`
- Replace plain `st.table(materials)` with richer UI:
  - grouped material tables (prep/execution/safety) when category exists
  - critical missing material warning summary
  - checkboxes for "Have it" state in-session
  - quick actions:
    - copy checklist
    - download CSV
- Improve empty state message with guidance:
  - suggest enabling web search / reference docs
  - suggest follow-up prompt for missing quantities/substitutes

## 5) Add evidence confidence and weak-claim filtering

### Files
- `app.py`
- `app_utils.py`
- `schemas/06_evidence_audit.schema.json` (optional)
- `prompts/06_evidence_audit.txt` (optional)

### Changes
- Add confidence/severity badge rendering in `_render_evidence_trail`.
- Add checkbox/toggle `Show weakly supported claims only` in Detailed Audit tab.
- Add helper:
  - `classify_claim_strength(claim: dict[str, Any]) -> str`
- Differentiate evidence origin badges:
  - document chunk
  - web context
  - inferred assumption

## 6) Improve Detailed Audit information hierarchy

### Files
- `app.py`
- `app_utils.py`

### Changes
- Add pass summary rows before raw payload expanders:
  - status
  - duration (from timing)
  - fallback usage
  - warning/error counts
- Add filters:
  - only failed/skipped/fallback passes
  - only passes with warnings
- Keep current raw JSON in expanders for power users.

## 7) Better extraction failure recovery guidance

### Files
- `app.py`
- `README.md`

### Changes
- Expand `_format_extraction_error` usage to include remediation tips and next-step actions.
- Add remediation snippets in UI per error type:
  - missing parser packages
  - image-only PDF (OCR recommendation)
  - unsupported file type conversion tips
- Add a README "Troubleshooting upload failures" section linked from UI text.

## 8) Export improvements for user workflows

### Files
- `app.py`
- `app_utils.py`

### Changes
- Add downloads beyond `final_answer.md`:
  - `materials_checklist.csv`
  - `top_actions.md`
  - `audit_summary.md`
- Add helpers to generate these exports from run artifacts.

## 9) Copy and labeling improvements

### Files
- `app.py`

### Changes
- Rename primary CTA from `Run Pipeline` to `Analyze Document & Build Plan`.
- Add short helper text near `Plan request` with example prompts.
- Add audience framing subtitle under app title (operator / reviewer / auditor use cases).

## 10) Tests to add with implementation

### Files
- `tests/test_app_utils.py`
- `tests/test_runtime_diagnostics.py` (if preflight messaging changes)

### Changes
- Add tests for new utility helpers:
  - readiness score
  - stage descriptions
  - ETA estimation bounds
  - material formatting defaults and optional fields
  - claim strength classification
- Add tests ensuring existing behavior remains backward compatible when optional plan fields are absent.

## Suggested implementation order

1. `app_utils.py` helper additions (pure functions + tests)
2. Materials rendering upgrade in `app.py`
3. Executive summary action/readiness cards
4. Detailed audit filtering and claim strength badges
5. Sidebar Simple/Advanced mode and profile presets
6. Exports and copy updates
7. Prompt/schema optional extensions

## Definition of done

- New users can run with `Simple` mode without seeing advanced controls.
- Executive summary provides an explicit top action list + readiness score.
- Materials section acts as a checklist/exportable procurement list.
- Detailed audit is filterable and highlights weak evidence quickly.
- Upload failures provide immediate, actionable recovery steps.
