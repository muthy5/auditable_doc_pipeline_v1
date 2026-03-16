from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def normalize_text(value: str) -> str:
    """Normalize text for approximate deduplication.

    Args:
        value: Input text value.

    Returns:
        Lowercased text with collapsed whitespace.
    """
    return " ".join(value.lower().split())


def merge_chunk_extractions(doc_id: str, chunk_extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-chunk extraction outputs into global aggregates.

    Args:
        doc_id: Document identifier.
        chunk_extractions: Outputs from the extract-chunk pass.

    Returns:
        Combined merge artifact for downstream passes.
    """
    entity_buckets: Dict[str, set[str]] = defaultdict(set)
    defined_terms: Dict[str, Dict[str, Any]] = {}
    undefined_terms: set[str] = set()
    fact_clusters: Dict[str, Dict[str, Any]] = {}
    missing_info_clusters: Dict[str, Dict[str, Any]] = {}
    all_claims: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    all_dependencies: List[Dict[str, Any]] = []
    all_inputs_required: List[str] = []
    all_outputs_produced: List[str] = []
    cross_reference_graph: List[Dict[str, Any]] = []
    merge_warnings: List[str] = []

    for chunk in chunk_extractions:
        chunk_id = chunk["chunk_id"]

        for bucket_name, items in chunk.get("entities", {}).items():
            if isinstance(items, list):
                for item in items:
                    entity_buckets[bucket_name].add(item)

        for term in chunk.get("defined_terms", []):
            if term not in defined_terms:
                defined_terms[term] = {
                    "term": term,
                    "defined_in_chunk_id": chunk_id,
                    "used_in_chunk_ids": [chunk_id],
                }
            elif chunk_id not in defined_terms[term]["used_in_chunk_ids"]:
                defined_terms[term]["used_in_chunk_ids"].append(chunk_id)

        for term in chunk.get("undefined_terms", []):
            undefined_terms.add(term)

        for fact in chunk.get("explicit_facts", []):
            key = normalize_text(fact["text"])
            if key not in fact_clusters:
                fact_clusters[key] = {
                    "fact_id": fact["fact_id"],
                    "text": fact["text"],
                    "source_chunk_ids": [chunk_id],
                    "source_refs": [
                        {
                            "source_chunk_id": chunk_id,
                            "source_span": fact["source_span"],
                        }
                    ],
                }
            else:
                if chunk_id not in fact_clusters[key]["source_chunk_ids"]:
                    fact_clusters[key]["source_chunk_ids"].append(chunk_id)
                    fact_clusters[key]["source_refs"].append(
                        {
                            "source_chunk_id": chunk_id,
                            "source_span": fact["source_span"],
                        }
                    )

        for claim in chunk.get("claims", []):
            all_claims.append(
                {
                    "claim_id": claim["claim_id"],
                    "text": claim["text"],
                    "source_chunk_id": chunk_id,
                }
            )

        for step in chunk.get("steps", []):
            all_steps.append({**step, "source_chunk_id": chunk_id})

        for dep in chunk.get("dependencies_mentioned", []):
            all_dependencies.append({**dep, "source_chunk_id": chunk_id})

        all_inputs_required.extend(chunk.get("inputs_required", []))
        all_outputs_produced.extend(chunk.get("outputs_produced", []))

        for signal in chunk.get("missing_information_signals", []):
            key = normalize_text(signal)
            if key not in missing_info_clusters:
                missing_info_clusters[key] = {
                    "text": signal,
                    "source_chunk_ids": [chunk_id],
                    "count": 1,
                }
            else:
                missing_info_clusters[key]["count"] += 1
                if chunk_id not in missing_info_clusters[key]["source_chunk_ids"]:
                    missing_info_clusters[key]["source_chunk_ids"].append(chunk_id)

        for ref in chunk.get("cross_references", []):
            cross_reference_graph.append(
                {
                    "source_chunk_id": chunk_id,
                    "ref_text": ref["ref_text"],
                    "resolved_chunk_id": None,
                }
            )

    return {
        "doc_id": doc_id,
        "chunks_seen": [chunk["chunk_id"] for chunk in chunk_extractions],
        "global_entities": {name: sorted(values) for name, values in entity_buckets.items()},
        "global_defined_terms": sorted(defined_terms.keys()),
        "global_undefined_terms": sorted(undefined_terms),
        "all_explicit_facts": [
            {
                "fact_id": cluster["fact_id"],
                "text": cluster["text"],
                "source_chunk_ids": cluster["source_chunk_ids"],
                "source_refs": cluster["source_refs"],
            }
            for cluster in fact_clusters.values()
        ],
        "all_claims": all_claims,
        "all_steps": all_steps,
        "all_dependencies": all_dependencies,
        "all_inputs_required": sorted(set(all_inputs_required)),
        "all_outputs_produced": sorted(set(all_outputs_produced)),
        "all_missing_information_signals": list(missing_info_clusters.values()),
        "cross_reference_graph": cross_reference_graph,
        "term_registry": list(defined_terms.values()),
        "merge_warnings": merge_warnings,
    }
