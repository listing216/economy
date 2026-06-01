"""Rule-based interpretability scoring.

The score is deliberately simple and deterministic.  It favors expressions that
are shallow, use a small number of fields/operators, and can be mapped to at
least one semantic tag.
"""
from __future__ import annotations


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def compute_interpretability_score(explainability: dict, config: dict | None = None) -> dict:
    cfg = (config or {}).get("scoring", {}) if isinstance(config, dict) else {}
    complexity = explainability.get("complexity", {}) or {}

    num_nodes = int(complexity.get("num_nodes", 0) or 0)
    max_depth = int(complexity.get("max_depth", 0) or 0)
    num_fields = int(complexity.get("num_fields", 0) or 0)
    num_operators = int(complexity.get("num_operators", 0) or 0)

    structure_score = 1.0
    structure_score -= max(0, num_nodes - cfg.get("ideal_max_nodes", 10)) * 0.03
    structure_score -= max(0, max_depth - cfg.get("ideal_max_depth", 5)) * 0.08
    structure_score = clamp(structure_score)

    field_score = 1.0 - max(0, num_fields - cfg.get("ideal_max_fields", 3)) * 0.10
    field_score = clamp(field_score)

    operator_score = 1.0 - max(0, num_operators - cfg.get("ideal_max_operators", 5)) * 0.08
    operator_score = clamp(operator_score)

    semantic_score = 1.0 if explainability.get("semantic_tags") else 0.60

    score = (
        0.35 * structure_score
        + 0.25 * field_score
        + 0.25 * operator_score
        + 0.15 * semantic_score
    )
    score = clamp(score)

    return {
        "interpretability_score": round(score, 6),
        "complexity_penalty": round(1.0 - structure_score, 6),
        "score_breakdown": {
            "structure_score": round(structure_score, 6),
            "field_score": round(field_score, 6),
            "operator_score": round(operator_score, 6),
            "semantic_score": round(semantic_score, 6),
        },
    }
