"""Structured explainability generation for alpha expressions."""
from __future__ import annotations

from typing import Any

from .expression_ast import summarize_expression
from .scoring import compute_interpretability_score
from .semantics import (
    get_field_semantic,
    get_operator_semantic,
    load_field_semantics,
    load_operator_semantics,
)


PRICE_FIELDS = {"open", "high", "low", "close", "vwap"}
LIQUIDITY_FIELDS = {"volume", "amount", "turnover", "turnover_rate"}
VOLATILITY_OPERATORS = {"Std", "Ts_Std", "TsStd", "stddev"}
MEAN_OPERATORS = {"Mean", "Ts_Mean", "TsMean", "Sma", "Wma"}
CORR_OPERATORS = {"Corr", "Correlation", "Ts_Corr"}
DELTA_OPERATORS = {"Delta", "Diff"}
RANK_OPERATORS = {"Rank", "Ts_Rank", "TsRank"}


def explain_factor(
    expression: str,
    known_fields: set[str] | None = None,
    known_operators: set[str] | None = None,
    llm_explanation: str | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Build a deterministic structured explanation for an alpha expression."""
    cfg = config or {}
    summary = summarize_expression(expression, known_fields, known_operators)

    field_semantics = load_field_semantics(
        cfg.get("semantics", {}).get("field_semantics_path")
    )
    operator_semantics = load_operator_semantics(
        cfg.get("semantics", {}).get("operator_semantics_path")
    )

    fields = summary["fields"]
    operators = summary["operators"]
    windows = summary["windows"]

    field_explanations = [get_field_semantic(f, field_semantics) for f in fields]
    operator_explanations = [get_operator_semantic(op, operator_semantics) for op in operators]
    semantic_tags = infer_semantic_tags(fields, operators)

    explainability = {
        "expression": expression,
        "parse_status": summary["parse_status"],
        "ast": summary["ast"],
        "fields": fields,
        "operators": operators,
        "windows": windows,
        "complexity": summary["complexity"],
        "semantic_tags": semantic_tags,
        "field_explanations": field_explanations,
        "operator_explanations": operator_explanations,
        "sub_expression_explanations": explain_subexpressions(
            summary["sub_expressions"], known_fields, known_operators, field_semantics, operator_semantics
        ),
        "llm_explanation": llm_explanation or "",
        "warnings": build_warnings(summary, cfg),
    }

    explainability.update(compute_interpretability_score(explainability, cfg))
    return explainability


def infer_semantic_tags(fields: list[str], operators: list[str]) -> list[str]:
    field_set = set(fields)
    op_set = set(operators)
    tags: set[str] = set()

    if field_set & PRICE_FIELDS:
        tags.add("price_action")
    if field_set & LIQUIDITY_FIELDS:
        tags.add("liquidity")
    if op_set & DELTA_OPERATORS and field_set & PRICE_FIELDS:
        tags.add("momentum_or_reversal")
    if op_set & VOLATILITY_OPERATORS:
        tags.add("volatility")
    if op_set & MEAN_OPERATORS:
        tags.add("smoothing_or_baseline")
    if op_set & CORR_OPERATORS:
        tags.add("time_series_relation")
        if field_set & LIQUIDITY_FIELDS:
            tags.add("price_volume_relation")
    if op_set & RANK_OPERATORS:
        tags.add("relative_strength")

    return sorted(tags)


def explain_subexpressions(
    sub_expressions: list[str],
    known_fields: set[str] | None,
    known_operators: set[str] | None,
    field_semantics: dict,
    operator_semantics: dict,
) -> list[dict[str, Any]]:
    results = []
    for sub_expr in sub_expressions:
        s = summarize_expression(sub_expr, known_fields, known_operators)
        fields = s["fields"]
        operators = s["operators"]
        windows = s["windows"]
        results.append(
            {
                "sub_expression": sub_expr,
                "fields": fields,
                "operators": operators,
                "windows": windows,
                "meaning": _compose_subexpression_meaning(
                    fields, operators, windows, field_semantics, operator_semantics
                ),
            }
        )
    return results


def _compose_subexpression_meaning(
    fields: list[str],
    operators: list[str],
    windows: list[int],
    field_semantics: dict,
    operator_semantics: dict,
) -> str:
    parts = []
    if fields:
        field_names = [get_field_semantic(f, field_semantics)["name_cn"] for f in fields]
        parts.append("涉及字段：" + "、".join(field_names))
    if operators:
        op_meanings = [get_operator_semantic(op, operator_semantics)["meaning"] for op in operators]
        parts.append("使用算子：" + "；".join(op_meanings))
    if windows:
        parts.append("窗口参数：" + "、".join(str(w) for w in windows))
    if not parts:
        return "该子表达式暂未匹配到明确字段或算子语义。"
    return "；".join(parts) + "。"


def build_warnings(summary: dict[str, Any], config: dict | None = None) -> list[str]:
    cfg = config or {}
    ast_cfg = cfg.get("ast", {}) if isinstance(cfg, dict) else {}
    complexity = summary.get("complexity", {}) or {}
    warnings = []

    if summary.get("parse_status") != "success":
        warnings.append("表达式 AST 解析失败，解释性结果仅供参考。")

    max_depth = ast_cfg.get("max_depth")
    if max_depth and complexity.get("max_depth", 0) > max_depth:
        warnings.append(f"表达式深度超过配置阈值 {max_depth}。")

    max_nodes = ast_cfg.get("max_nodes")
    if max_nodes and complexity.get("num_nodes", 0) > max_nodes:
        warnings.append(f"表达式节点数超过配置阈值 {max_nodes}。")

    return warnings
