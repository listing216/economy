"""
method_utils.py - 挖掘方法层的共用工具

将“论文方法里都会复用的表达式分析能力”抽离出来：
- AST 相似度与子树统计
- 复杂度特征与复杂度分数
- 假设 / 描述 / 表达式的一致性启发式评分
- 因子换手代理指标
- Frequent Subtree Avoidance 所需的频繁子树挖掘
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


_THEME_LIBRARY = {
    "trend": {
        "keywords": ["trend", "momentum", "延续", "趋势", "强势", "breakout", "突破"],
        "evidence": ["delta", "pct", "rank", "mean", "close", "vwap", "high", "low"],
    },
    "reversal": {
        "keywords": ["reversal", "mean reversion", "反转", "回归", "超跌", "超涨"],
        "evidence": ["rank", "quantile", "delta", "mean", "std", "close", "vwap"],
    },
    "volatility": {
        "keywords": ["volatility", "波动", "震荡", "range", "dispersion"],
        "evidence": ["std", "skew", "kurt", "high", "low", "abs", "delta"],
    },
    "liquidity": {
        "keywords": ["liquidity", "volume", "flow", "成交", "流动性", "换手", "participation"],
        "evidence": ["volume", "amount", "vwap", "sum", "mean", "corr"],
    },
    "price_volume": {
        "keywords": ["量价", "价量", "price-volume", "confirmation", "确认"],
        "evidence": ["close", "vwap", "volume", "amount", "corr", "rank"],
    },
    "relative_value": {
        "keywords": ["relative", "归一", "normal", "deviation", "偏离", "spread"],
        "evidence": ["div", "sub", "mean", "std", "close", "vwap"],
    },
}


@dataclass
class EvaluatedCandidate:
    alpha_id: str
    expression: str
    explanation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    factor_values: pd.DataFrame | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "alpha_id": self.alpha_id,
            "expression": self.expression,
            "explanation": self.explanation,
        }
        record.update(self.metrics)
        record.update(self.extra)
        return record


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def parse_expression_ast(expression: str):
    try:
        return ast.parse(expression, mode="eval").body
    except Exception:
        return None


def ast_size(node) -> int:
    if node is None:
        return 0
    return 1 + sum(ast_size(child) for child in ast.iter_child_nodes(node))


def ast_depth(node) -> int:
    if node is None:
        return 0
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(ast_depth(child) for child in children)


def _node_signature(node, abstract_numbers: bool = False):
    if isinstance(node, ast.Call):
        func = _node_signature(node.func, abstract_numbers)
        args = tuple(_node_signature(arg, abstract_numbers) for arg in node.args)
        return ("Call", func, args)
    if isinstance(node, ast.Name):
        return ("Name", node.id.lower())
    if isinstance(node, ast.Attribute):
        return ("Attr", _node_signature(node.value, abstract_numbers), node.attr.lower())
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return ("Num", "t" if abstract_numbers else str(node.value))
        return ("Const", repr(node.value))
    if isinstance(node, ast.BinOp):
        return (
            "BinOp",
            type(node.op).__name__,
            _node_signature(node.left, abstract_numbers),
            _node_signature(node.right, abstract_numbers),
        )
    if isinstance(node, ast.UnaryOp):
        return ("UnaryOp", type(node.op).__name__, _node_signature(node.operand, abstract_numbers))
    if isinstance(node, ast.Compare):
        return (
            "Compare",
            tuple(type(op).__name__ for op in node.ops),
            tuple(_node_signature(x, abstract_numbers) for x in [node.left] + node.comparators),
        )
    return ast.dump(node, include_attributes=False)


def _abstract_expression(node) -> str:
    if isinstance(node, ast.Call):
        func = _abstract_expression(node.func)
        args = ", ".join(_abstract_expression(arg) for arg in node.args)
        return f"{func}({args})"
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return f"{_abstract_expression(node.value)}.{node.attr.lower()}"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return "t"
        return repr(node.value)
    if isinstance(node, ast.BinOp):
        op = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
            ast.Mod: "%",
        }.get(type(node.op), "?")
        return f"({_abstract_expression(node.left)} {op} {_abstract_expression(node.right)})"
    if isinstance(node, ast.UnaryOp):
        op = {ast.UAdd: "+", ast.USub: "-", ast.Not: "not "}.get(type(node.op), "")
        return f"{op}{_abstract_expression(node.operand)}"
    return ast.dump(node, include_attributes=False)


def collect_subtree_sizes(expression: str, abstract_numbers: bool = True) -> dict[Any, int]:
    root = parse_expression_ast(expression)
    if root is None:
        return {}

    sizes = {}

    def _walk(node):
        signature = _node_signature(node, abstract_numbers=abstract_numbers)
        sizes[signature] = max(sizes.get(signature, 0), ast_size(node))
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(root)
    return sizes


def normalized_ast_similarity(expression_a: str, expression_b: str) -> float:
    sizes_a = collect_subtree_sizes(expression_a, abstract_numbers=True)
    sizes_b = collect_subtree_sizes(expression_b, abstract_numbers=True)
    if not sizes_a or not sizes_b:
        return 0.0

    common = set(sizes_a) & set(sizes_b)
    if not common:
        return 0.0

    common_size = max(min(sizes_a[sig], sizes_b[sig]) for sig in common)
    max_size = max(max(sizes_a.values()), max(sizes_b.values()))
    return common_size / max(1, max_size)


def max_similarity_to_records(expression: str, records: list[dict]) -> tuple[float, dict | None]:
    best_score = 0.0
    best_record = None
    for record in records or []:
        other_expr = str(record.get("expression", "")).strip()
        if not other_expr:
            continue
        score = normalized_ast_similarity(expression, other_expr)
        if score > best_score:
            best_score = score
            best_record = record
    return best_score, best_record


def compute_complexity_features(expression: str, fields: list[str]) -> dict[str, Any]:
    root = parse_expression_ast(expression)
    if root is None:
        return {
            "symbolic_length": 0,
            "operator_count": 0,
            "parameter_count": 0,
            "feature_count": 0,
            "depth": 0,
            "features": [],
        }

    field_set = {x.lower() for x in fields}
    operators = 0
    parameters = 0
    used_features = set()

    for node in ast.walk(root):
        if isinstance(node, ast.Call):
            operators += 1
        elif isinstance(node, ast.BinOp):
            operators += 1
        elif isinstance(node, ast.UnaryOp):
            operators += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            parameters += 1
        elif isinstance(node, ast.Name) and node.id.lower() in field_set:
            used_features.add(node.id.lower())

    return {
        "symbolic_length": ast_size(root),
        "operator_count": operators,
        "parameter_count": parameters,
        "feature_count": len(used_features),
        "depth": ast_depth(root),
        "features": sorted(used_features),
    }


def compute_complexity_score(
    complexity: dict[str, Any],
    max_symbolic_length: int = 28,
    max_free_params: int = 6,
    max_features: int = 5,
    max_depth: int = 7,
) -> float:
    length_penalty = min(1.0, safe_float(complexity.get("symbolic_length"), 0.0) / max(1, max_symbolic_length))
    param_penalty = min(1.0, safe_float(complexity.get("parameter_count"), 0.0) / max(1, max_free_params))
    feature_penalty = min(1.0, safe_float(complexity.get("feature_count"), 0.0) / max(1, max_features))
    depth_penalty = min(1.0, safe_float(complexity.get("depth"), 0.0) / max(1, max_depth))
    score = 1.0 - (0.4 * length_penalty + 0.2 * param_penalty + 0.2 * feature_penalty + 0.2 * depth_penalty)
    return max(0.0, min(1.0, score))


def _infer_themes(text: str) -> list[str]:
    text = (text or "").lower()
    matched = []
    for theme, rule in _THEME_LIBRARY.items():
        if any(keyword in text for keyword in rule["keywords"]):
            matched.append(theme)
    return matched


def _expression_tokens(expression: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_]+", expression or "")}


def compute_hypothesis_alignment_score(
    hypothesis: str,
    description: str,
    expression: str,
    fields: list[str],
) -> float:
    tokens = _expression_tokens(expression)
    theme_hits = []
    themes = _infer_themes(f"{hypothesis}\n{description}")

    for theme in themes:
        evidence = set(_THEME_LIBRARY[theme]["evidence"])
        overlap = len(tokens & evidence) / max(1, len(evidence))
        theme_hits.append(overlap)

    field_hits = 0.0
    lower_description = f"{hypothesis}\n{description}".lower()
    field_set = {f.lower() for f in fields}
    mentioned_fields = [field for field in field_set if field in lower_description]
    if mentioned_fields:
        field_hits = len([field for field in mentioned_fields if field in tokens]) / len(mentioned_fields)

    if not theme_hits and not mentioned_fields:
        return 0.55

    theme_score = sum(theme_hits) / max(1, len(theme_hits)) if theme_hits else 0.55
    return max(0.0, min(1.0, 0.8 * theme_score + 0.2 * field_hits))


def compute_turnover_proxy(factor_values: pd.DataFrame) -> float:
    if factor_values is None or factor_values.empty:
        return np.nan

    values = factor_values.replace([np.inf, -np.inf], np.nan)
    row_mean = values.mean(axis=1)
    row_std = values.std(axis=1).replace(0, np.nan)
    zscore = values.sub(row_mean, axis=0).div(row_std, axis=0)
    drift = zscore.diff().abs().to_numpy(dtype=float)
    if np.isnan(drift).all():
        return np.nan
    return float(np.nanmean(drift))


def percentile_rank_score(value: float, peers: list[float], higher_better: bool = True) -> float:
    peers = [safe_float(v) for v in peers if pd.notna(v)]
    value = safe_float(value)
    if not peers:
        return max(0.0, min(1.0, value))

    if higher_better:
        rank = sum(v <= value for v in peers) / len(peers)
    else:
        rank = sum(v >= value for v in peers) / len(peers)
    return max(0.0, min(1.0, rank))


def _leaves_are_raw_fields(node, fields: set[str]) -> bool:
    children = list(ast.iter_child_nodes(node))
    if not children:
        if isinstance(node, ast.Name):
            return node.id.lower() in fields
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return True
        return False
    return all(_leaves_are_raw_fields(child, fields) for child in children)


def mine_frequent_subtrees(
    records: list[dict],
    fields: list[str],
    top_k: int = 5,
    min_size: int = 3,
) -> list[dict[str, Any]]:
    if not records:
        return []

    field_set = {f.lower() for f in fields}
    counter = Counter()
    pattern_text = {}
    pattern_size = {}
    total = 0

    for record in records:
        expression = str(record.get("expression", "")).strip()
        root = parse_expression_ast(expression)
        if root is None:
            continue
        total += 1
        seen = set()
        for node in ast.walk(root):
            size = ast_size(node)
            if size < min_size:
                continue
            if not _leaves_are_raw_fields(node, field_set):
                continue
            signature = _node_signature(node, abstract_numbers=True)
            seen.add(signature)
            pattern_text[signature] = _abstract_expression(node)
            pattern_size[signature] = size
        for signature in seen:
            counter[signature] += 1

    if total == 0:
        return []

    ranked = sorted(
        counter.items(),
        key=lambda item: (-item[1], -pattern_size.get(item[0], 0), pattern_text.get(item[0], "")),
    )
    results = []
    for signature, support_count in ranked[:top_k]:
        results.append(
            {
                "signature": signature,
                "pattern": pattern_text.get(signature, ""),
                "support": support_count / total,
                "count": support_count,
                "size": pattern_size.get(signature, 0),
            }
        )
    return results


def expression_contains_subtree(expression: str, signature: Any) -> bool:
    root = parse_expression_ast(expression)
    if root is None:
        return False
    return any(_node_signature(node, abstract_numbers=True) == signature for node in ast.walk(root))


def softmax_choice_index(values: list[float], temperature: float = 1.0) -> int:
    if not values:
        raise ValueError("softmax_choice_index requires a non-empty list")

    temp = max(1e-6, temperature)
    arr = np.array(values, dtype=float)
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr / temp)
    probs = exp_arr / np.sum(exp_arr)
    return int(np.random.choice(len(values), p=probs))


def clipped_ratio(value: float, scale: float) -> float:
    value = safe_float(value, 0.0)
    scale = max(scale, 1e-8)
    return max(0.0, min(1.0, value / scale))
