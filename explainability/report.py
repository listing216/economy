"""Save explainability JSON and Markdown reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_explainability_artifacts(
    factor_id: str,
    expression: str,
    explainability: dict[str, Any],
    metrics: dict[str, Any] | None,
    output_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_id = str(factor_id).replace("/", "_")
    json_path = out_dir / f"{safe_id}_explainability.json"
    md_path = out_dir / f"{safe_id}_report.md"

    payload = {
        "factor_id": factor_id,
        "expression": expression,
        "metrics": _json_safe(metrics or {}),
        "explainability": _json_safe(explainability or {}),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(factor_id, expression, explainability, metrics or {}))

    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown_report(
    factor_id: str,
    expression: str,
    explainability: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    exp = explainability or {}
    lines = [
        f"# 因子解释性报告：{factor_id}",
        "",
        "## 1. 基本信息",
        "",
        f"- 因子 ID：`{factor_id}`",
        f"- 表达式：`{expression}`",
        f"- 解析状态：{exp.get('parse_status', 'unknown')}",
        f"- 解释性评分：{exp.get('interpretability_score', '')}",
        f"- 语义标签：{', '.join(exp.get('semantic_tags', []))}",
        "",
        "## 2. 表达式结构",
        "",
        f"- 字段：{', '.join(exp.get('fields', []))}",
        f"- 算子：{', '.join(exp.get('operators', []))}",
        f"- 窗口：{', '.join(map(str, exp.get('windows', [])))}",
        f"- 复杂度：`{exp.get('complexity', {})}`",
        "",
        "## 3. 字段解释",
        "",
        "| 字段 | 类型 | 中文名 | 含义 | 金融角色 |",
        "|---|---|---|---|---|",
    ]
    for item in exp.get("field_explanations", []):
        lines.append(
            f"| {item.get('field','')} | {item.get('type','')} | {item.get('name_cn','')} | "
            f"{item.get('meaning','')} | {item.get('role','')} |"
        )

    lines += [
        "",
        "## 4. 算子解释",
        "",
        "| 算子 | 类型 | 含义 | 金融解释 | 风险 |",
        "|---|---|---|---|---|",
    ]
    for item in exp.get("operator_explanations", []):
        lines.append(
            f"| {item.get('operator','')} | {item.get('category','')} | {item.get('meaning','')} | "
            f"{item.get('financial_interpretation','')} | {item.get('risk','')} |"
        )

    lines += [
        "",
        "## 5. 子表达式解释",
        "",
        "| 子表达式 | 含义 |",
        "|---|---|",
    ]
    for item in exp.get("sub_expression_explanations", []):
        lines.append(f"| `{item.get('sub_expression','')}` | {item.get('meaning','')} |")

    lines += [
        "",
        "## 6. 主要指标",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
    ]
    for key in ["ic_mean", "ic_ir", "long_excret", "long_sharpe", "ls_ret", "ls_sharpe"]:
        if key in metrics:
            lines.append(f"| {key} | {metrics.get(key)} |")

    warnings = exp.get("warnings", [])
    if warnings:
        lines += ["", "## 7. 风险提示", ""]
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj
