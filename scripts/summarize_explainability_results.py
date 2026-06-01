#!/usr/bin/env python3
"""
Summarize factor explainability artifacts into experiment-level tables.

This script is intentionally independent from the mining runtime. It scans:
  - results/explainability/**/*_explainability.json
  - results/metrics/**/*.{csv,json,jsonl}

and writes:
  - results/explainability_summary.csv
  - results/explainability_summary.json
  - results/explainability_summary.md

Usage:
  python scripts/summarize_explainability_results.py
  python scripts/summarize_explainability_results.py --results-dir results
  python scripts/summarize_explainability_results.py --out-dir results/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_COLUMNS = [
    "alpha_id",
    "cycle",
    "expression",
    "parse_status",
    "interpretability_score",
    "complexity_penalty",
    "num_nodes",
    "max_depth",
    "num_fields",
    "num_operators",
    "fields",
    "operators",
    "windows",
    "semantic_tags",
    "ic_mean",
    "ic_ir",
    "ict",
    "icstocknum",
    "long_excret",
    "long_sharpe",
    "long_ir",
    "long_excmdd",
    "ls_ret",
    "ls_std",
    "ls_sharpe",
    "ls_mdd",
    "ic_5_mean",
    "ic_20_mean",
    "source_explainability_file",
    "source_metrics_file",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                loaded = json.loads(value)
                return loaded if isinstance(loaded, list) else [loaded]
            except Exception:
                pass
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]


def _join_list(value: Any) -> str:
    items = _as_list(value)
    return ",".join(str(x) for x in items if str(x) != "")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _guess_alpha_id(path: Path, data: dict[str, Any]) -> str:
    for key in ("alpha_id", "factor_id", "name", "id"):
        value = data.get(key)
        if value:
            return str(value)

    stem = path.stem
    stem = re.sub(r"_explainability$", "", stem)
    stem = re.sub(r"_report$", "", stem)
    return stem


def _guess_cycle(path: Path, data: dict[str, Any]) -> str:
    for key in ("cycle", "cycle_id", "cycle_name"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)

    for part in reversed(path.parts):
        if part.startswith("cycle") or re.search(r"\d{8}_\d{6}", part):
            return part
    return ""


def _normalize_explainability_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    complexity = data.get("complexity") if isinstance(data.get("complexity"), dict) else {}
    score_breakdown = data.get("score_breakdown") if isinstance(data.get("score_breakdown"), dict) else {}

    fields = data.get("fields", [])
    operators = data.get("operators", [])
    windows = data.get("windows", [])
    semantic_tags = data.get("semantic_tags", [])

    return {
        "alpha_id": _guess_alpha_id(path, data),
        "cycle": _guess_cycle(path, data),
        "expression": data.get("expression") or data.get("alpha") or "",
        "parse_status": data.get("parse_status") or data.get("explainability_parse_status") or "unknown",
        "interpretability_score": _safe_float(
            data.get("interpretability_score")
            or score_breakdown.get("interpretability_score")
            or data.get("score")
        ),
        "complexity_penalty": _safe_float(data.get("complexity_penalty") or score_breakdown.get("complexity_penalty")),
        "num_nodes": _safe_float(complexity.get("num_nodes") or data.get("num_nodes")),
        "max_depth": _safe_float(complexity.get("max_depth") or data.get("max_depth")),
        "num_fields": _safe_float(complexity.get("num_fields") or data.get("num_fields") or len(_as_list(fields))),
        "num_operators": _safe_float(complexity.get("num_operators") or data.get("num_operators") or len(_as_list(operators))),
        "fields": _join_list(fields),
        "operators": _join_list(operators),
        "windows": _join_list(windows),
        "semantic_tags": _join_list(semantic_tags),
        "source_explainability_file": str(path),
    }


def collect_explainability_records(results_dir: Path) -> list[dict[str, Any]]:
    explainability_dir = results_dir / "explainability"
    if not explainability_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(explainability_dir.rglob("*_explainability.json")):
        data = _read_json(path)
        if not data:
            continue
        records.append(_normalize_explainability_record(path, data))
    return records


def _read_csv_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["source_metrics_file"] = str(path)
                rows.append(row)
    except Exception:
        return []
    return rows


def _read_json_metrics(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not data:
        return []

    rows: list[dict[str, Any]] = []
    if isinstance(data.get("metrics"), list):
        rows = [x for x in data["metrics"] if isinstance(x, dict)]
    elif isinstance(data.get("factors"), list):
        rows = [x for x in data["factors"] if isinstance(x, dict)]
    elif any(k in data for k in ("alpha_id", "factor_id", "expression", "ic_mean")):
        rows = [data]

    for row in rows:
        row["source_metrics_file"] = str(path)
    return rows


def _read_jsonl_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj["source_metrics_file"] = str(path)
                    rows.append(obj)
    except Exception:
        return []
    return rows


def collect_metrics_records(results_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = results_dir / "metrics"
    if not metrics_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows.extend(_read_csv_metrics(path))
        elif suffix == ".json":
            rows.extend(_read_json_metrics(path))
        elif suffix == ".jsonl":
            rows.extend(_read_jsonl_metrics(path))
    return rows


def _metric_key(row: dict[str, Any]) -> tuple[str, str]:
    alpha_id = str(row.get("alpha_id") or row.get("factor_id") or row.get("name") or "").strip()
    expression = str(row.get("expression") or row.get("alpha") or "").strip()
    return alpha_id, expression


def build_metrics_index(metrics: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_alpha_id: dict[str, dict[str, Any]] = {}
    by_expression: dict[str, dict[str, Any]] = {}

    for row in metrics:
        alpha_id, expression = _metric_key(row)
        if alpha_id:
            by_alpha_id[alpha_id] = row
        if expression:
            by_expression[expression] = row
    return by_alpha_id, by_expression


def merge_records(
    explainability_records: list[dict[str, Any]],
    metrics_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_alpha_id, by_expression = build_metrics_index(metrics_records)
    merged: list[dict[str, Any]] = []

    for exp in explainability_records:
        row = dict(exp)
        metrics = by_alpha_id.get(row.get("alpha_id", ""))
        if metrics is None and row.get("expression"):
            metrics = by_expression.get(row["expression"])
        if metrics:
            for key, value in metrics.items():
                if key not in row or row.get(key) in (None, ""):
                    row[key] = value
                elif key in {
                    "ic_mean",
                    "ic_ir",
                    "ict",
                    "icstocknum",
                    "long_excret",
                    "long_sharpe",
                    "long_ir",
                    "long_excmdd",
                    "ls_ret",
                    "ls_std",
                    "ls_sharpe",
                    "ls_mdd",
                    "ic_5_mean",
                    "ic_20_mean",
                }:
                    row[key] = value
            row["source_metrics_file"] = metrics.get("source_metrics_file", "")
        merged.append(row)

    # Also keep metric rows without explainability artifacts, for diagnosing missing reports.
    known_alpha_ids = {x.get("alpha_id") for x in merged if x.get("alpha_id")}
    known_exprs = {x.get("expression") for x in merged if x.get("expression")}
    for metric in metrics_records:
        alpha_id, expression = _metric_key(metric)
        if (alpha_id and alpha_id in known_alpha_ids) or (expression and expression in known_exprs):
            continue
        if not alpha_id and not expression:
            continue
        row = {
            "alpha_id": alpha_id,
            "cycle": "",
            "expression": expression,
            "parse_status": "missing_explainability",
            "interpretability_score": None,
            "complexity_penalty": None,
            "num_nodes": None,
            "max_depth": None,
            "num_fields": None,
            "num_operators": None,
            "fields": "",
            "operators": "",
            "windows": "",
            "semantic_tags": "",
            "source_explainability_file": "",
            "source_metrics_file": metric.get("source_metrics_file", ""),
        }
        row.update(metric)
        merged.append(row)

    return merged


def _all_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen = set(DEFAULT_SUMMARY_COLUMNS)
    extra = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                extra.append(key)
                seen.add(key)
    return DEFAULT_SUMMARY_COLUMNS + extra


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _all_columns(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: Any, digits: int = 4) -> str:
    value = _safe_float(value)
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _counter_from_csv_field(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in _as_list(row.get(field)):
            text = str(item).strip()
            if text:
                counter[text] += 1
    return counter


def make_markdown(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    explainable = [r for r in rows if r.get("parse_status") not in {"missing_explainability", "failed"}]
    parse_success = [r for r in rows if str(r.get("parse_status", "")).lower() == "success"]

    scores = [_safe_float(r.get("interpretability_score")) for r in rows]
    penalties = [_safe_float(r.get("complexity_penalty")) for r in rows]
    depths = [_safe_float(r.get("max_depth")) for r in rows]
    nodes = [_safe_float(r.get("num_nodes")) for r in rows]
    long_sharpes = [_safe_float(r.get("long_sharpe")) for r in rows]
    ic_means = [_safe_float(r.get("ic_mean")) for r in rows]
    ic_5_means = [_safe_float(r.get("ic_5_mean")) for r in rows]
    ic_20_means = [_safe_float(r.get("ic_20_mean")) for r in rows]

    field_counter = _counter_from_csv_field(rows, "fields")
    operator_counter = _counter_from_csv_field(rows, "operators")
    tag_counter = _counter_from_csv_field(rows, "semantic_tags")

    lines = [
        "# 因子解释性统计汇总",
        "",
        "## 1. 总览",
        "",
        f"- 因子记录数：{total}",
        f"- 有解释性记录数：{len(explainable)}",
        f"- AST 解析成功数：{len(parse_success)}",
        f"- AST 解析成功率：{(len(parse_success) / total * 100):.2f}%" if total else "- AST 解析成功率：-",
        f"- 平均 interpretability_score：{_fmt(_mean([x for x in scores if x is not None]))}",
        f"- 平均 complexity_penalty：{_fmt(_mean([x for x in penalties if x is not None]))}",
        f"- 平均 AST 深度：{_fmt(_mean([x for x in depths if x is not None]))}",
        f"- 平均 AST 节点数：{_fmt(_mean([x for x in nodes if x is not None]))}",
        f"- 平均 ic_mean：{_fmt(_mean([x for x in ic_means if x is not None]))}",
        f"- 平均 ic_5_mean：{_fmt(_mean([x for x in ic_5_means if x is not None]))}",
        f"- 平均 ic_20_mean：{_fmt(_mean([x for x in ic_20_means if x is not None]))}",
        f"- 平均 long_sharpe：{_fmt(_mean([x for x in long_sharpes if x is not None]))}",
        "",
        "## 2. 字段使用频次 Top 10",
        "",
        "| 字段 | 次数 |",
        "|---|---:|",
    ]

    for key, count in field_counter.most_common(10):
        lines.append(f"| {key} | {count} |")
    if not field_counter:
        lines.append("| - | - |")

    lines.extend([
        "",
        "## 3. 算子使用频次 Top 10",
        "",
        "| 算子 | 次数 |",
        "|---|---:|",
    ])
    for key, count in operator_counter.most_common(10):
        lines.append(f"| {key} | {count} |")
    if not operator_counter:
        lines.append("| - | - |")

    lines.extend([
        "",
        "## 4. 语义标签频次 Top 10",
        "",
        "| 语义标签 | 次数 |",
        "|---|---:|",
    ])
    for key, count in tag_counter.most_common(10):
        lines.append(f"| {key} | {count} |")
    if not tag_counter:
        lines.append("| - | - |")

    sorted_rows = sorted(
        rows,
        key=lambda r: (_safe_float(r.get("interpretability_score")) or -1.0),
        reverse=True,
    )

    lines.extend([
        "",
        "## 5. 解释性评分 Top 10",
        "",
        "| alpha_id | interpretability_score | complexity_penalty | long_sharpe | expression |",
        "|---|---:|---:|---:|---|",
    ])
    for row in sorted_rows[:10]:
        expr = str(row.get("expression", "")).replace("|", "\\|")
        if len(expr) > 120:
            expr = expr[:117] + "..."
        lines.append(
            f"| {row.get('alpha_id', '')} | {_fmt(row.get('interpretability_score'))} "
            f"| {_fmt(row.get('complexity_penalty'))} | {_fmt(row.get('long_sharpe'))} | `{expr}` |"
        )
    if not sorted_rows:
        lines.append("| - | - | - | - | - |")

    return "\n".join(lines) + "\n"


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_markdown(rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize factor explainability artifacts.")
    parser.add_argument("--results-dir", default="results", help="Project results directory. Default: results")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: results")
    parser.add_argument("--prefix", default="explainability_summary", help="Output file prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    prefix = args.prefix

    explainability_records = collect_explainability_records(results_dir)
    metrics_records = collect_metrics_records(results_dir)
    rows = merge_records(explainability_records, metrics_records)

    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"

    write_csv(rows, csv_path)
    write_json(rows, json_path)
    write_markdown(rows, md_path)

    print(f"explainability records: {len(explainability_records)}")
    print(f"metrics records: {len(metrics_records)}")
    print(f"summary rows: {len(rows)}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
