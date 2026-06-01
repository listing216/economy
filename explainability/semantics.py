"""Semantic dictionaries for fields and operators."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FIELD_SEMANTICS = BASE_DIR / "semantics" / "field_semantics.yaml"
DEFAULT_OPERATOR_SEMANTICS = BASE_DIR / "semantics" / "operator_semantics.yaml"


def load_yaml_dict(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_field_semantics(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml_dict(path or DEFAULT_FIELD_SEMANTICS)


def load_operator_semantics(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml_dict(path or DEFAULT_OPERATOR_SEMANTICS)


def get_field_semantic(field: str, semantics: dict[str, Any]) -> dict[str, Any]:
    item = semantics.get(field, {}) or {}
    return {
        "field": field,
        "type": item.get("type", "unknown"),
        "name_cn": item.get("name_cn", field),
        "meaning": item.get("meaning", "未配置字段语义"),
        "role": item.get("role", "unknown"),
    }


def get_operator_semantic(operator: str, semantics: dict[str, Any]) -> dict[str, Any]:
    item = semantics.get(operator, {}) or {}
    return {
        "operator": operator,
        "category": item.get("category", "unknown"),
        "meaning": item.get("meaning", "未配置算子语义"),
        "financial_interpretation": item.get("financial_interpretation", "未配置金融解释"),
        "risk": item.get("risk", "未配置风险提示"),
    }
