"""Lightweight AST utilities for formulaic alpha expressions.

This module is intentionally independent from operators.py.  It does not
calculate factor values; it only parses expression strings for explainability.
If parsing fails, it returns an ``unknown`` node instead of interrupting mining.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExprNode:
    node_type: str
    value: str
    children: list["ExprNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
        }


def parse_expression(expression: str) -> ExprNode:
    """Parse an alpha expression into a compact AST.

    The parser accepts Python-like formula DSL expressions such as
    ``Rank(Delta(close, 5)) * Rank(volume)``.  It is best-effort and safe:
    invalid syntax returns an ``unknown`` node.
    """
    expr = str(expression or "").strip()
    if not expr:
        return ExprNode("unknown", "")
    try:
        tree = ast.parse(expr, mode="eval")
        return _from_py_ast(tree.body)
    except Exception:
        return ExprNode("unknown", expr)


def _from_py_ast(node: ast.AST) -> ExprNode:
    if isinstance(node, ast.Call):
        func_name = _name_of(node.func)
        args = [_from_py_ast(arg) for arg in node.args]
        return ExprNode("operator", func_name, args)

    if isinstance(node, ast.Name):
        return ExprNode("field", node.id)

    if isinstance(node, ast.Constant):
        return ExprNode("constant", str(node.value))

    if isinstance(node, ast.UnaryOp):
        op_name = _unary_op_name(node.op)
        return ExprNode("unary_op", op_name, [_from_py_ast(node.operand)])

    if isinstance(node, ast.BinOp):
        op_name = _binary_op_name(node.op)
        return ExprNode("binary_op", op_name, [_from_py_ast(node.left), _from_py_ast(node.right)])

    if isinstance(node, ast.Compare):
        children = [_from_py_ast(node.left)] + [_from_py_ast(c) for c in node.comparators]
        return ExprNode("compare", node.__class__.__name__, children)

    if isinstance(node, ast.Subscript):
        return ExprNode("subscript", _safe_unparse(node), [])

    return ExprNode("unknown", _safe_unparse(node))


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return _safe_unparse(node)


def _binary_op_name(op: ast.AST) -> str:
    return {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Pow: "**",
        ast.Mod: "%",
    }.get(type(op), op.__class__.__name__)


def _unary_op_name(op: ast.AST) -> str:
    return {
        ast.USub: "-",
        ast.UAdd: "+",
        ast.Not: "not",
    }.get(type(op), op.__class__.__name__)


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def extract_fields(root: ExprNode, known_fields: set[str] | None = None) -> list[str]:
    fields: set[str] = set()

    def walk(node: ExprNode):
        if node.node_type == "field":
            if known_fields is None or node.value in known_fields:
                fields.add(node.value)
        for child in node.children:
            walk(child)

    walk(root)
    return sorted(fields)


def extract_operators(root: ExprNode, known_operators: set[str] | None = None) -> list[str]:
    operators: list[str] = []

    def walk(node: ExprNode):
        if node.node_type == "operator":
            if known_operators is None or node.value in known_operators:
                operators.append(node.value)
        if node.node_type in {"binary_op", "unary_op"}:
            operators.append(node.value)
        for child in node.children:
            walk(child)

    walk(root)
    return operators


def extract_windows(root: ExprNode) -> list[int]:
    """Extract likely positive integer window arguments.

    Constants equal to 0/1 are usually flags or scalar offsets, so they are not
    treated as windows here.
    """
    windows: set[int] = set()

    def walk(node: ExprNode):
        if node.node_type == "constant":
            try:
                value = int(float(node.value))
                if value > 1:
                    windows.add(value)
            except Exception:
                pass
        for child in node.children:
            walk(child)

    walk(root)
    return sorted(windows)


def count_ast_nodes(root: ExprNode) -> int:
    return 1 + sum(count_ast_nodes(child) for child in root.children)


def compute_ast_depth(root: ExprNode) -> int:
    if not root.children:
        return 1
    return 1 + max(compute_ast_depth(child) for child in root.children)


def extract_subexpressions(expression: str) -> list[str]:
    """Extract nested function-call subexpressions from the original string."""
    expr = str(expression or "").strip()
    subexprs: list[str] = []
    stack: list[int] = []

    for idx, ch in enumerate(expr):
        if ch == "(":
            start = _find_func_start(expr, idx)
            stack.append(start)
        elif ch == ")" and stack:
            start = stack.pop()
            sub = expr[start : idx + 1].strip()
            if sub and sub not in subexprs:
                subexprs.append(sub)

    if expr and expr not in subexprs:
        subexprs.append(expr)
    return subexprs


def _find_func_start(expr: str, paren_idx: int) -> int:
    i = paren_idx - 1
    while i >= 0 and re.match(r"[A-Za-z0-9_\.]", expr[i]):
        i -= 1
    return i + 1


def summarize_expression(
    expression: str,
    known_fields: set[str] | None = None,
    known_operators: set[str] | None = None,
) -> dict[str, Any]:
    root = parse_expression(expression)
    fields = extract_fields(root, known_fields)
    operators = extract_operators(root, known_operators)
    windows = extract_windows(root)
    return {
        "parse_status": "failed" if root.node_type == "unknown" else "success",
        "ast": root.to_dict(),
        "fields": fields,
        "operators": operators,
        "windows": windows,
        "sub_expressions": extract_subexpressions(expression),
        "complexity": {
            "num_nodes": count_ast_nodes(root),
            "max_depth": compute_ast_depth(root),
            "num_fields": len(fields),
            "num_operators": len(operators),
            "num_windows": len(windows),
        },
    }
