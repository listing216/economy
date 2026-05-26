#!/usr/bin/env python3
"""生成模拟 MCTS 搜索树数据 & 输出交互式 HTML 可视化报告。"""

import sys
from pathlib import Path

# 确保能找到 tree_viz.py
sys.path.insert(0, str(Path(__file__).parent))

from tree_viz import SearchTreeVisualizer


def make_node(node_id, depth, expression, reward, target_dimension,
              refinement_suggestion, dimension_scores, metrics,
              parent_id, children):
    return {
        "node_id": node_id,
        "depth": depth,
        "reward": reward,
        "q_value": reward * 0.95,  # Q 略低于 Reward
        "visits": max(3, 12 - depth * 2),
        "target_dimension": target_dimension,
        "refinement_suggestion": refinement_suggestion,
        "candidate": {
            "alpha_id": f"alpha_{node_id}",
            "expression": expression,
            "explanation": f"通过改进 {target_dimension} 维度生成的因子" if target_dimension else "初始种子因子",
            "metrics": metrics,
            "extra": {},
        },
        "dimension_scores": dimension_scores,
        "parent_id": parent_id,
        "children": children,
    }


def build_tree():
    """构建模拟的 MCTS 搜索树（约 24 节点，含语义等价节点）。"""

    # ═══════════════════════════════════════════════════════
    #  定义所有节点数据
    # ═══════════════════════════════════════════════════════

    nodes = {}

    # ── 根节点 ──
    nodes["root"] = dict(
        expression="Rank(Delta(close, 5))",
        reward=0.72,
        target_dimension="",
        suggestion="初始种子因子：价格变化率的排名",
        dim_scores=dict(effectiveness=0.65, diversity=0.40, stability=0.55, turnover=0.70, overfit=0.60),
        metrics=dict(ic_mean=0.032, ic_ir=0.58, sharpe=0.92),
    )

    # ── Level 1 ──
    nodes["A"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10))",
        reward=0.85,
        target_dimension="effectiveness",
        suggestion="加入波动率过滤，提高 IC 稳定性",
        dim_scores=dict(effectiveness=0.88, diversity=0.42, stability=0.72, turnover=0.65, overfit=0.52),
        metrics=dict(ic_mean=0.045, ic_ir=0.70, sharpe=1.12),
    )
    nodes["B"] = dict(
        expression="Rank(Delta(close, 5)) + Rank(Volume, 5)",
        reward=0.78,
        target_dimension="diversity",
        suggestion="加入成交量维度，增加因子多样性",
        dim_scores=dict(effectiveness=0.70, diversity=0.75, stability=0.50, turnover=0.60, overfit=0.55),
        metrics=dict(ic_mean=0.038, ic_ir=0.62, sharpe=0.98),
    )
    nodes["C"] = dict(
        expression="Rank(Delta(close, 5)) * Ts_Argmin(Close, 20)",
        reward=0.80,
        target_dimension="stability",
        suggestion="加入长期价格位置指标，提高稳定性",
        dim_scores=dict(effectiveness=0.75, diversity=0.50, stability=0.78, turnover=0.62, overfit=0.48),
        metrics=dict(ic_mean=0.041, ic_ir=0.65, sharpe=1.05),
    )

    # ── Level 2 (from A) ──
    nodes["A1"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5)",
        reward=0.90,
        target_dimension="stability",
        suggestion="加入成交量排名，过滤低流动性时段",
        dim_scores=dict(effectiveness=0.85, diversity=0.55, stability=0.85, turnover=0.58, overfit=0.50),
        metrics=dict(ic_mean=0.050, ic_ir=0.78, sharpe=1.20),
    )
    nodes["A2"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) - Rank(Delta(Volume, 5))",
        reward=0.82,
        target_dimension="turnover",
        suggestion="减去成交量变化，降低换手率",
        dim_scores=dict(effectiveness=0.82, diversity=0.48, stability=0.70, turnover=0.75, overfit=0.55),
        metrics=dict(ic_mean=0.044, ic_ir=0.68, sharpe=1.08),
    )
    nodes["A3"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) + Rank(Close, 20)",
        reward=0.87,
        target_dimension="overfit",
        suggestion="加入收盘价位置排名，降低过拟合风险",
        dim_scores=dict(effectiveness=0.86, diversity=0.52, stability=0.74, turnover=0.62, overfit=0.70),
        metrics=dict(ic_mean=0.047, ic_ir=0.72, sharpe=1.15),
    )

    # ── Level 2 (from B) ──
    # B1 是 B 的语义等价（Add 交换律）
    nodes["B1"] = dict(
        expression="Rank(Volume, 5) + Rank(Delta(close, 5))",
        reward=0.80,
        target_dimension="stability",
        suggestion="尝试交换操作数顺序以改善稳定性",
        dim_scores=dict(effectiveness=0.72, diversity=0.72, stability=0.58, turnover=0.60, overfit=0.53),
        metrics=dict(ic_mean=0.039, ic_ir=0.63, sharpe=0.99),
    )
    nodes["B2"] = dict(
        expression="Rank(Volume, 5) * Rank(Delta(close, 5)) / Mean(Volume, 10)",
        reward=0.88,
        target_dimension="effectiveness",
        suggestion="加入成交量均值归一化，增强有效性",
        dim_scores=dict(effectiveness=0.90, diversity=0.65, stability=0.68, turnover=0.55, overfit=0.58),
        metrics=dict(ic_mean=0.048, ic_ir=0.75, sharpe=1.18),
    )

    # ── Level 2 (from C) ──
    nodes["C1"] = dict(
        expression="Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) - Rank(Std(Volume, 5))",
        reward=0.76,
        target_dimension="diversity",
        suggestion="加入成交量波动率维度，增加多样性",
        dim_scores=dict(effectiveness=0.68, diversity=0.78, stability=0.72, turnover=0.60, overfit=0.45),
        metrics=dict(ic_mean=0.036, ic_ir=0.58, sharpe=0.95),
    )
    nodes["C2"] = dict(
        expression="Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) + Rank(Delta(Volume, 10))",
        reward=0.84,
        target_dimension="turnover",
        suggestion="加入成交量变化率，改善换手率特征",
        dim_scores=dict(effectiveness=0.76, diversity=0.55, stability=0.74, turnover=0.78, overfit=0.52),
        metrics=dict(ic_mean=0.043, ic_ir=0.67, sharpe=1.10),
    )

    # ── Level 3 ──
    nodes["A1a"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) + Rank(Std(Volume, 10))",
        reward=0.92,
        target_dimension="diversity",
        suggestion="加入成交量波动率排名，提升因子多样性",
        dim_scores=dict(effectiveness=0.88, diversity=0.80, stability=0.82, turnover=0.52, overfit=0.48),
        metrics=dict(ic_mean=0.055, ic_ir=0.82, sharpe=1.25),
    )
    nodes["A1b"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) * Ts_Argmin(Close, 10)",
        reward=0.88,
        target_dimension="overfit",
        suggestion="加入短期价格位置，控制过拟合",
        dim_scores=dict(effectiveness=0.80, diversity=0.58, stability=0.78, turnover=0.50, overfit=0.72),
        metrics=dict(ic_mean=0.048, ic_ir=0.74, sharpe=1.16),
    )
    nodes["A2a"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) - Rank(Delta(Volume, 5)) * Rank(Std(Volume, 10))",
        reward=0.85,
        target_dimension="stability",
        suggestion="加入成交量波动稳定项，提高稳定性",
        dim_scores=dict(effectiveness=0.78, diversity=0.55, stability=0.82, turnover=0.70, overfit=0.60),
        metrics=dict(ic_mean=0.045, ic_ir=0.69, sharpe=1.11),
    )
    # A3a 是 A3 的语义等价（Add 交换律）
    nodes["A3a"] = dict(
        expression="Rank(Close, 20) + Rank(Delta(close, 5)) * Rank(Std(close, 10))",
        reward=0.89,
        target_dimension="effectiveness",
        suggestion="调整操作顺序，尝试提高有效性",
        dim_scores=dict(effectiveness=0.89, diversity=0.50, stability=0.72, turnover=0.60, overfit=0.68),
        metrics=dict(ic_mean=0.049, ic_ir=0.73, sharpe=1.17),
    )
    nodes["B2a"] = dict(
        expression="Rank(Volume, 5) * Rank(Delta(close, 5)) * Rank(Std(close, 10)) / Mean(Volume, 10)",
        reward=0.91,
        target_dimension="overfit",
        suggestion="加入波动率排名，降低过拟合",
        dim_scores=dict(effectiveness=0.86, diversity=0.68, stability=0.72, turnover=0.50, overfit=0.74),
        metrics=dict(ic_mean=0.052, ic_ir=0.79, sharpe=1.22),
    )
    nodes["B2b"] = dict(
        expression="Rank(Volume, 5) * Rank(Delta(close, 5)) / Mean(Volume, 10) + Mean(Volume, 5)",
        reward=0.86,
        target_dimension="stability",
        suggestion="加入成交量均值，提高稳定性",
        dim_scores=dict(effectiveness=0.84, diversity=0.62, stability=0.80, turnover=0.52, overfit=0.60),
        metrics=dict(ic_mean=0.046, ic_ir=0.71, sharpe=1.13),
    )
    nodes["C2a"] = dict(
        expression="Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) + Rank(Delta(Volume, 10)) - Rank(Std(Volume, 5))",
        reward=0.87,
        target_dimension="diversity",
        suggestion="综合多个成交量维度，提升多样性",
        dim_scores=dict(effectiveness=0.74, diversity=0.85, stability=0.70, turnover=0.72, overfit=0.50),
        metrics=dict(ic_mean=0.047, ic_ir=0.70, sharpe=1.14),
    )
    nodes["C2b"] = dict(
        expression="Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) + Rank(Delta(Volume, 10)) * Rank(Close, 10)",
        reward=0.83,
        target_dimension="effectiveness",
        suggestion="加入收盘价排名，增强有效性",
        dim_scores=dict(effectiveness=0.85, diversity=0.58, stability=0.68, turnover=0.70, overfit=0.55),
        metrics=dict(ic_mean=0.042, ic_ir=0.66, sharpe=1.09),
    )

    # ── Level 4（叶节点） ──
    nodes["A1a1"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) + Rank(Std(Volume, 10)) / Mean(Volume, 5)",
        reward=0.90,
        target_dimension="turnover",
        suggestion="加入成交量均值归一化，优化换手率",
        dim_scores=dict(effectiveness=0.86, diversity=0.78, stability=0.78, turnover=0.80, overfit=0.46),
        metrics=dict(ic_mean=0.051, ic_ir=0.76, sharpe=1.21),
    )
    nodes["A1b1"] = dict(
        expression="Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) * Ts_Argmin(Close, 10) + Rank(Std(Volume, 5))",
        reward=0.91,
        target_dimension="diversity",
        suggestion="加入成交量波动，进一步提升多样性",
        dim_scores=dict(effectiveness=0.82, diversity=0.88, stability=0.74, turnover=0.48, overfit=0.68),
        metrics=dict(ic_mean=0.053, ic_ir=0.80, sharpe=1.23),
    )
    nodes["B2a1"] = dict(
        expression="Rank(Std(close, 10)) * Rank(Volume, 5) * Rank(Delta(close, 5)) / Mean(Volume, 10)",
        reward=0.93,
        target_dimension="effectiveness",
        suggestion="重新排序操作数以优化有效性",
        dim_scores=dict(effectiveness=0.92, diversity=0.65, stability=0.70, turnover=0.48, overfit=0.72),
        metrics=dict(ic_mean=0.056, ic_ir=0.85, sharpe=1.28),
    )

    # ═══════════════════════════════════════════════════════
    #  构建树结构
    # ═══════════════════════════════════════════════════════

    def n(name, depth, parent_id, children_names=None):
        data = nodes[name]
        child_list = []
        if children_names:
            for cn in children_names:
                child_list.append(n(cn, depth + 1, name, None))
        return make_node(
            node_id=name,
            depth=depth,
            expression=data["expression"],
            reward=data["reward"],
            target_dimension=data["target_dimension"],
            refinement_suggestion=data["suggestion"],
            dimension_scores=data["dim_scores"],
            metrics=data["metrics"],
            parent_id=parent_id,
            children=child_list,
        )

def build_tree_fixed():
    """构建模拟的 MCTS 搜索树（含语义等价节点）。"""

    def n(name, depth, expression, reward, target_dim, suggestion,
          dim_scores, metrics, parent_id, children):
        return make_node(
            node_id=name,
            depth=depth,
            expression=expression,
            reward=reward,
            target_dimension=target_dim,
            refinement_suggestion=suggestion,
            dimension_scores=dim_scores,
            metrics=metrics,
            parent_id=parent_id,
            children=children,
        )

    # ── Level 4 ──
    A1a1 = n("A1a1", 4,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) + Rank(Std(Volume, 10)) / Mean(Volume, 5)",
        0.90, "turnover", "加入成交量均值归一化，优化换手率",
        dict(effectiveness=0.86, diversity=0.78, stability=0.78, turnover=0.80, overfit=0.46),
        dict(ic_mean=0.051, ic_ir=0.76, sharpe=1.21),
        "A1a", [])

    A1b1 = n("A1b1", 4,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) * Ts_Argmin(Close, 10) + Rank(Std(Volume, 5))",
        0.91, "diversity", "加入成交量波动，进一步提升多样性",
        dict(effectiveness=0.82, diversity=0.88, stability=0.74, turnover=0.48, overfit=0.68),
        dict(ic_mean=0.053, ic_ir=0.80, sharpe=1.23),
        "A1b", [])

    A2a1 = n("A2a1", 4,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) - Rank(Delta(Volume, 5)) * Rank(Std(Volume, 10)) / Mean(Volume, 10)",
        0.88, "effectiveness", "加入成交量均值归一化，提升有效性",
        dict(effectiveness=0.84, diversity=0.58, stability=0.78, turnover=0.68, overfit=0.58),
        dict(ic_mean=0.047, ic_ir=0.72, sharpe=1.14),
        "A2a", [])

    # B2a1 是 B2a 的语义等价（Mult 交换律）
    B2a1 = n("B2a1", 4,
        "Rank(Std(close, 10)) * Rank(Volume, 5) * Rank(Delta(close, 5)) / Mean(Volume, 10)",
        0.93, "effectiveness", "重新排序操作数以优化有效性",
        dict(effectiveness=0.92, diversity=0.65, stability=0.70, turnover=0.48, overfit=0.72),
        dict(ic_mean=0.056, ic_ir=0.85, sharpe=1.28),
        "B2a", [])

    # ── Level 3 ──
    A1a = n("A1a", 3,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) + Rank(Std(Volume, 10))",
        0.92, "diversity", "加入成交量波动率排名，提升因子多样性",
        dict(effectiveness=0.88, diversity=0.80, stability=0.82, turnover=0.52, overfit=0.48),
        dict(ic_mean=0.055, ic_ir=0.82, sharpe=1.25),
        "A1", [A1a1])

    A1b = n("A1b", 3,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5) * Ts_Argmin(Close, 10)",
        0.88, "overfit", "加入短期价格位置，控制过拟合",
        dict(effectiveness=0.80, diversity=0.58, stability=0.78, turnover=0.50, overfit=0.72),
        dict(ic_mean=0.048, ic_ir=0.74, sharpe=1.16),
        "A1", [A1b1])

    A2a = n("A2a", 3,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) - Rank(Delta(Volume, 5)) * Rank(Std(Volume, 10))",
        0.85, "stability", "加入成交量波动稳定项，提高稳定性",
        dict(effectiveness=0.78, diversity=0.55, stability=0.82, turnover=0.70, overfit=0.60),
        dict(ic_mean=0.045, ic_ir=0.69, sharpe=1.11),
        "A2", [A2a1])

    # A3a 是 A3 的语义等价（Add 交换律）
    A3a = n("A3a", 3,
        "Rank(Close, 20) + Rank(Delta(close, 5)) * Rank(Std(close, 10))",
        0.89, "effectiveness", "调整操作顺序，尝试提高有效性",
        dict(effectiveness=0.89, diversity=0.50, stability=0.72, turnover=0.60, overfit=0.68),
        dict(ic_mean=0.049, ic_ir=0.73, sharpe=1.17),
        "A3", [])

    B2a = n("B2a", 3,
        "Rank(Volume, 5) * Rank(Delta(close, 5)) * Rank(Std(close, 10)) / Mean(Volume, 10)",
        0.91, "overfit", "加入波动率排名，降低过拟合",
        dict(effectiveness=0.86, diversity=0.68, stability=0.72, turnover=0.50, overfit=0.74),
        dict(ic_mean=0.052, ic_ir=0.79, sharpe=1.22),
        "B2", [B2a1])

    B2b = n("B2b", 3,
        "Rank(Volume, 5) * Rank(Delta(close, 5)) / Mean(Volume, 10) + Mean(Volume, 5)",
        0.86, "stability", "加入成交量均值，提高稳定性",
        dict(effectiveness=0.84, diversity=0.62, stability=0.80, turnover=0.52, overfit=0.60),
        dict(ic_mean=0.046, ic_ir=0.71, sharpe=1.13),
        "B2", [])

    C2a = n("C2a", 3,
        "Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) + Rank(Delta(Volume, 10)) - Rank(Std(Volume, 5))",
        0.87, "diversity", "综合多个成交量维度，提升多样性",
        dict(effectiveness=0.74, diversity=0.85, stability=0.70, turnover=0.72, overfit=0.50),
        dict(ic_mean=0.047, ic_ir=0.70, sharpe=1.14),
        "C2", [])

    C2b = n("C2b", 3,
        "Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) + Rank(Delta(Volume, 10)) * Rank(Close, 10)",
        0.83, "effectiveness", "加入收盘价排名，增强有效性",
        dim_scores=dict(effectiveness=0.85, diversity=0.58, stability=0.68, turnover=0.70, overfit=0.55),
        metrics=dict(ic_mean=0.042, ic_ir=0.66, sharpe=1.09),
        parent_id="C2", children=[])

    # ── Level 2 ──
    A1 = n("A1", 2,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) * Rank(Volume, 5)",
        0.90, "stability", "加入成交量排名，过滤低流动性时段",
        dict(effectiveness=0.85, diversity=0.55, stability=0.85, turnover=0.58, overfit=0.50),
        dict(ic_mean=0.050, ic_ir=0.78, sharpe=1.20),
        "A", [A1a, A1b])

    A2 = n("A2", 2,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) - Rank(Delta(Volume, 5))",
        0.82, "turnover", "减去成交量变化，降低换手率",
        dict(effectiveness=0.82, diversity=0.48, stability=0.70, turnover=0.75, overfit=0.55),
        dict(ic_mean=0.044, ic_ir=0.68, sharpe=1.08),
        "A", [A2a])

    A3 = n("A3", 2,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10)) + Rank(Close, 20)",
        0.87, "overfit", "加入收盘价位置排名，降低过拟合风险",
        dict(effectiveness=0.86, diversity=0.52, stability=0.74, turnover=0.62, overfit=0.70),
        dict(ic_mean=0.047, ic_ir=0.72, sharpe=1.15),
        "A", [A3a])

    # B1 是 B 的语义等价（Add 交换律）
    B1 = n("B1", 2,
        "Rank(Volume, 5) + Rank(Delta(close, 5))",
        0.80, "stability", "尝试交换操作数顺序以改善稳定性",
        dict(effectiveness=0.72, diversity=0.72, stability=0.58, turnover=0.60, overfit=0.53),
        dict(ic_mean=0.039, ic_ir=0.63, sharpe=0.99),
        "B", [])

    B2 = n("B2", 2,
        "Rank(Volume, 5) * Rank(Delta(close, 5)) / Mean(Volume, 10)",
        0.88, "effectiveness", "加入成交量均值归一化，增强有效性",
        dict(effectiveness=0.90, diversity=0.65, stability=0.68, turnover=0.55, overfit=0.58),
        dict(ic_mean=0.048, ic_ir=0.75, sharpe=1.18),
        "B", [B2a, B2b])

    C1 = n("C1", 2,
        "Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) - Rank(Std(Volume, 5))",
        0.76, "diversity", "加入成交量波动率维度，增加多样性",
        dict(effectiveness=0.68, diversity=0.78, stability=0.72, turnover=0.60, overfit=0.45),
        dict(ic_mean=0.036, ic_ir=0.58, sharpe=0.95),
        "C", [])

    C2 = n("C2", 2,
        "Rank(Delta(close, 5)) * Ts_Argmin(Close, 20) + Rank(Delta(Volume, 10))",
        0.84, "turnover", "加入成交量变化率，改善换手率特征",
        dict(effectiveness=0.76, diversity=0.55, stability=0.74, turnover=0.78, overfit=0.52),
        dict(ic_mean=0.043, ic_ir=0.67, sharpe=1.10),
        "C", [C2a, C2b])

    # ── Level 1 ──
    A = n("A", 1,
        "Rank(Delta(close, 5)) * Rank(Std(close, 10))",
        0.85, "effectiveness", "加入波动率过滤，提高 IC 稳定性",
        dict(effectiveness=0.88, diversity=0.42, stability=0.72, turnover=0.65, overfit=0.52),
        dict(ic_mean=0.045, ic_ir=0.70, sharpe=1.12),
        "root", [A1, A2, A3])

    B = n("B", 1,
        "Rank(Delta(close, 5)) + Rank(Volume, 5)",
        0.78, "diversity", "加入成交量维度，增加因子多样性",
        dict(effectiveness=0.70, diversity=0.75, stability=0.50, turnover=0.60, overfit=0.55),
        dict(ic_mean=0.038, ic_ir=0.62, sharpe=0.98),
        "root", [B1, B2])

    C = n("C", 1,
        "Rank(Delta(close, 5)) * Ts_Argmin(Close, 20)",
        0.80, "stability", "加入长期价格位置指标，提高稳定性",
        dict(effectiveness=0.75, diversity=0.50, stability=0.78, turnover=0.62, overfit=0.48),
        dict(ic_mean=0.041, ic_ir=0.65, sharpe=1.05),
        "root", [C1, C2])

    # ── Root ──
    root = n("root", 0,
        "Rank(Delta(close, 5))",
        0.72, "", "初始种子因子：价格变化率的排名",
        dict(effectiveness=0.65, diversity=0.40, stability=0.55, turnover=0.70, overfit=0.60),
        dict(ic_mean=0.032, ic_ir=0.58, sharpe=0.92),
        None, [A, B, C])

    return root


if __name__ == "__main__":
    tree_dict = build_tree_fixed()
    print(f"构建模拟树完成，根节点: {tree_dict['node_id']}")

    viz = SearchTreeVisualizer(tree_dict)
    stats = viz.get_stats()
    print(f"统计信息:")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  最大深度: {stats['max_depth']}")
    print(f"  语义等价组: {stats['equivalent_groups']} 组 ({stats['equivalent_nodes']} 节点)")
    print(f"  平均 Reward: {stats['avg_reward']:.4f}")
    print(f"  最佳 Reward: {stats['best_reward']:.4f}")

    output_dir = Path("viz_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    out = viz.save_html(str(output_dir / "mcts_demo.html"))
    print(f"\n静态 HTML 报告已保存: {out}")

    anim_out = viz.save_animated_html(
        str(output_dir / "mcts_demo_animated.html"),
        frame_duration=600,
    )
    print(f"动态 HTML 报告（树逐步生长动画）: {anim_out}")

    json_path = viz.save_json(str(output_dir / "mcts_demo.json"))
    print(f"JSON 数据已保存: {json_path}")
    print(f"\n打开 {anim_out} 点击 ▶ Play 查看搜索树演化动画")
