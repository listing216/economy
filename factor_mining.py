"""
factor_mining.py — 因子挖掘通用工具

职责：
1. calculate_factors_performance() 批量计算因子表现指标（IC、分层收益等）
2. is_low_correlated_with_fixed_factors() 检查因子与已有因子的相关性
3. mmr_selection() MMR 多样性筛选（IC 质量 vs 相关性惩罚）

本版本增加解释性字段的可选透传，并允许 MMR 在配置开启时使用
interpretability_score 与 complexity_penalty。未传 explainability_config 时，
原始 MMR 公式保持不变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from correlations import calculate_correlation_numba
from evaluator import Config, FactorPerformanceAnalyzer

_performance_analyzer = None
_last_start_date = None
_last_end_date = None
_last_poolsel_path = None


def get_performance_analyzer(
    start_date: str,
    end_date: str,
    poolsel_path: str | None = None,
) -> FactorPerformanceAnalyzer:
    """获取全局因子表现分析器（相同日期范围则复用）。"""
    global _performance_analyzer, _last_start_date, _last_end_date, _last_poolsel_path

    if (
        _performance_analyzer is None
        or start_date != _last_start_date
        or end_date != _last_end_date
        or poolsel_path != _last_poolsel_path
    ):
        perf_config = Config(
            group_num=10,
            date_start=start_date,
            date_end=end_date,
            adj_dates="10day",
            auto_reverse=True,
            field=None,
            calculate_corr_exposure=False,
            poolsel_path=poolsel_path,
        )
        _performance_analyzer = FactorPerformanceAnalyzer(perf_config)
        _last_start_date = start_date
        _last_end_date = end_date
        _last_poolsel_path = poolsel_path

    return _performance_analyzer


def calculate_factors_performance(
    fac_dict: dict,
    expression_list: list,
    close: pd.DataFrame,
    start_date: str = "20000101",
    end_date: str = "20150101",
    explanation_list: list | None = None,
    poolsel_path: str | None = None,
    explainability_list: list[dict] | None = None,
) -> list:
    """
    批量计算因子表现指标。

    Args:
        explainability_list: 可选解释性结果列表。传入时会把解释性字段写入
            返回记录；不传时保持原逻辑。

    Returns:
        list[dict]: 每个因子的表现指标字典列表。
    """

    def empty_record() -> dict:
        return {
            "ic_mean": np.nan,
            "ic_ir": np.nan,
            "ict": np.nan,
            "icstocknum": np.nan,
            "long_excret": np.nan,
            "long_sharpe": np.nan,
            "long_ir": np.nan,
            "long_excmdd": np.nan,
            "ls_ret": np.nan,
            "ls_std": np.nan,
            "ls_sharpe": np.nan,
            "ls_mdd": np.nan,
        }

    def attach_meta(perf: dict, idx: int):
        if explanation_list:
            perf["explanation"] = explanation_list[idx]
        if explainability_list and idx < len(explainability_list):
            exp = explainability_list[idx] or {}
            perf["explainability"] = exp
            perf["interpretability_score"] = exp.get("interpretability_score", np.nan)
            perf["complexity_penalty"] = exp.get("complexity_penalty", np.nan)
            perf["semantic_tags"] = ",".join(exp.get("semantic_tags", []))
            perf["explainability_parse_status"] = exp.get("parse_status", "unknown")

    try:
        performance_analyzer = get_performance_analyzer(start_date, end_date, poolsel_path)
        performance_result = performance_analyzer.analyze(fac_dict, close, factor_lag=True)
        stat_output = performance_result["stat_output"]

        results = []
        for idx, factor_name in enumerate(fac_dict.keys()):
            perf = empty_record()
            perf["alpha_id"] = factor_name
            perf["expression"] = expression_list[idx]
            attach_meta(perf, idx)

            if factor_name in stat_output.index:
                row = stat_output.loc[factor_name]
                for key in [
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
                ]:
                    perf[key] = row.get(key, np.nan)
            results.append(perf)

        return results

    except Exception as e:
        print(f"批量计算因子表现失败：{e}")
        results = []
        for idx, factor_name in enumerate(fac_dict.keys()):
            perf = empty_record()
            perf["alpha_id"] = factor_name
            perf["expression"] = expression_list[idx]
            attach_meta(perf, idx)
            results.append(perf)
        return results


def is_low_correlated_with_fixed_factors(
    factor_values: pd.DataFrame,
    fixed_factors_dict: dict,
    threshold: float,
) -> bool:
    """
    检查 factor_values 与 fixed_factors_dict 中所有因子的截面相关性是否均低于 threshold。

    Returns:
        True -> 相关性均低于阈值
        False -> 与某因子相关性过高
    """
    for _, fixed_df in fixed_factors_dict.items():
        corr = calculate_correlation_numba(factor_values, fixed_df, type="cs")
        if abs(corr) > threshold:
            return False
    return True


def mmr_selection(
    factors_df: pd.DataFrame,
    candidate_factor_values: dict,
    baseline_factor_names: list,
    correlation_matrix: pd.DataFrame,
    num_to_select: int,
    lambda_param: float,
    threshold: float,
    explainability_config: dict | None = None,
) -> tuple[list, dict]:
    """
    MMR 筛选算法：在 IC 质量与多样性之间取得平衡。

    当 explainability_config.enabled=true 且配置 mmr.use_interpretability_score=true 时，
    额外把 interpretability_score 和 complexity_penalty 纳入打分；否则保持原始公式。

    Returns:
        (selected_names, selected_values)
    """
    print(f"开始MMR选择, 目标 {num_to_select} 个因子, lambda={lambda_param}, 阈值={threshold}")
    print(f"baseline因子数: {len(baseline_factor_names)}, 候选因子数: {len(candidate_factor_values)}")

    exp_cfg = explainability_config or {}
    mmr_cfg = exp_cfg.get("mmr", {}) if isinstance(exp_cfg, dict) else {}
    use_explainability = (
        bool(exp_cfg.get("enabled", False))
        and bool(mmr_cfg.get("use_interpretability_score", False))
        and "interpretability_score" in factors_df.columns
    )

    if use_explainability:
        print(
            "MMR启用解释性评分: "
            f"quality={mmr_cfg.get('quality_weight', 0.70)}, "
            f"explainability={mmr_cfg.get('explainability_weight', 0.15)}, "
            f"correlation={mmr_cfg.get('correlation_weight', 0.10)}, "
            f"complexity={mmr_cfg.get('complexity_weight', 0.05)}"
        )

    candidate_names = list(candidate_factor_values.keys())
    candidate_performances = {}
    candidate_explainability = {}
    candidate_complexity = {}

    for _, row in factors_df.iterrows():
        alpha_id = row["alpha_id"]
        if alpha_id in candidate_factor_values:
            candidate_performances[alpha_id] = abs(row.get("ic_mean", 0) or 0)
            candidate_explainability[alpha_id] = _safe_float(row.get("interpretability_score", 0.0), 0.0)
            candidate_complexity[alpha_id] = _safe_float(row.get("complexity_penalty", 0.0), 0.0)

    if use_explainability and mmr_cfg.get("filter_by_interpretability", False):
        min_score = float(mmr_cfg.get("min_interpretability_score", 0.0))
        candidate_names = [
            name for name in candidate_names
            if candidate_explainability.get(name, 0.0) >= min_score
        ]
        print(f"解释性硬过滤后候选因子数: {len(candidate_names)}")

    selected_factors = []
    remaining_factors = candidate_names.copy()

    for _ in range(num_to_select):
        if not remaining_factors:
            print("没有剩余候选因子，提前结束")
            break

        best_factor = None
        best_score = -float("inf")
        best_quality = 0.0
        best_max_corr = 0.0
        best_interpretability = 0.0
        best_correlations = []

        for factor in remaining_factors:
            quality_score = candidate_performances.get(factor, 0.0)
            max_correlation = 0.0
            correlations = []

            for selected in selected_factors:
                if factor in correlation_matrix.index and selected in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[factor, selected])
                    correlations.append((selected, corr))
                    max_correlation = max(max_correlation, corr)

            for base in baseline_factor_names:
                if factor in correlation_matrix.index and base in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[factor, base])
                    correlations.append((base, corr))
                    max_correlation = max(max_correlation, corr)

            if max_correlation > threshold:
                continue

            if use_explainability:
                interpretability_score = candidate_explainability.get(factor, 0.0)
                complexity_penalty = candidate_complexity.get(factor, 0.0)
                mmr_score = (
                    mmr_cfg.get("quality_weight", 0.70) * quality_score
                    + mmr_cfg.get("explainability_weight", 0.15) * interpretability_score
                    - mmr_cfg.get("correlation_weight", 0.10) * max_correlation
                    - mmr_cfg.get("complexity_weight", 0.05) * complexity_penalty
                )
            else:
                interpretability_score = 0.0
                mmr_score = lambda_param * quality_score - (1 - lambda_param) * max_correlation

            if mmr_score > best_score:
                best_score = mmr_score
                best_factor = factor
                best_quality = quality_score
                best_max_corr = max_correlation
                best_interpretability = interpretability_score
                best_correlations = correlations

        if best_factor is None:
            print("所有剩余候选因子相关性过高，提前结束")
            break

        selected_factors.append(best_factor)
        remaining_factors.remove(best_factor)

        msg = (
            f"[{len(selected_factors)}/{num_to_select}] 选中: {best_factor}, "
            f"IC={best_quality:.4f}, MMR={best_score:.4f}, 最大相关性={best_max_corr:.4f}"
        )
        if use_explainability:
            msg += f", 解释性={best_interpretability:.4f}"
        print(msg)

        high_corr = [f"{name}:{corr:.4f}" for name, corr in best_correlations if corr > 0.3]
        if high_corr:
            print(f" 高相关因子: {', '.join(high_corr[:5])}")

    selected_values = {factor: candidate_factor_values[factor] for factor in selected_factors}
    print(f"共选择了 {len(selected_factors)}/{num_to_select} 个因子")
    return selected_factors, selected_values


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default
