"""
metrics_hooks.py - 可插拔的额外指标计算区

当前实现仅保留普通 forward IC 指标，不引入 YoY 加权或额外的
duration 评价逻辑。当前内置 hook:

- plain_ic_metrics
  默认计算 +5 天和 +20 天的普通 IC 均值，输出:
  - ic_5_mean
  - ic_20_mean

验证配置可以通过 YAML 控制:
- metric_hooks: 启用哪些 hook
- metric_hook_params: 给 hook 传什么参数
- train_filter / validation_filter: 如何筛选
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluator import Config, FactorPerformanceAnalyzer


METRIC_HOOKS: dict[str, callable] = {}
_ANALYZER_CACHE: dict[tuple[str, str, int, str | None], FactorPerformanceAnalyzer] = {}


def register_hook(name: str):
    """注册新的指标 hook。"""

    def decorator(fn):
        METRIC_HOOKS[name] = fn
        return fn

    return decorator


def load_data_deps(data_deps_config: dict) -> dict:
    """
    预留给更复杂的 profile 使用。
    普通 IC 计算不依赖额外外部数据，因此这里只返回解析后的路径字典。
    """
    if not data_deps_config:
        return {}
    return dict(data_deps_config)


def compute_extra_metrics(
    factor_df: pd.DataFrame,
    hook_names: list,
    loaded_deps: dict,
    start_date: str,
    end_date: str,
    hook_params: dict | None = None,
) -> dict:
    """按配置调用额外指标 hook，并合并结果。"""
    if not hook_names:
        return {}

    result = {}
    hook_params = hook_params or {}

    for name in hook_names:
        if name not in METRIC_HOOKS:
            raise ValueError(
                f"未知 metric hook: '{name}'，已注册: {list(METRIC_HOOKS.keys())}"
            )
        params = hook_params.get(name) or {}
        metrics = METRIC_HOOKS[name](factor_df, loaded_deps, start_date, end_date, params)
        result.update(metrics)

    return result


def passes_filter(metrics: dict, filter_conditions: list) -> tuple[bool, str]:
    """
    检查指标字典是否通过全部筛选条件。

    direction:
    - gt: value > threshold
    - lt: value < threshold
    - abs_gt: abs(value) > threshold
    - abs_lt: abs(value) < threshold
    """
    for cond in filter_conditions:
        metric = cond["metric"]
        threshold = cond["threshold"]
        direction = cond.get("direction", "gt")

        val = metrics.get(metric)
        if val is None or pd.isna(val):
            return False, f"指标 {metric} 缺失或为 NaN"

        if direction == "gt":
            ok = val > threshold
        elif direction == "lt":
            ok = val < threshold
        elif direction == "abs_gt":
            ok = abs(val) > threshold
        elif direction == "abs_lt":
            ok = abs(val) < threshold
        else:
            raise ValueError(f"未知 direction: '{direction}'，支持 gt/lt/abs_gt/abs_lt")

        if not ok:
            return False, f"{metric}={val:.4f} 未满足 {direction} {threshold}"

    return True, "通过所有条件"


def _get_analyzer(
    start_date: str,
    end_date: str,
    horizon: int,
    poolsel_path: str | None,
) -> FactorPerformanceAnalyzer:
    key = (start_date, end_date, int(horizon), poolsel_path)
    if key not in _ANALYZER_CACHE:
        config = Config(
            group_num=10,
            date_start=start_date,
            date_end=end_date,
            adj_dates=f"{int(horizon)}day",
            auto_reverse=True,
            field=None,
            calculate_corr_exposure=False,
            poolsel_path=poolsel_path,
        )
        _ANALYZER_CACHE[key] = FactorPerformanceAnalyzer(config)
    return _ANALYZER_CACHE[key]


def _compute_single_horizon_ic_mean(
    factor_df: pd.DataFrame,
    close: pd.DataFrame,
    start_date: str,
    end_date: str,
    horizon: int,
    poolsel_path: str | None = None,
) -> float:
    analyzer = _get_analyzer(start_date, end_date, horizon, poolsel_path)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)

    result = analyzer.analyze({"hook_factor": factor_df}, close, factor_lag=True)
    stat_output = result.get("stat_output")
    if stat_output is None or stat_output.empty or "hook_factor" not in stat_output.index:
        return np.nan

    return float(stat_output.loc["hook_factor"].get("ic_mean", np.nan))


@register_hook("plain_ic_metrics")
def plain_ic_metrics_hook(
    factor_df: pd.DataFrame,
    loaded_deps: dict,
    start_date: str,
    end_date: str,
    params: dict | None = None,
) -> dict:
    """
    计算普通 N 天 forward IC 均值。

    YAML 示例:
    metric_hooks:
      - plain_ic_metrics
    metric_hook_params:
      plain_ic_metrics:
        horizons: [5, 20]
    """
    params = params or {}
    horizons = params.get("horizons") or [5, 20]
    close = loaded_deps.get("close")
    poolsel_path = loaded_deps.get("poolsel_path")

    if close is None:
        raise ValueError("plain_ic_metrics hook 需要 runtime_context 中提供 close DataFrame")

    metrics = {}
    for horizon in horizons:
        horizon = int(horizon)
        metrics[f"ic_{horizon}_mean"] = _compute_single_horizon_ic_mean(
            factor_df=factor_df,
            close=close,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            poolsel_path=poolsel_path,
        )

    return metrics
