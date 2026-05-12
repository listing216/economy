"""
correlations.py — 因子相关性计算工具（用于 MMR 多样性筛选）

提供截面相关性和时序相关性的 Numba 加速实现，
以及用于 MMR 选择的综合相关性矩阵计算。

主要函数：
    calculate_correlation_numba()         截面或时序相关性（单对）
    calculate_comprehensive_correlation() 综合相关性（截面 + 时序加权）
    cal_correlation_matrix_comprehensive() 批量计算因子对的综合相关性矩阵
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from joblib import Parallel, delayed
from tqdm import tqdm


# ================================================================
# =================== Numba 加速核心函数 =========================
# ================================================================

@njit
def _numba_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相关系数（Numba 加速）"""
    n = len(x)
    mean_x = x.mean()
    mean_y = y.mean()
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    for i in range(n):
        xd = x[i] - mean_x
        yd = y[i] - mean_y
        sum_xy += xd * yd
        sum_x2 += xd * xd
        sum_y2 += yd * yd
    if sum_x2 == 0 or sum_y2 == 0:
        return np.nan
    return sum_xy / np.sqrt(sum_x2 * sum_y2)


def _numba_batch_correlation(
    factor1_values: np.ndarray,
    factor2_values: np.ndarray,
    valid_mask: np.ndarray,
    min_stocks: int = 20,
) -> np.ndarray:
    """对每个时间截面（或股票时序）计算相关系数"""
    n_times = factor1_values.shape[0]
    results = np.full(n_times, np.nan)
    for i in range(n_times):
        mask = valid_mask[i]
        n_valid = np.sum(mask)
        if n_valid >= min_stocks:
            x = factor1_values[i][mask]
            y = factor2_values[i][mask]
            results[i] = _numba_corrcoef(x, y)
    return results


# ================================================================
# =================== 对外接口 ===================================
# ================================================================

def calculate_correlation_numba(
    factor1: pd.DataFrame,
    factor2: pd.DataFrame,
    type: str = "cs",
) -> float:
    """
    计算两个因子的平均相关系数（Numba 加速）。

    Args:
        factor1, factor2: 因子 DataFrame（日期 × 股票）
        type: 'cs' = 截面相关性（跨股票），'ts' = 时序相关性（跨时间）

    Returns:
        float: 平均相关系数
    """
    if type == "ts":
        f1 = factor1.T
        f2 = factor2.T
    else:
        f1 = factor1
        f2 = factor2

    common_times = f1.index.intersection(f2.index)
    common_stocks = f1.columns.intersection(f2.columns)

    if len(common_times) == 0 or len(common_stocks) == 0:
        return 0.0

    v1 = f1.loc[common_times, common_stocks].values.astype(np.float64)
    v2 = f2.loc[common_times, common_stocks].values.astype(np.float64)

    valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
    valid_counts = np.sum(valid_mask, axis=1)
    if not np.any(valid_counts >= 20):
        return 0.0

    correlations = _numba_batch_correlation(v1, v2, valid_mask, min_stocks=20)
    valid = correlations[~np.isnan(correlations)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def calculate_comprehensive_correlation(
    factor1: pd.DataFrame,
    factor2: pd.DataFrame,
    cs_weight: float = 0.5,
    ts_weight: float = 0.5,
) -> float:
    """
    综合相关性 = cs_weight × 截面相关性 + ts_weight × 时序相关性

    Args:
        factor1, factor2: 因子 DataFrame
        cs_weight: 截面相关性权重（默认 0.5）
        ts_weight: 时序相关性权重（默认 0.5）

    Returns:
        float: 综合相关系数
    """
    cs = calculate_correlation_numba(factor1, factor2, type="cs")
    ts = calculate_correlation_numba(factor1, factor2, type="ts")
    return cs_weight * cs + ts_weight * ts


def cal_correlation_matrix_comprehensive(
    alpha_dict: dict,
    cs_weight: float = 0.5,
    ts_weight: float = 0.5,
) -> pd.DataFrame:
    """
    并行计算所有因子对的综合相关性矩阵（上三角 → 对称填充）。

    Args:
        alpha_dict: {因子名: DataFrame} 字典（可含候选因子 + baseline 因子）
        cs_weight:  截面权重
        ts_weight:  时序权重

    Returns:
        pd.DataFrame: 相关性矩阵，行列均为因子名
    """
    factor_keys = list(alpha_dict.keys())
    n = len(factor_keys)

    corr_matrix = pd.DataFrame(np.eye(n), index=factor_keys, columns=factor_keys)

    factor_pairs = [
        (factor_keys[i], factor_keys[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]

    def _compute(pair):
        ki, kj = pair
        corr = calculate_comprehensive_correlation(
            alpha_dict[ki], alpha_dict[kj], cs_weight, ts_weight
        )
        return ki, kj, corr

    results = Parallel(n_jobs=-1)(
        delayed(_compute)(pair)
        for pair in tqdm(factor_pairs, desc="计算综合相关性矩阵")
    )

    for ki, kj, corr in results:
        if not np.isnan(corr):
            corr_matrix.loc[ki, kj] = corr
            corr_matrix.loc[kj, ki] = corr

    return corr_matrix
