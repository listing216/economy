"""
evaluator.py — 因子表现评估器

包含：
- Config: 评估配置（分组数、日期范围、换仓频率等）
- FactorPerformanceAnalyzer: 核心评估器（IC + 分层收益 + 统计指标）
- cal_alpha(): 批量计算因子值工具函数

主要使用方式：
    from evaluator import FactorPerformanceAnalyzer, Config, cal_alpha
    config = Config(group_num=10, date_start='20100101', date_end='20191231', adj_dates='10day')
    analyzer = FactorPerformanceAnalyzer(config)
    result = analyzer.analyze(fac_dict, close, factor_lag=True)
    print(result['stat_output'])   # IC均值、IC IR、多头超额等
"""

import sys
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from joblib import Parallel, delayed
from numba import prange, njit
import re
from typing import Dict, Optional, Union, Tuple, List
from dataclasses import dataclass, field
import warnings
from dataloader import dataloader

# ================================================================
# ========================== config ===========================
# ================================================================



@dataclass
class Config:
    """因子分析器配置类"""

    group_num: int = 10
    

    date_start: Optional[str] = None
    date_end: Optional[str] = None
    

    adj_dates: str = '10day'
    auto_reverse: bool = True  # 是否根据IC自动判断因子方向
    field: Optional[str] = None  # None表示全A，"沪深300"、"中证500"
    calculate_corr_exposure: bool = False  # 缺少这个配置项
    poolsel_path: Optional[str] = None  # 指数成分股过滤文件路径，如 'poolsel/zz800.pqt'
    

def _normalize_poolsel_mask(obj) -> pd.DataFrame:
    """将不同 pickle/parquet 结构统一成宽表 bool DataFrame。"""
    if isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.MultiIndex):
            mask = obj.notna().unstack(fill_value=False)
        else:
            mask = obj.to_frame().notna()
    elif isinstance(obj, pd.DataFrame):
        if isinstance(obj.index, pd.MultiIndex) and obj.shape[1] == 1:
            mask = obj.iloc[:, 0].notna().unstack(fill_value=False)
        elif {"trade_dt", "code"}.issubset(set(obj.columns)):
            value_cols = [c for c in obj.columns if c not in {"trade_dt", "code"}]
            if not value_cols:
                raise ValueError("poolsel pickle 缺少取值列")
            value_col = value_cols[0]
            long_df = obj[["trade_dt", "code", value_col]].copy()
            long_df["trade_dt"] = long_df["trade_dt"].astype(str)
            long_df["code"] = long_df["code"].astype(str)
            long_df[value_col] = long_df[value_col].notna()
            mask = long_df.pivot(index="trade_dt", columns="code", values=value_col)
        else:
            mask = obj.notna()
    else:
        raise TypeError(f"不支持的 poolsel 类型: {type(obj)!r}")

    mask = mask.fillna(False).astype(bool)
    mask.index = mask.index.astype(str)
    mask.columns = mask.columns.astype(str)
    return mask


def _load_poolsel_mask(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pkl", ".pickle"}:
        return _normalize_poolsel_mask(pd.read_pickle(path))
    return _normalize_poolsel_mask(pd.read_parquet(path))

    
# ================================================================
# ========================== numba ===========================
# ================================================================


@njit
def spearman_corr(x, y):
    """Spearman相关系数计算"""
    n = len(x)
    if n < 2:
        return np.nan
    x_rank = np.empty(n)
    y_rank = np.empty(n)

    x_sorted_idx = np.argsort(x)
    y_sorted_idx = np.argsort(y)

    for i in range(n):
        x_rank[x_sorted_idx[i]] = i + 1.0
        y_rank[y_sorted_idx[i]] = i + 1.0
    x_mean = np.mean(x_rank)
    y_mean = np.mean(y_rank)

    numerator = 0.0
    x_var = 0.0
    y_var = 0.0

    for i in range(n):
        x_diff = x_rank[i] - x_mean
        y_diff = y_rank[i] - y_mean
        numerator += x_diff * y_diff
        x_var += x_diff * x_diff
        y_var += y_diff * y_diff
    
    if x_var == 0 or y_var == 0:
        return np.nan
    
    return numerator / np.sqrt(x_var * y_var)

@njit
def pearson_corr(x, y):
    """Pearson相关系数计算"""
    n = len(x)
    if n < 2:
        return np.nan  
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = 0.0  
    x_var = 0.0     
    y_var = 0.0     

    for i in range(n):
        x_diff = x[i] - x_mean  
        y_diff = y[i] - y_mean
        numerator += x_diff * y_diff  
        x_var += x_diff * x_diff      
        y_var += y_diff * y_diff      
    
    if x_var == 0 or y_var == 0:
        return np.nan
    
    return numerator / np.sqrt(x_var * y_var)

@njit(parallel=True)
def batch_correlation(factor_values, return_values, valid_mask, type = 'spearman'):
    n_dates = factor_values.shape[0]
    ic_values = np.empty(n_dates)
    
    for i in prange(n_dates):
        mask = valid_mask[i]
        if np.sum(mask) < 2:
            ic_values[i] = np.nan
        else:
            x = factor_values[i][mask]
            y = return_values[i][mask]
            if type == 'spearman':
                ic_values[i] = spearman_corr(x, y)
            elif type == 'pearson':   
                ic_values[i] = pearson_corr(x, y)

    
    return ic_values

@njit
def split_groups(stock_count, group_num):
    """分组函数"""
    base_size = stock_count // group_num
    remainder = stock_count % group_num
    
    start = (group_num - 1) >> 1
    
    order = np.empty(group_num, dtype=np.int32)
    order[0] = start
    
    for i in range(1, group_num):
        if i % 2 == 1:
            order[i] = start + (i + 1) // 2
        else:
            order[i] = start - i // 2
    
    group_sizes = np.empty(group_num, dtype=np.int32)
    for i in range(group_num):
        is_in_remainder = False
        for j in range(min(remainder, group_num)):
            if order[j] == i:
                is_in_remainder = True
                break
        
        group_sizes[i] = base_size + (1 if is_in_remainder else 0)
    
    indices = np.empty((group_num, 2), dtype=np.int32)
    cumulative = 0
    for i in range(group_num):
        indices[i, 0] = cumulative
        cumulative += group_sizes[i]
        indices[i, 1] = cumulative
    
    return indices

@njit(parallel=True)
def calculate_portfolio_returns_nday(group_matrix, daily_returns, nday_freq, group_num):
    """
    计算N-day分散调仓的组合收益率（对日期并行优化版本）  
    Parameters:
    -----------
    group_matrix : 3D array, shape=(n_dates, nday_freq, n_stocks)
        分组矩阵，group_matrix[t, p, stock] = g 表示股票属于第g组
    daily_returns : 2D array, shape=(n_dates, n_stocks)  
        日收益率矩阵
    nday_freq : int
        N-day频率，同时也是子组合数量
    group_num : int
        分组数量
        
    Returns:
    --------
    portfolio_returns : 3D array, shape=(n_dates, nday_freq, group_num)
        每个子组合每个分组的日收益率
    """
    n_dates, n_stocks = daily_returns.shape
    portfolio_returns = np.empty((n_dates, nday_freq, group_num))
    
    # 对日期并行，充分利用多核CPU
    for t in prange(n_dates):
        if t == 0:
            # 第一天所有收益率为0
            for p in range(nday_freq):
                for g in range(group_num):
                    portfolio_returns[t, p, g] = 0.0
        else:
            current_returns = daily_returns[t, :]
            
            for p in range(nday_freq):
                for g in range(group_num):
                    # 找到属于第g+1组的股票（group_id从1开始）
                    group_id = g + 1
                    group_mask = group_matrix[t-1, p, :] == group_id
                    
                    # 计算该组的收益率
                    if np.sum(group_mask) > 0:
                        group_returns = current_returns[group_mask]
                        valid_returns = group_returns[~np.isnan(group_returns)]
                        if len(valid_returns) > 0:
                            portfolio_returns[t, p, g] = np.mean(valid_returns)
                        else:
                            portfolio_returns[t, p, g] = 0.0
                    else:
                        portfolio_returns[t, p, g] = 0.0
    
    return portfolio_returns

@njit(parallel=True)
def generate_nday_holdings(factor_matrix, group_num, nday_freq, reverse=False):
    """
    按(组合×分组)并行,避免重复计算
    Parameters:
    -----------
    factor_matrix : 2D array, shape=(n_dates, n_stocks)
        因子值矩阵
    group_num : int
        分组数量
    nday_freq : int
        N-day频率,同时也是子组合数量和调仓频率
    reverse : bool
        是否反转排序,True表示升序排列(小的在前),False表示降序排列(大的在前)
    
    Returns:
    --------
    group_matrix : 3D array, shape=(n_dates, nday_freq, n_stocks)
        分组矩阵,group_matrix[t, p, stock] = g 表示第t天第p个子组合中stock属于第g组
        0表示不属于任何组(无有效因子值)
    """
    n_dates, n_stocks = factor_matrix.shape
    group_matrix = np.zeros((n_dates, nday_freq, n_stocks))
    
    # 并行处理每个(子组合, 分组)组合
    for portfolio_idx in prange(nday_freq * group_num):
        p = portfolio_idx // group_num  # 子组合索引
        g = portfolio_idx % group_num   # 分组索引
        group_id = g + 1
        
        for t in range(n_dates):
            # 判断是否为调仓日
            is_rebalance_day = (t >= p) and ((t - p) % nday_freq == 0)
            
            if is_rebalance_day:
                # 调仓日：重新分组
                factor_values = factor_matrix[t, :]
                valid_mask = ~np.isnan(factor_values)
                
                if np.sum(valid_mask) >= group_num:
                    valid_factors = factor_values[valid_mask]
                    valid_indices = np.where(valid_mask)[0]
                    
                    # 排序并分组
                    if reverse:
                        sorted_indices = valid_indices[np.argsort(valid_factors)]
                    else:
                        sorted_indices = valid_indices[np.argsort(-valid_factors)]
                    
                    # 计算当前分组的股票范围
                    n_valid = len(sorted_indices)
                    group_indices = split_groups(n_valid, group_num)
                    start_pos, end_pos = group_indices[g, 0], group_indices[g, 1]
                    
                    # 分配股票到当前分组
                    for pos in range(start_pos, min(end_pos, n_valid)):
                        group_matrix[t, p, sorted_indices[pos]] = group_id
                        
            elif t > 0:
                # 非调仓日：继承前一日分组
                group_matrix[t, p, :] = group_matrix[t-1, p, :]
    
    return group_matrix


@njit(parallel=True)
def batch_regression(X_matrix, factor_values, valid_mask):
    """
    批量回归计算
    """
    n_dates, n_stocks, n_factors = X_matrix.shape

    exposure_matrix = np.empty((n_dates, n_factors))
    stock_counts = np.empty(n_dates)

    for i in prange(n_dates):
        mask = valid_mask[i]
        n_valid = np.sum(mask)

        if n_valid < 10:
            exposure_matrix[i] = np.full(n_factors, np.nan) 
            stock_counts[i] = 0
        
        else:
            X = X_matrix[i][mask]
            y = factor_values[i][mask]
        
            try:
                XTX = X.T @ X
                XTX_pinv = np.linalg.pinv(XTX)  # 伪逆矩阵
                XTy = X.T @ y
                beta = XTX_pinv @ XTy
                
                exposure_matrix[i] = beta
                stock_counts[i] = n_valid
            except:
                exposure_matrix[i] = np.full(n_factors, np.nan) 
                stock_counts[i] = 0
    
    return exposure_matrix, stock_counts

# ================================================================
# ========================== 补充函数 ===========================
# ================================================================

def _vectorized_ic_calculation(factor_aligned, returns_aligned, type = 'spearman'):
    """
    矢量化IC计算
    """
    # 转换为numpy数组用于numba计算
    factor_values = factor_aligned.values
    return_values = returns_aligned.values
    
    # 创建有效值掩码
    factor_valid = ~np.isnan(factor_values)
    return_valid = ~np.isnan(return_values)
    valid_mask = factor_valid & return_valid
    
    # 使用numba批量计算
    
    ic_values = batch_correlation(factor_values, return_values, valid_mask)
    
    # 计算有效股票数
    stock_counts = np.sum(valid_mask, axis=1)
    
    # 创建结果DataFrame
    ic_df = pd.DataFrame(
        {'ic': ic_values}, 
        index=factor_aligned.index
    ).shift(1)
    
    stock_count_df = pd.DataFrame(
        {'stock_count': stock_counts}, 
        index=factor_aligned.index
    ).shift(1)
    
    return {
        'ic': ic_df,
        'stock_count': stock_count_df
    }

def calculate_corr(factor, barra_dict, method = 'cs'):

    barra_dict_copy = barra_dict.copy()
    factor_copy = factor.copy()
    if method == 'ts':
        factor_copy = factor.T
        for key, barra in barra_dict.items():
            barra_dict_copy[key] = barra.T

    all_corr_results = {}
    for key, barra in barra_dict_copy.items():

            common_dates = factor_copy.index.intersection(barra.index)
            common_stocks = factor_copy.columns.intersection(barra.columns)
            
            factor_aligned = factor_copy.loc[common_dates, common_stocks]
            barra_aligned = barra.loc[common_dates, common_stocks]

            result = _vectorized_ic_calculation(factor_aligned, barra_aligned, 'pearson')

            all_corr_results[key] = np.nanmean(result['ic']['ic'])
    
    result_df = pd.DataFrame(all_corr_results, index=[0]).T
    result_df.columns = ['exposure']  # 给列命名
    return result_df.sort_values('exposure', ascending=False)

# 生成行业哑变量字典
def create_indus_dict(data):
    all_values = data.values.ravel()  # 将DataFrame转为一维数组
    all_values_non_null = [x for x in all_values if x is not None]  # 过滤None
    all_unique_vals = np.unique(all_values_non_null)
    indus_list = list(all_unique_vals)
    indus_dict = {}
    for indus_name in tqdm(indus_list):
            indus_mask = (data == indus_name).astype(np.float32)
            indus_dict[indus_name] = indus_mask

    print(f"成功构建 {len(indus_dict)} 个因子")
    return indus_dict

def calculate_factor_exposure(factor, factor_dict):
    """
    通用暴露度计算函数
    
    Parameters:
    -----------
    factor : DataFrame
        时间×股票的目标因子数据
    factor_dict : dict
        {因子名: DataFrame(时间×股票)} 解释变量字典   
    Returns:
    --------
    dict
        - 'exposures': DataFrame(时间×因子) 暴露度矩阵
        - 'stock_count': DataFrame(时间×1) 有效股票数
    """


    
    # 获取因子名称列表
    factor_names = list(factor_dict.keys())

    
    # 对齐数据
    common_dates = factor.index
    for factor_data in factor_dict.values():
        common_dates = common_dates.intersection(factor_data.index)
    
    common_stocks = factor.columns
    for factor_data in factor_dict.values():
        common_stocks = common_stocks.intersection(factor_data.columns)
    
    factor_aligned = factor.loc[common_dates, common_stocks]

    # 构建3D因子矩阵
    n_dates, n_stocks = factor_aligned.shape
    n_factors = len(factor_names)
    X_matrix = np.zeros((n_dates, n_stocks, n_factors), dtype=np.float32)

    for i, factor_name in enumerate(factor_names):
        factor_data = factor_dict[factor_name].loc[common_dates, common_stocks]
        X_matrix[:, :, i] = factor_data.values.astype(np.float32)

    # 准备目标因子数据和有效掩码
    factor_values = factor_aligned.values.astype(np.float32)
    factor_valid = ~np.isnan(factor_values)
    
    # 检查解释变量是否有效
    X_valid = ~np.isnan(X_matrix)
    X_all_valid = np.all(X_valid, axis=2)  # 所有解释变量都有效
    
    valid_mask = factor_valid & X_all_valid

    exposure_matrix, stock_counts = batch_regression(
        X_matrix, factor_values, valid_mask
    )

    
    # 创建结果DataFrame
    exposure_df = pd.DataFrame(
        exposure_matrix,
        index=factor_aligned.index,
        columns=factor_names
    )

    # exposure_df_cumsum = exposure_df.cumsum()
    
    stock_count_df = pd.DataFrame(
        {'stock_count': stock_counts},
        index=factor_aligned.index
    )

    return {
        'exposures': exposure_df,
        'stock_count': stock_count_df
    }


# ================================================================
# ========================== 计算IC类 ===========================
# ================================================================




class ICCalculator:

    
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_single(self, factor: pd.DataFrame, close: pd.DataFrame, 
                        ) -> Dict:
        """计算单个因子的IC"""
        
        # 数据预处理
        factor, close = self._preprocess_data(factor, close)
        
        # 计算收益率
        ret = (close.shift(-1) - close) / close
        
        # 对齐数据
        common_dates = factor.index.intersection(ret.index)
        common_stocks = factor.columns.intersection(ret.columns)
        
        factor_aligned = factor.loc[common_dates, common_stocks]
        returns_aligned = ret.loc[common_dates, common_stocks]
        
        return self._vectorized_ic_calculation(factor_aligned, returns_aligned)
    
    def calculate_batch(self, fac_dict_filtered: Dict[str, pd.DataFrame], close: pd.DataFrame,
                       date_start: Optional[str] = None, date_end: Optional[str] = None) -> Dict:
        """批量计算多个因子的IC"""
        
        factor_names = list(fac_dict_filtered.keys())
        all_ic_results = {}
        all_stock_count_results = {}
        
        for factor_name in factor_names:
            result = self.calculate_single(
                fac_dict_filtered[factor_name], close
            )
            all_ic_results[factor_name] = result['ic']['ic']
            all_stock_count_results[factor_name] = result['stock_count']['stock_count']
        
        return {
            'ic': pd.DataFrame(all_ic_results),
            'stock_count': pd.DataFrame(all_stock_count_results)
        }
    
    def _preprocess_data(self, factor: pd.DataFrame, close: pd.DataFrame,
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        
        if self.config.date_start:
            close = close[close.index.astype(str) >= self.config.date_start]
            factor = factor[factor.index.astype(str) >= self.config.date_start]
        if self.config.date_end:
            close = close[close.index.astype(str) <= self.config.date_end]
            factor = factor[factor.index.astype(str) <= self.config.date_end]
        
        # 调整频率
        if self.config.adj_dates.endswith('day'):
            day_offset = int(self.config.adj_dates[:-3])
            factor = factor.iloc[::day_offset]
            close = close.iloc[::day_offset]
        
        return factor, close
    
    def _vectorized_ic_calculation(self, factor_aligned: pd.DataFrame, 
                                  returns_aligned: pd.DataFrame) -> Dict:
        """矢量化IC计算"""
        
        factor_values = factor_aligned.values
        return_values = returns_aligned.values
        
        factor_valid = ~np.isnan(factor_values)
        return_valid = ~np.isnan(return_values)
        valid_mask = factor_valid & return_valid
        
        # 使用numba计算
        ic_values = batch_correlation(
            factor_values, return_values, valid_mask
        )
        
        stock_counts = np.sum(valid_mask, axis=1)
        
        ic_df = pd.DataFrame(
            {'ic': ic_values}, 
            index=factor_aligned.index
        ).shift(1)
        
        stock_count_df = pd.DataFrame(
            {'stock_count': stock_counts}, 
            index=factor_aligned.index
        ).shift(1)
        
        return {
            'ic': ic_df,
            'stock_count': stock_count_df
        }
    

# ================================================================
# ========================== 计算return类 =========================
# ================================================================

class GroupReturnsCalculator:

    
    def __init__(self, config: Config):
        self.config = config

    
    def calculate_single(self, factor: pd.DataFrame, close: pd.DataFrame,
                        reverse: Optional[bool] = None) -> Dict:
        """计算单个因子的分组收益"""
        
        # 数据预处理
        factor, close = self._preprocess_data(factor, close)
        
        # 计算日收益率
        daily_returns = close.pct_change(fill_method=None)
        daily_returns.iloc[0] = 0.0
        
        # 自动判断因子方向
        if reverse is None and self.config.auto_reverse:
            reverse = self._auto_determine_direction(factor, close)
        
        nday_freq = int(self.config.adj_dates[:-3])
        return self._calculate_nday_returns(factor, daily_returns, reverse, nday_freq)
    
    def calculate_batch(self, fac_dict_filtered: Dict[str, pd.DataFrame], close: pd.DataFrame,
                       reverse_list: Optional[List[bool]] = None) -> Dict:
        """批量计算多个因子的分组收益"""
        
        factor_names = list(fac_dict_filtered.keys())
        
        if reverse_list is None:
            reverse_list = [None] * len(factor_names)
        
        result = {}
        for i, factor_name in enumerate(factor_names):
            factor_result = self.calculate_single(
                fac_dict_filtered[factor_name], close, reverse_list[i]
            )
            result[factor_name] = factor_result
        
        return result
    
    def _preprocess_data(self, factor: pd.DataFrame, close: pd.DataFrame,
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        
        if self.config.date_start:
            close = close[close.index.astype(str) >= self.config.date_start]
            factor = factor[factor.index.astype(str) >= self.config.date_start]
        if self.config.date_end:
            close = close[close.index.astype(str) <= self.config.date_end]
            factor = factor[factor.index.astype(str) <= self.config.date_end]

        
        return factor, close
    
    def _auto_determine_direction(self, factor: pd.DataFrame, close: pd.DataFrame,
                                ) -> bool:
        """自动判断因子方向"""
        
        print("自动判断因子方向...")
        ic_calc = ICCalculator(self.config)
        ic_result = ic_calc.calculate_single(factor, close)
        mean_ic = ic_result['ic']['ic'].mean()
        reverse = mean_ic < 0
        print(f"平均IC: {mean_ic:.4f}, {'使用反转' if reverse else '不反转'}")
        return reverse
    
    def _calculate_nday_returns(self, factor: pd.DataFrame, daily_returns: pd.DataFrame,
                               reverse: bool, nday_freq: int) -> Dict:

        """
        计算单个因子的N-day分散调仓收益率
        
        Parameters:
        -----------
        reverse : bool
            是否反转因子排序，True表示因子值小的排在Group_1
        """
        # 对齐数据
        common_dates = factor.index.intersection(daily_returns.index)
        common_stocks = factor.columns.intersection(daily_returns.columns)
        
        factor_aligned = factor.loc[common_dates, common_stocks]
        returns_aligned = daily_returns.loc[common_dates, common_stocks]
        

        factor_matrix = factor_aligned.values.astype(np.float64)
        returns_matrix = returns_aligned.values.astype(np.float64)
        
        # 生成N个子组合的分组矩阵
        group_matrix = generate_nday_holdings(
            factor_matrix, self.config.group_num, nday_freq, reverse=reverse
        )
        
        # 计算子组合收益率
        subportfolio_returns = calculate_portfolio_returns_nday(
            group_matrix, returns_matrix, nday_freq, self.config.group_num
        )
        
        # 平均所有子组合的收益率，得到最终的分组收益率
        final_group_returns = np.mean(subportfolio_returns, axis=1)
        
        # 构建结果DataFrame
        returns_df = pd.DataFrame(
            final_group_returns,
            index=common_dates,
            columns=[f'Group_{i+1}' for i in range(self.config.group_num)]
        )
        
        # 添加市场收益率和L-S收益率
        returns_df['市场'] = returns_df.mean(axis=1)
        returns_df['L-S'] = returns_df['Group_1'] - returns_df[f'Group_{self.config.group_num}']
        
        cumulative_nav = (1 + returns_df).cumprod()
        excess_returns = returns_df.sub(returns_df['市场'], axis=0)
        excess_nav = (1 + excess_returns).cumprod()
        
        return {
            'returns': returns_df,
            'cumulative_nav': cumulative_nav,
            'excess_returns': excess_returns,
            'excess_nav': excess_nav,
            'subportfolio_returns':subportfolio_returns
        }

# ================================================================
# ========================== 合成统计指标类 ===========================
# ================================================================


class StatisticsCalculator:

    
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_ic_stats(self, ic_output: Dict) -> pd.DataFrame:
        """计算IC统计指标"""
        
        ic_df = ic_output['ic']
        stock_count_df = ic_output['stock_count']
        factor_names = ic_df.columns.tolist()
        
        stats_dict = {
            'ic_mean': [],
            'ic_ir': [],
            'ict': [],
            'icstocknum': [],
        }
        
        for factor_name in factor_names:
            ic_series = ic_df[factor_name]
            stock_count_series = stock_count_df[factor_name]
            
            # IC平均值
            ic_mean = ic_series.mean()
            stats_dict['ic_mean'].append(ic_mean)
            
            # IC IR
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            stats_dict['ic_ir'].append(ic_ir)
            
            # t统计量
            t_stat, _ = stats.ttest_1samp(ic_series.dropna(), 0)
            stats_dict['ict'].append(t_stat)
            
            # 股票数平均值
            stock_count_mean = stock_count_series.mean()
            stats_dict['icstocknum'].append(stock_count_mean)
        
        return pd.DataFrame(stats_dict, index=factor_names)
    
    def calculate_return_stats(self, group_returns_output: Dict) -> pd.DataFrame:
        """计算收益率统计指标"""
        annual_factor = 252

        factor_names = list(group_returns_output.keys())
        
        stats_dict = {
            'long_excret': [],
            'long_sharpe': [],
            'long_ir': [],
            'long_excmdd': [],
            'ls_ret': [],
            'ls_std': [],
            'ls_sharpe': [],
            'ls_mdd': [],
        }
        
        for factor_name in factor_names:
            factor_data = group_returns_output[factor_name]
            returns_df = factor_data['returns']
            excess_returns_df = factor_data['excess_returns']
            cumulative_nav = factor_data['cumulative_nav']
            excess_nav = factor_data['excess_nav']
            
            # 多头组（Group_1）数据
            long_returns = returns_df['Group_1']
            long_nav = cumulative_nav['Group_1']
            long_excess_returns = excess_returns_df['Group_1']
            long_excess_nav = excess_nav['Group_1']
            
            # L-S数据
            ls_returns = returns_df['L-S']
            ls_nav = cumulative_nav['L-S']

            # 计算期数
            periods = len(long_returns) - 1
            
            # 1. 多头年化超额收益
            total_excess_return = long_excess_nav.iloc[-1] - 1
            if periods > 0 :
                long_excret = (1 + total_excess_return) ** (annual_factor / periods) - 1
            else:
                long_excret = 0
            stats_dict['long_excret'].append(long_excret)
            
            # 2. 多头sharpe比率
            if long_returns.std() > 0:
                total_return = long_nav.iloc[-1] - 1
                long_ret = (1 + total_return) ** (annual_factor / periods) - 1
                long_sharpe = long_ret / (long_returns.std() * np.sqrt(annual_factor))
            else:
                long_sharpe = 0
            stats_dict['long_sharpe'].append(long_sharpe)
            
            # 3. 多头信息比率
            if long_excess_returns.std() > 0:
                long_ir = long_excret / (long_excess_returns.std() * np.sqrt(annual_factor))
            else:
                long_ir = 0
            stats_dict['long_ir'].append(long_ir)
            
            # 4. 多头超额最大回撤
            long_excmdd = self._calculate_max_drawdown(long_excess_nav)
            stats_dict['long_excmdd'].append(long_excmdd)
            
            # 5. 多空年化收益
            total_ls_return = ls_nav.iloc[-1] - 1
            if periods > 0 :
                ls_ret = (1 + total_ls_return) ** (annual_factor / periods) - 1
            else:
                ls_ret = 0
            stats_dict['ls_ret'].append(ls_ret)
            
            # 6. 多空年化波动率
            ls_std = ls_returns.std() * np.sqrt(annual_factor)
            stats_dict['ls_std'].append(ls_std)
            
            # 7. 多空sharpe比率
            if ls_std > 0:
                ls_sharpe = ls_ret / ls_std
            else:
                ls_sharpe = 0
            stats_dict['ls_sharpe'].append(ls_sharpe)
            
            # 8. 多空最大回撤
            ls_mdd = self._calculate_max_drawdown(ls_nav)
            stats_dict['ls_mdd'].append(ls_mdd)
        
        # 构建结果DataFrame
        stat_df = pd.DataFrame(stats_dict, index=factor_names)
    
        return stat_df
    def _calculate_max_drawdown(self, nav_series: pd.Series) -> float:
        """计算最大回撤"""
        running_max = -np.inf
        max_dd = -np.inf
        
        for value in nav_series:
            if value > running_max:
                running_max = value
            else:
                dd = (running_max - value) / running_max
                max_dd = max(max_dd, dd)
        
        return max_dd


# ================================================================
# ========================== 因子表现 ===========================
# ================================================================
class FactorPerformanceAnalyzer:
    _numba_warmed_up = False 
    _STOCK_POOL_PATH = "index/"  # 类变量，统一管理路径
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        self._load_stock_pool()

        # ------------------------------------------------------------------
        # poolsel：指数成分股过滤掩码（由 build_data.py 生成）
        # config.poolsel_path 如 'poolsel/zz800.pqt'，值=1 表示在成分股内
        # 不设置则全A运行
        # ------------------------------------------------------------------
        poolsel_path = self.config.poolsel_path
        if poolsel_path and os.path.exists(poolsel_path):
            self.pool_mask = _load_poolsel_mask(poolsel_path)
            print(f"[poolsel] 已加载成分股掩码: {poolsel_path}  shape={self.pool_mask.shape}")
        else:
            self.pool_mask = None
            if poolsel_path:
                print(f"[poolsel] 文件不存在，跳过过滤: {poolsel_path}")

        if self.config.calculate_corr_exposure:
            self.barra_dict = dataloader('barra',self.config.date_start,self.config.date_end)
            self._load_indus_factors()

        # 初始化子模块
        self.ic_calculator = ICCalculator(self.config)
        self.returns_calculator = GroupReturnsCalculator(self.config)
        self.stats_calculator = StatisticsCalculator(self.config)

        if not FactorPerformanceAnalyzer._numba_warmed_up:
            print("正在预热numba函数...")
            self._warmup()
            FactorPerformanceAnalyzer._numba_warmed_up = True
            print("预热完成!")

        print(f"因子分析器初始化完成，配置: {self.config}")


    def _load_stock_pool(self):
        """根据配置加载对应的股票池数据"""
        if self.config.field is None:
            self.field = None
            return
            
        pool_file = f"{self._STOCK_POOL_PATH}{self.config.field}.pqt"
        try:
            self.field = pd.read_parquet(pool_file)
            self.field.index = self.field.index.astype(str)
        except Exception as e:
            self.field = None

    def _warmup(self):
        """预热"""

        n_dates = 30  
        n_stocks = 100  

        
        dates = pd.date_range('20240101', periods=n_dates, freq='D').strftime('%Y%m%d').tolist()
        stocks = [f'stock_{i:03d}' for i in range(n_stocks)]
        
        np.random.seed(42) 
        

        factor_values = np.random.randn(n_dates, n_stocks).astype(np.float64)
        factor_values[::10, ::20] = np.nan  
        warmup_factor = pd.DataFrame(factor_values, index=dates, columns=stocks)
        

        returns = np.random.randn(n_dates, n_stocks) * 0.01
        returns[0] = 0
        prices = np.cumprod(1 + returns, axis=0) * 100
        prices[::15, ::25] = np.nan 
        warmup_close = pd.DataFrame(prices, index=dates, columns=stocks)
        

        try:
            warmup_fac_dict = {'warmup': warmup_factor}
            _ = self.analyze(warmup_fac_dict, warmup_close)
        except:

            pass
    def _load_indus_factors(self):
        """加载行业数据并按时间范围截取，然后生成哑变量"""

        # 加载并截取行业数据
        indus_data = pd.read_parquet("indus/申万行业.pqt")
        indus_data.index = indus_data.index.astype(str)
        
        if self.config.date_start:
            indus_data = indus_data[indus_data.index >= self.config.date_start]
        if self.config.date_end:
            indus_data = indus_data[indus_data.index <= self.config.date_end]
        
        # 生成哑变量字典
        self.indus_dict = create_indus_dict(indus_data)


    def _apply_stock_pool_filter(self, fac_dict_filtered: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """对所有因子应用股票池过滤"""
        if self.field is None:
            return fac_dict_filtered
        
        filtered_dict = {}
        for factor_name, factor_df in fac_dict_filtered.items():
            # 日期截取股票池数据
            pool_data = self.field.copy()
            if self.config.date_start:
                pool_data = pool_data[pool_data.index >= self.config.date_start]
            if self.config.date_end:
                pool_data = pool_data[pool_data.index <= self.config.date_end]
            
            # 对齐并过滤
            pool_aligned = pool_data.reindex(
                index=factor_df.index, 
                columns=factor_df.columns, 
                fill_value=0
            )
            filtered_factor = factor_df.where(pool_aligned.astype(bool), np.nan)
            filtered_dict[factor_name] = filtered_factor
        
        return filtered_dict
    

    def _calculate_batch_corr(self, fac_dict: Dict[str, pd.DataFrame], 
                              ex_dict: Dict[str, pd.DataFrame],method: str) -> Dict:
        """批量计算barra相关性"""
        results = {}
        for factor_name, factor_df in fac_dict.items():
            results[factor_name] = calculate_corr(
                factor_df, ex_dict, method = method 
            )
        return results

    def _calculate_batch_exposure(self, fac_dict: Dict[str, pd.DataFrame], 
                                  ex_dict: Dict[str, pd.DataFrame]) -> Dict:
        """批量计算暴露度"""
        results = {}
        for factor_name, factor_df in fac_dict.items():
            results[factor_name] = calculate_factor_exposure(
                factor_df, ex_dict,
            )
        return results
    
    def analyze(self, fac_dict: Dict[str, pd.DataFrame], close: pd.DataFrame, factor_lag: bool = False) -> Dict:
        """
        全面分析因子表现
        
        Parameters:
        -----------
        fac_dict_filtered : dict
            因子字典，格式: {因子名: DataFrame}
        close : DataFrame
            股价数据
            
        Returns:
        --------
        dict
            完整的分析结果
        """
        
        print("开始因子表现分析...")

        if factor_lag:
            print("对因子数据执行lag(1)处理...")
            factor_dict = {}
            for fac_name, fac_value in fac_dict.items():
                # 先shift(1)，保持完整长度
                factor_dict[fac_name] = fac_value.shift(1)
            fac_dict = factor_dict.copy()
        else:
            print("因子数据不做lag处理...")

        fac_dict_filtered = self._apply_stock_pool_filter(fac_dict)

        # ------------------------------------------------------------------
        # poolsel 过滤：在 IC / 分层收益计算前，将不在指数成分股内的截面置 NaN
        # pool_mask 格式：index=日期(str)，columns=股票代码，值=1/NaN
        # ------------------------------------------------------------------
        if self.pool_mask is not None:
            filtered_with_pool = {}
            for factor_name, factor_df in fac_dict_filtered.items():
                pool_aligned = self.pool_mask.reindex(
                    index=factor_df.index,
                    columns=factor_df.columns
                )
                filtered_with_pool[factor_name] = factor_df.where(pool_aligned == 1)
            fac_dict_filtered = filtered_with_pool

            # 同步过滤 close，保持对齐（使用临时变量，不破坏原始 close）
            pool_close_aligned = self.pool_mask.reindex(
                index=close.index,
                columns=close.columns
            )
            close = close.where(pool_close_aligned == 1)

        # 1. 计算IC
        print("计算IC...")
        ic_series = self.ic_calculator.calculate_batch(fac_dict_filtered, close)

        # 2. 计算IC统计指标
        print("计算IC统计指标...")
        stat_ic = self.stats_calculator.calculate_ic_stats(ic_series)

        # 3. 根据IC自动判断因子方向
        reverse_list = (stat_ic['ic_mean'] < 0).tolist()

        # 4. 计算分组收益
        print("计算分组收益...")
        group_returns = self.returns_calculator.calculate_batch(fac_dict_filtered, close, reverse_list)
        
        # 5. 计算收益率统计指标
        print("计算收益率统计指标...")
        stat_return = self.stats_calculator.calculate_return_stats(group_returns)
        
        # 6. 合并统计结果
        stat_output = pd.concat([stat_ic, stat_return], axis=1)
        
        # 7. 提取净值序列等其他结果
        group_annual_excess_returns = self._get_group_annual_excess_returns(group_returns)
        ls_nav_df, long_excess_nav_df = self._get_nav_series(group_returns)


        if self.config.calculate_corr_exposure:
            self.barra_dict = self._apply_stock_pool_filter(self.barra_dict)
            self.indus_dict = self._apply_stock_pool_filter(self.indus_dict)

            print("计算barra相关性统计指标...")
            cs_barra = self._calculate_batch_corr(fac_dict_filtered, self.barra_dict, 'cs')
            ts_barra = self._calculate_batch_corr(fac_dict_filtered, self.barra_dict, 'ts')

            print("计算因子暴露统计指标...")
            indus_exposure  = self._calculate_batch_exposure (fac_dict_filtered, self.indus_dict)
            barra_exposure = self._calculate_batch_exposure (fac_dict_filtered, self.barra_dict)

            print("分析完成!")
        
            return {
                'stat_output': stat_output,
                'ls_nav_df': ls_nav_df,
                'long_excess_nav_df': long_excess_nav_df,
                'group_annual_excess_returns': group_annual_excess_returns,
                'ic_series': ic_series,
                'group_returns': group_returns,
                'cs_barra':cs_barra,
                'ts_barra':ts_barra,
                'indus_exposure':indus_exposure,
                'barra_exposure':barra_exposure

            }
        
        print("分析完成!")
        
        return {
            'stat_output': stat_output,
            'ls_nav_df': ls_nav_df,
            'long_excess_nav_df': long_excess_nav_df,
            'group_annual_excess_returns': group_annual_excess_returns,
            'ic_series': ic_series,
            'group_returns': group_returns,
        }
    
    def _get_group_annual_excess_returns(self, group_returns_output: Dict) -> pd.DataFrame:
        """生成各因子分组年化超额收益汇总表"""
        annual_factor = 252
        result = {}
        for factor_name, factor_data in group_returns_output.items():
            excess_nav = factor_data['excess_nav']
            periods = len(excess_nav) - 1
            
            annual_returns = (excess_nav.iloc[-1]) ** (annual_factor / periods) - 1 if periods > 0 else excess_nav.iloc[-1] * 0
            annual_returns['市场'] = 0
            
            result[factor_name] = annual_returns
        
        return pd.DataFrame(result)
    
    def _get_nav_series(self, group_returns_output: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """提取净值序列"""
        ls_nav_dict = {}
        long_excess_nav_dict = {}
        
        for factor_name, factor_data in group_returns_output.items():
            ls_nav_dict[factor_name] = factor_data['cumulative_nav']['L-S']
            long_excess_nav_dict[factor_name] = factor_data['excess_nav']['Group_1']
        
        ls_nav_df = pd.DataFrame(ls_nav_dict)
        long_excess_nav_df = pd.DataFrame(long_excess_nav_dict)
        
        return ls_nav_df, long_excess_nav_df
    
    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        
        # 重新初始化子模块
        self.ic_calculator = ICCalculator(self.config)
        self.returns_calculator = GroupReturnsCalculator(self.config)
        self.stats_calculator = StatisticsCalculator(self.config)
        
        print(f"配置已更新: {kwargs}")






# ================================================================
# ====================== 因子值计算工具 ==========================
# ================================================================

def cal_alpha(alpha_list: list, alpha_name: list, calculator) -> dict:
    """
    批量计算因子表达式的值。

    Args:
        alpha_list:  因子表达式字符串列表
        alpha_name:  对应的因子名称列表
        calculator:  ExpressionCalculator 实例

    Returns:
        dict: {因子名: DataFrame(日期×股票)}
    """
    from tqdm import tqdm
    import numpy as np
    alpha_dict = {}
    for i, expression in tqdm(enumerate(alpha_list, start=1), total=len(alpha_list), desc="计算因子值"):
        alpha = calculator.calculate(expression)
        alpha = alpha.replace([np.inf, -np.inf], np.nan)
        alpha_dict[alpha_name[i - 1]] = alpha
    return alpha_dict
