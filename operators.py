"""
operators.py — 因子表达式算子库 + 表达式计算器

包含：
- Numba 加速的滚动窗口计算函数（均值/标准差/相关性/回归等）
- Operators 类：所有可在因子表达式中调用的静态方法
- ExpressionCalculator 类：通过 eval() 计算因子表达式字符串

使用方式：
    from operators import ExpressionCalculator, Operators
    calc = ExpressionCalculator(factor_dfs)
    factor = calc.calculate("Rank(Delta(close, 5) / Std(close, 10))")
"""

import  numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import inspect
from numba import njit, prange


# ============================
# ======= 兼容numba的函数 ========
# ============================
@njit
def _compute_mad(data):
    """计算平均绝对偏差"""
    if len(data) == 0:
        return np.nan
    mean_val = np.mean(data)
    return np.mean(np.abs(data - mean_val))

@njit
def _compute_skew(data):
    """计算偏度"""
    if len(data) < 3:
        return np.nan
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return np.nan
    
    # 计算三阶中心矩
    n = len(data)
    skewness = 0.0
    for i in range(n):
        skewness += ((data[i] - mean_val) / std_val) ** 3
    
    return skewness / n

@njit  
def _compute_kurt(data):
    """计算峰度"""
    if len(data) < 4:
        return np.nan
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return np.nan
    
    # 计算四阶中心矩
    n = len(data)
    kurtosis = 0.0
    for i in range(n):
        kurtosis += ((data[i] - mean_val) / std_val) ** 4
    
    return kurtosis / n - 3.0  

@njit
def _max_idx(data):
    """
    正确实现最大值索引，返回距离当前位置的偏移
    例如：[1, nan, 3, 2, 5] -> 最大值5在最后，距离=0
          [1, nan, 5, 2, 3] -> 最大值5在索引2，距离=4-2=2
    """
    if len(data) == 0:
        return np.nan
    
    max_val = -np.inf
    max_idx = -1
    has_valid = False
    
    # 从原始数据中找最大值，跳过NaN
    for i in range(len(data)):
        if not np.isnan(data[i]):
            has_valid = True
            if data[i] > max_val:
                max_val = data[i]
                max_idx = i
    
    if not has_valid:
        return np.nan
    
    # 返回距离当前位置的偏移：当前位置索引 - 最大值位置索引
    return float(len(data) - 1 - max_idx)

@njit
def _min_idx(data):
    """
    正确实现最小值索引，返回距离当前位置的偏移
    """
    if len(data) == 0:
        return np.nan
    
    min_val = np.inf
    min_idx = -1
    has_valid = False
    
    # 从原始数据中找最小值，跳过NaN
    for i in range(len(data)):
        if not np.isnan(data[i]):
            has_valid = True
            if data[i] < min_val:
                min_val = data[i]
                min_idx = i
    
    if not has_valid:
        return np.nan
    
    # 返回距离当前位置的偏移：当前位置索引 - 最小值位置索引
    return float(len(data) - 1 - min_idx)


# ============================
# ======= Numba加速函数 ========
# ============================


@njit(parallel=True)
def compute_basic_rolling(data, window, metric):
    """
    rolling方法通用的计算函数
    """
    n_rows, n_cols = data.shape
    result = np.full_like(data, np.nan)

    for col in prange(n_cols):
        for row in range(n_rows):
            if window == 0: # expanding窗口
                start_idx = 0
            else:
                # rolling 窗口
                start_idx = max(0, row - window + 1)

            window_slice = data[start_idx:row + 1, col]
            valid_data = window_slice[~np.isnan(window_slice)]

            if len(valid_data) == 0 or (metric in ['std', 'var'] and len(valid_data) <= 1):
                continue

            if metric == 'max':
                result[row, col] = np.max(valid_data)
            elif metric == 'min':
                result[row, col] = np.min(valid_data)
            elif metric == 'sum':
                result[row, col] = np.sum(valid_data)
            elif metric == 'mean':
                result[row, col] = np.mean(valid_data)
            elif metric == 'std':
                result[row, col] = np.std(valid_data)
            elif metric == 'var':
                result[row, col] = np.var(valid_data)
            elif metric == 'median':
                result[row, col] = np.median(valid_data)
            elif metric == 'count':
                result[row, col] = len(valid_data)
            elif metric == 'skew':
                result[row, col] = _compute_skew(valid_data)
            elif metric == 'kurt':
                result[row, col] = _compute_kurt(valid_data)
            elif metric == 'mad':
                result[row, col] = _compute_mad(valid_data)
            
    return result


# @njit(parallel=True)
# def compute_idx_rolling(data, window, metric):
#     n_rows, n_cols = data.shape
#     result = np.full_like(data, np.nan)

#     for col in prange(n_cols):
#         for row in range(n_rows):
#             if window == 0:  # expanding窗口
#                 start_idx = 0
#             else:
#                 start_idx = max(0, row - window + 1)

#             # 关键：直接使用原始窗口数据，不过滤NaN
#             window_slice = data[start_idx:row + 1, col]
            
#             if len(window_slice) == 0:
#                 continue

#             if metric == 'idxmax':
#                 result[row, col] = _max_idx(window_slice)
#             elif metric == 'idxmin':
#                 result[row, col] = _min_idx(window_slice)
        
#     return result


@njit(parallel=True)
def compute_idx_rolling_inline(data, window, metric):
    """完全内联版本，避免任何函数调用"""
    n_rows, n_cols = data.shape
    result = np.full_like(data, np.nan)

    for col in prange(n_cols):
        for row in range(n_rows):
            if window == 0:  # expanding窗口
                start_idx = 0
            else:
                start_idx = max(0, row - window + 1)

            window_slice = data[start_idx:row + 1, col]
            
            if len(window_slice) == 0:
                continue

            # 完全内联的逻辑，不调用任何函数
            if metric == 'idxmax':
                # 内联最大值索引查找
                max_val = -np.inf
                max_idx = -1
                has_valid = False
                
                for i in range(len(window_slice)):
                    if not np.isnan(window_slice[i]):
                        has_valid = True
                        if window_slice[i] > max_val:
                            max_val = window_slice[i]
                            max_idx = i
                
                if has_valid:
                    result[row, col] = float(len(window_slice) - 1 - max_idx)
                    
            elif metric == 'idxmin':
                # 内联最小值索引查找
                min_val = np.inf
                min_idx = -1
                has_valid = False
                
                for i in range(len(window_slice)):
                    if not np.isnan(window_slice[i]):
                        has_valid = True
                        if window_slice[i] < min_val:
                            min_val = window_slice[i]
                            min_idx = i
                
                if has_valid:
                    result[row, col] = float(len(window_slice) - 1 - min_idx)
        
    return result      


@njit(parallel=True)
def compute_weighted_rolling(data, window, weights):
    """
    weighted_rolling方法通用的计算函数
    """
    n_rows, n_cols = data.shape
    result = np.full_like(data, np.nan)

    # 预计算归一化权重,方阵， 直接读取矩阵就行
    all_weights = np.zeros((window, window))

    for i in range(1, window + 1):
        if weights[0] < 1: # EMA 权重必然小于1
            w = weights[window - i:] # EMA权重
        else:
            w = weights[:i] # WMA权重
        
        all_weights[i-1, :i] = w

    
    for col in prange(n_cols):
        for row in range(n_rows):
            start_idx = max(0, row - window + 1)
            win_len = min(row + 1, window)
            
            window_slice = data[start_idx:row+1, col]
            
            if len(window_slice) > 0:
                w = all_weights[win_len-1, :win_len]
                weighted_sum = np.nansum(w * window_slice) #nansum直接处理相加
                weight_sum = np.nansum(w * (~np.isnan(window_slice)).astype(np.float64)) # 非nan的权重相加
                
                if weight_sum > 0:  # 确保至少有一个有效数据
                    result[row, col] = weighted_sum / weight_sum
    
    return result


@njit(parallel=True)
def compute_regression(data, window, metric):
    """
    regression方法通用的计算函数
    """
    n_rows, n_cols = data.shape
    result = np.full_like(data, np.nan)

    for col in prange(n_cols):
        for row in range(n_rows):
            if window == 0:
                if row < 1:
                    continue
                start_idx = 0
            else:
                if row < window - 1:
                    continue
                start_idx = row - window + 1

            window_slice = data[start_idx:row+1, col]
            n = len(window_slice)
            
            # 找有效数据点
            valid_count = 0
            for i in range(n):
                if not np.isnan(window_slice[i]):
                    valid_count += 1
                
            if valid_count <= 1:
                continue

            # 明确指定dtype，避免类型推断问题
            valid_y = np.empty(valid_count, dtype=np.float64)
            valid_x = np.empty(valid_count, dtype=np.float64)
            last_valid_idx = -1

            j = 0
            for i in range(n):
                if not np.isnan(window_slice[i]):
                    valid_y[j] = float(window_slice[i])  # 明确转换为float
                    valid_x[j] = float(i + 1)
                    last_valid_idx = i
                    j += 1

            x_mean = np.mean(valid_x)
            y_mean = np.mean(valid_y)
            x_diff = valid_x - x_mean
            y_diff = valid_y - y_mean

            numerator = np.dot(x_diff, y_diff)
            denominator = np.dot(x_diff, x_diff)

            if abs(denominator) < 1e-10:
                continue

            slope = numerator / denominator

            if metric == 'slope':
                result[row, col] = slope
            elif metric == 'rsquare':
                tss = np.dot(y_diff, y_diff) 
                if tss < 1e-10:
                    result[row, col] = 0.0
                else:
                    alpha = y_mean - slope * x_mean
                    y_hat = alpha + slope * valid_x
                    residual = valid_y - y_hat
                    rss = np.dot(residual, residual)
                    r_squared = 1.0 - (rss / tss)
                    result[row, col] = r_squared
            elif metric == 'residual':
                alpha = y_mean - slope * x_mean
                last_x = float(last_valid_idx + 1)
                predicted = alpha + slope * last_x
                result[row, col] = float(window_slice[last_valid_idx]) - predicted
                
    return result

@njit(parallel=True)
def compute_correlation(values1, values2, window, metric):
    """
    corr方法通用的计算函数
    """
    n_rows, n_cols = values1.shape
    result = np.full_like(values1, np.nan)

    for col in prange(n_cols):
        for row in range(n_rows):
            if window == 0:
                start_idx = 0
            else:
                start_idx = max(0, row - window + 1)
            
            x = values1[start_idx:row+1, col]
            y = values2[start_idx:row+1, col]
            n = len(x)

            if n < 1:
                continue

            # 创建同时有效的mask
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            valid_count = np.sum(valid_mask)
            
            if valid_count < 2:  
                continue
            

            valid_x = x[valid_mask]
            valid_y = y[valid_mask]
            

            x_mean = np.mean(valid_x)
            y_mean = np.mean(valid_y)
            x_diff = valid_x - x_mean
            y_diff = valid_y - y_mean
            

            x_var = np.dot(x_diff, x_diff) / valid_count
            y_var = np.dot(y_diff, y_diff) / valid_count
            cov = np.dot(x_diff, y_diff) / valid_count
            
            x_std = np.sqrt(x_var)
            y_std = np.sqrt(y_var)
            
            if metric == 'corr':  # correlation
                if x_std < 2e-05 or y_std < 2e-05:
                    result[row, col] = np.nan
                else:
                    corr = cov / (x_std * y_std)
                    result[row, col] = corr 
                    
            elif metric == 'cov':  # covariance
                if x_std < 2e-05 or y_std < 2e-05:
                    result[row, col] = np.nan
                else:
                    result[row, col] = cov
    
    return result

# ============================
# ========== 基类 ============
# ============================
            

class OperatorBase:
    """操作符基类"""
    
    def __init__(self, df1, window=1):
        self.df1 = df1
        self.window = window
    
    def __str__(self):
        return f"{type(self).__name__}({self.df1}, {self.window})"


class Rolling(OperatorBase):
    """rolling方法基类"""
    
    def __init__(self, df1, window, metrics):
        super().__init__(df1, window)
        self.metrics = metrics
    
    def _load_internal(self):
        data = self.df1.values.astype(np.float64)
        result = compute_basic_rolling(data, self.window, self.metrics)
        return pd.DataFrame(result, index=self.df1.index, columns=self.df1.columns)
    

# class Idx(OperatorBase):  # 不再继承Rolling
#     def __init__(self, df1, window, metrics):
#         super().__init__(df1, window)
#         self.metrics = metrics
    
#     def _load_internal(self):
#         data = self.df1.data.astype(np.float64)
#         result = compute_idx_rolling(data, self.window, self.metrics)
#         return pd.DataFrame(result, index=self.df1.index, columns=self.df1.columns)

class Idx(OperatorBase):
    def __init__(self, df1, window, metrics):
        super().__init__(df1, window)
        self.metrics = metrics
    
    def _load_internal(self):
        if self.metrics == 'idxmax':
            if self.window == 0:
                return self.df1.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
            else:
                return self.df1.rolling(self.window, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        elif self.metrics == 'idxmin':
            if self.window == 0:
                return self.df1.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
            else:
                return self.df1.rolling(self.window, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)


class Rollingpd(OperatorBase):
    """rollingpd方法基类"""
    
    def __init__(self, df1, window, func):
        super().__init__(df1, window)
        self.func = func
    
    def _load_internal(self):
        if self.window == 0:
            return getattr(self.df1.expanding(min_periods=1), self.func)()
        else:
            return getattr(self.df1.rolling(self.window, min_periods=1), self.func)()

    
class WeightedRolling(OperatorBase):
    """WeightedRolling方法基类"""
    
    def __init__(self, df1, window, weight_generator):
        super().__init__(df1, window)
        self.weight_generator = weight_generator
    
    def _load_internal(self):
        data = self.df1.values.astype(np.float64)
        weights = self.weight_generator(self.window)
        result = compute_weighted_rolling(data, self.window, weights)
        return pd.DataFrame(result, index=self.df1.index, columns=self.df1.columns)
    

class RegressionRolling(OperatorBase):
    """RegressionRolling方法基类"""
    
    def __init__(self, df1, window, metric):
        super().__init__(df1, window)
        self.metric = metric
    
    def _load_internal(self):
        data = self.df1.values.astype(np.float64)
        result = compute_regression(data, self.window, self.metric)
        return pd.DataFrame(result, index=self.df1.index, columns=self.df1.columns)
    
class Correlation(OperatorBase):
    """Corr方法基类"""
    
    def __init__(self, df1, df2, window, metric):
        super().__init__(df1, window)
        self.df2 = df2 
        self.metric = metric
    
    def _load_internal(self):
        common_cols = self.df1.columns.intersection(self.df2.columns)
        if len(common_cols) == 0:
            raise ValueError("两个DataFrame没有公共列")
        
        values1 = self.df1[common_cols].values.astype(np.float64)
        values2 = self.df2[common_cols].values.astype(np.float64)
        result = compute_correlation(values1, values2, self.window, self.metric)
        return pd.DataFrame(result, index=self.df1.index, columns=common_cols)

import numpy as np

class IndustrialCrossSectionNeutralize(OperatorBase):
    """行业中性化算子（行业内Z-Score标准化）"""
    
    def __init__(self, df1, industry_data_path=r"indus/中信行业.pqt"):
        super().__init__(df1, window=None)
        self.industry_data_path = industry_data_path
        
    def _load_internal(self):
        # 加载行业数据
        industry_df = pd.read_parquet(self.industry_data_path)
        industry_df.index  = industry_df.index.astype(str)
        
        # 确保索引对齐
        common_dates = self.df1.index.intersection(industry_df.index)
        common_stocks = self.df1.columns.intersection(industry_df.columns)
        
        if len(common_dates) == 0 or len(common_stocks) == 0:
            return pd.DataFrame(np.nan, index=self.df1.index, columns=self.df1.columns)
        
        # 提取对齐后的数据
        factor_data = self.df1.loc[common_dates, common_stocks].values.astype(np.float64)
        industry_df_aligned = industry_df.loc[common_dates, common_stocks]
        
        # 将行业数据转换为整数编码 - 不使用numba
        industry_data_encoded = self._encode_industries_python(industry_df_aligned)
        
        # 进行行业内Z-Score标准化
        result = _compute_cross_section_zscore(factor_data, industry_data_encoded)
        
        # 构建完整结果DataFrame
        result_df = pd.DataFrame(np.nan, index=self.df1.index, columns=self.df1.columns)
        result_df.loc[common_dates, common_stocks] = result
        
        return result_df
    
    def _encode_industries_python(self, industry_df):
        """在Python中快速将行业数据转换为整数编码"""
        n_dates, n_stocks = industry_df.shape
        industry_codes = np.full((n_dates, n_stocks), -1, dtype=np.int32)
        
        for date_idx in range(n_dates):
            # 获取当前行的行业数据
            date_industries = industry_df.iloc[date_idx]
            
            # 使用factorize进行快速编码
            # pd.factorize会自动将NaN/None编码为-1
            codes, _ = pd.factorize(date_industries, sort=True)
            
            # 将编码存储到数组中
            industry_codes[date_idx, :] = codes.astype(np.int32)
        
        return industry_codes


@njit(parallel=True)
def _compute_cross_section_zscore(factor_data, industry_codes):
    """行业内Z-Score标准化 - numba并行版"""
    n_dates, n_stocks = factor_data.shape
    result = np.full_like(factor_data, np.nan)
    
    for date_idx in prange(n_dates):
        # 当前截面的数据
        factors = factor_data[date_idx]
        industries = industry_codes[date_idx]
        
        # 找到当前截面的最大行业编码
        max_industry = -1
        for i in range(n_stocks):
            industry = industries[i]
            if industry > max_industry:
                max_industry = industry
        
        if max_industry < 0:
            continue  # 没有有效的行业数据
        
        # 为每个行业收集数据
        industry_counts = np.zeros(max_industry + 1, dtype=np.int32)
        industry_sums = np.zeros(max_industry + 1, dtype=np.float64)
        industry_sums_sq = np.zeros(max_industry + 1, dtype=np.float64)
        
        for i in range(n_stocks):
            factor = factors[i]
            industry = industries[i]
            
            if not np.isnan(factor) and industry >= 0:
                industry_counts[industry] += 1
                industry_sums[industry] += factor
                industry_sums_sq[industry] += factor * factor
        
        # 计算每个行业的均值和标准差
        industry_means = np.zeros(max_industry + 1, dtype=np.float64)
        industry_stds = np.zeros(max_industry + 1, dtype=np.float64)
        #print(date_idx,industry_means,industry_stds)
        
        for industry in range(max_industry + 1):
            count = industry_counts[industry]
            if count > 1:
                mean_val = industry_sums[industry] / count
                # 计算标准差，避免数值问题
                if count > 0:
                    variance = (industry_sums_sq[industry] / count) - (mean_val * mean_val)
                    # 处理可能的数值误差导致的负方差
                    if variance < 0:
                        variance = 0.0
                    std_val = np.sqrt(variance)
                else:
                    std_val = 1.0
                industry_means[industry] = mean_val
                industry_stds[industry] = std_val if std_val > 1e-10 else 1e-10
            elif count == 1:
                # 只有一个样本，均值为该值，标准差设为1
                industry_means[industry] = industry_sums[industry]
                industry_stds[industry] = 1.0
        
        # 计算行业内Z-Score
        for i in range(n_stocks):
            factor = factors[i]
            industry = industries[i]
            
            if not np.isnan(factor) and industry >= 0 and industry_counts[industry] > 0:
                mean_val = industry_means[industry]
                std_val = industry_stds[industry]
                result[date_idx, i] = (factor - mean_val) / std_val
    
    return result
# =================================
# ========== 权重生成器 ============
# =================================

def generate_ema_weights(window):
    """生成EMA权重"""
    alpha = 2 / (window + 1)
    return np.power(1-alpha, np.arange(window)[::-1])


def generate_wma_weights(window):
    """生成WMA权重"""
    return np.arange(1, window+1, dtype=np.float64)

# =================================
# ========== 操作符实现 ============
# =================================

class Max(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'max')

class Min(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'min')

class Sum(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'sum')

class Mean(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'mean')

class Std(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'std')

class Var(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'var')

class Med(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'median')

class Count(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'count')

class IdxMax(Idx):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'idxmax')

class IdxMin(Idx):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'idxmin')

class Count(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'count')

class Skew(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'skew')

class Kurt(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'kurt')

class Mad(Rolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, "mad")

class Rank(Rollingpd):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, "rank")

class Quantile(Rollingpd):
    def __init__(self, df1, window=0, q=0.5):
        super().__init__(df1, window, "quantile")
        self.q = q
    
    def _load_internal(self):
        if self.window == 0:
            return self.df1.expanding(min_periods=1).quantile(self.q)
        else:
            return self.df1.rolling(self.window, min_periods=1).quantile(self.q)


class EMA(WeightedRolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, generate_ema_weights)

class WMA(WeightedRolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, generate_wma_weights)

class Slope(RegressionRolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'slope')

class Rsquare(RegressionRolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'rsquare')

class Resi(RegressionRolling):
    def __init__(self, df1, window=0):
        super().__init__(df1, window, 'residual')

class Corr(Correlation):
    def __init__(self, df1, df2, window=0):
        super().__init__(df1, df2, window, 'corr')

class Cov(Correlation):
    def __init__(self, df1, df2, window=0):
        super().__init__(df1, df2, window, 'cov')


# =================================
# ========== 运算符 ============
# =================================

class Operators:
    """定义因子计算中常用的操作函数"""

    @staticmethod
    def IndustrialCrossSectionNeutralize(df1):
        """行业中性化，计算该字段的按行业zscore的结果 """
        return IndustrialCrossSectionNeutralize(df1)._load_internal()
    
    @staticmethod
    def Ref(df1, window=1):
        """变量检索,
        N=0,检索第一个数据;
        N>0,检索 N 个周期前的数据;
        N<0,检索未来数据
        """
        return df1.shift(window)
    
    @staticmethod
    def Max(df1, window=0):
        """计算单字段滚动 N 个窗口期内最大值"""
        return Max(df1, window)._load_internal()
    
    @staticmethod
    def Min(df1, window=0):
        """计算单字段滚动 N 个窗口期内最小值"""
        return Min(df1, window)._load_internal()

    @staticmethod
    def IdxMax(df1, window=0):
        """计算单字段滚动N个窗口期内最大值的索引"""
        return IdxMax(df1, window)._load_internal()

    @staticmethod
    def IdxMin(df1, window=0):
        """计算单字段滚动N个窗口期内最小值的索引"""
        return IdxMin(df1, window)._load_internal()


    @staticmethod
    def Sum(df1, window=0):
        """计算单字段滚动 N 个窗口期求和"""
        return Sum(df1, window)._load_internal()
    
    @staticmethod
    def Mean(df1, window=0):
        """计算单字段滚动 N 个窗口期均值"""
        return Mean(df1, window)._load_internal()
    
    @staticmethod
    def Std(df1, window=0):
        """计算单字段滚动 N 个窗口期标准差"""
        return Std(df1, window)._load_internal()
    
    @staticmethod
    def Var(df1, window=0):
        """计算单字段滚动 N 个窗口期方差"""
        return Var(df1, window)._load_internal()
    
    @staticmethod
    def Skew(df1, window=0):
        """计算单字段滚动 N 个窗口期偏度"""
        return Skew(df1, window)._load_internal()
    
    @staticmethod
    def Kurt(df1, window=0):
        """计算单字段滚动 N 个窗口期峰度"""
        return Kurt(df1, window)._load_internal()
    
    @staticmethod
    def Med(df1, window=0):
        """计算单字段滚动 N 个窗口期中位数"""
        return Med(df1, window)._load_internal()
    
    @staticmethod
    def Mad(df1, window=0):
        """计算单字段滚动 N 个窗口期内和均值偏离的绝对值"""
        return Mad(df1, window)._load_internal()
    
    @staticmethod
    def Count(df1, window=0):
        """计算单字段滚动 N 个窗口期非空数值"""
        return Count(df1, window)._load_internal()
    
    @staticmethod
    def EMA(df1, window=0):
        """计算单字段滚动N个窗口期的指数加权平均"""
        return EMA(df1, window)._load_internal()
    
    @staticmethod
    def WMA(df1, window=0):
        """计算单字段滚动N个窗口期的加权移动平均"""
        return WMA(df1, window)._load_internal()
    
    @staticmethod
    def Slope(df1, window=0):
        """计算单字段与 T(1,2,3...)的滚动回归的回归系数项"""
        return Slope(df1, window)._load_internal()
    
    @staticmethod
    def Rsquare(df1, window=0):
        """计算单字段与 T(1,2,3...)的滚动回归的R方"""
        return Rsquare(df1, window)._load_internal()
    
    @staticmethod
    def Resi(df1, window=0):
        """计算单字段与 T(1,2,3...)的滚动回归的残差"""
        return Resi(df1, window)._load_internal()
    
    @staticmethod
    def Corr(df1, df2, window=0):
        """两个变量在滚动 N 个窗口期的相关性"""
        return Corr(df1, df2, window)._load_internal()
    
    @staticmethod
    def Cov(df1, df2, window=0):
        """两个变量在滚动 N 个窗口期的协方差"""
        return Cov(df1, df2, window)._load_internal()
    
    @staticmethod
    def Rank(df1, window=0):
        """计算单字段滚动 N 个窗口期排名"""
        return Rank(df1, window)._load_internal()
    
    @staticmethod
    def Quantile(df1, window=0, q=0.5):
        """计算单字段滚动 N 个窗口期百分位"""
        return Quantile(df1, window, q)._load_internal()
    
    
    @staticmethod
    def Abs(df1):
        """计算单字段的绝对值"""
        return df1.abs()
    
    @staticmethod
    def Sign(df1):
        """单字段大于 0 的值置为 1,小于 0 的值置为-1"""
        return np.sign(df1)
    
    @staticmethod
    def Log(df1):
        """计算单字段的自然对数"""
        result = df1.copy()
        result[result <= 0] = np.nan
        return np.log(result)
    
    @staticmethod
    def Power(df1, power):
        """计算单字段指定次幂"""
        return np.power(df1, power)
    
    @staticmethod
    def Delta(df1, window=1):
        """计算单字段在滚动 N 个窗口期的最后值减开始值"""
        return df1 - df1.shift(window)
    
    @staticmethod
    def Less(df1, df2):
        """返回两个字段的较小值"""
        index, columns = df1.index, df1.columns
        df1_arr, df2_arr = np.array(df1), np.array(df2)
        return pd.DataFrame(
            np.where(np.isnan(df1_arr), df2_arr, 
                    np.where(np.isnan(df2_arr), df1_arr, 
                            np.minimum(df1_arr, df2_arr))),
            index=index, columns=columns)
    
    @staticmethod
    def Greater(df1, df2):
        """返回两个字段的较大值"""
        index, columns = df1.index, df1.columns
        df1_arr, df2_arr = np.array(df1), np.array(df2)
        return pd.DataFrame(
            np.where(np.isnan(df1_arr), df2_arr, 
                    np.where(np.isnan(df2_arr), df1_arr, 
                            np.maximum(df1_arr, df2_arr))),
            index=index, columns=columns)
    
    @classmethod
    def get_all_method_info(cls):
        method_info = {}
        for name, obj in vars(cls).items():
            if callable(obj) and not name.startswith('__'):
                sig = inspect.signature(obj)
                params = [str(param) for param in sig.parameters.values()]
                doc =[inspect.getdoc(obj)]
                method_info[name]=(params, doc)
        return method_info


    @classmethod
    def get_all_method_names(cls):
        """
        获取类中所有非特殊方法的名称
        """
        method_names = []
        for name, obj in vars(cls).items():
            if callable(obj) and not name.startswith('__'):
                method_names.append(name)
        return method_names
    
class ExpressionCalculator:
    """处理因子表达式的计算"""
    
    def __init__(self, factor_dfs):
        """
        初始化计算器
        
        参数:
            factor_dfs: 包含所有因子数据的字典
        """
        # 创建计算环境
        self.env = {}
        
        # 添加所有因子数据
        for name, data in factor_dfs.items():
            self.env[name] = data
        
        # 添加所有操作函数
        for name, func in self._get_operators():
            self.env[name] = func
    
    def _get_operators(self):
        """获取所有操作函数"""
        ops = Operators()
        for name in dir(ops):
            # 排除私有方法和内置方法
            if not name.startswith('_'):
                yield name, getattr(ops, name)
    
    def calculate(self, expression):
        """计算单个表达式"""
        return eval(expression, {"__builtins__": {}}, self.env)
    
    def batch_calculate(self, expressions):
        """批量计算多个表达式"""
        if isinstance(expressions, list):
            return [self.calculate(expr) for expr in expressions]
        elif isinstance(expressions, dict):
            return {name: self.calculate(expr) for name, expr in expressions.items()}
        


