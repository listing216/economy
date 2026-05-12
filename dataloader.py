"""
dataloader.py — 量价数据加载工具

从指定目录批量读取 .pqt（parquet）文件，
返回 {字段名: DataFrame} 字典。

DataFrame 格式：
    - 行索引：交易日期字符串（YYYYMMDD），index.name = None
    - 列：股票代码（如 000001.SZ），columns.name = None
"""

import os
import pandas as pd
from tqdm import tqdm


def dataloader(data_path: str, start_date: str = None, end_date: str = None) -> dict:
    """
    批量加载目录下所有 .pqt 文件。

    Args:
        data_path:  数据目录路径
        start_date: 起始日期（含），格式 YYYYMMDD，None 表示不过滤
        end_date:   结束日期（含），格式 YYYYMMDD，None 表示不过滤

    Returns:
        dict: {字段名(不含扩展名): DataFrame}
    """
    factor_dfs = {}

    for file in tqdm(os.listdir(data_path), desc="数据读取"):
        if not file.endswith(".pqt"):
            continue
        try:
            factor_name = file.replace(".pqt", "")
            df = pd.read_parquet(os.path.join(data_path, file))
            df.index = df.index.astype(str)

            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]

            # 去除北交所股票（.BJ 后缀）
            df = df.loc[:, ~df.columns.str.endswith(".BJ")]

            factor_dfs[factor_name] = df
        except Exception as e:
            print(f"读取文件 {file} 出错: {e}")

    return factor_dfs
