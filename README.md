# SJTU LLM 因子挖掘项目

该目录提供一套面向 A 股量价因子挖掘的精简实现，聚焦股票因子生成、评估与验证主线。

项目现在显式分成两层配置：

- `method_config/*.yaml`：决定“怎么挖因子”
- `validation_config/*.yaml`：决定“怎么验因子”

这种分层方式可以在不改主流程的情况下，分别切换挖掘方法与验证口径。

---

## 目录结构

```text
reorg_proj/
├── config.py
├── dataloader.py
├── operators.py
├── evaluator.py
├── correlations.py
├── llm_client.py
├── factor_mining.py
├── metrics_hooks.py
├── method_utils.py
├── mining_methods.py
├── run.py
├── poolsel/
│   ├── value.pkl
│   └── duration.pkl
├── method_config/
│   ├── factor_mad.yaml
│   ├── alpha_agent.yaml
│   └── alpha_jungle_mcts.yaml
├── validation_config/
│   ├── full_domain.yaml
│   ├── value.yaml
│   └── duration.yaml
```

---

## 核心流程

默认主流程在 `run.py`，一轮循环分 3 步：

1. 样本内挖掘：由 `method_config` 指定具体方法
2. MMR 选择：兼顾效果和多样性
3. 样本外验证：验证通过后保存结果

底层执行分工：

- `operators.py`：DSL 算子与表达式计算
- `evaluator.py`：IC、分层收益、统计指标
- `factor_mining.py`：并行生成、评估、MMR
- `metrics_hooks.py`：额外验证指标 hook
- `mining_methods.py`：不同论文方法的样本内挖掘逻辑

---

## 当前验证体系

当前实现仅保留普通 forward IC，不包含 YoY 加权、bench-factor 增量 IC 等扩展指标。

内置 hook：

- `plain_ic_metrics`
  - 默认计算 `ic_5_mean`
  - 默认计算 `ic_20_mean`

对应实现位于 `metrics_hooks.py`。

YAML 里可以这样控制：

```yaml
metric_hooks:
  - plain_ic_metrics

metric_hook_params:
  plain_ic_metrics:
    horizons: [5, 20]
```

筛选规则在 `train_filter` 和 `validation_filter` 里定义。
如果 `train_filter` 留空，程序会默认继承 `validation_filter`，这样训练期和样本外验证默认使用同一套筛选阈值。
现在提供 3 个示例 profile：

- `full_domain.yaml`
  - 全市场口径
- `value.yaml`
  - 仅在 `poolsel/value.pkl` 股票池内计算绩效
- `duration.yaml`
  - 仅在 `poolsel/duration.pkl` 股票池内计算绩效

这 3 个 profile 都只启用两项指标：

- `ic_5_mean`
- `ic_20_mean`

这些 profile 也保留了注释掉的 `long_excret` 条目，按需取消注释即可启用。

---

## 股票池说明

`poolsel/` 目录下的 pickle 现在都表示股票池掩码：

- 非 NaN -> `True`
- NaN -> `False`

运行时如果某个 validation profile 配了 `poolsel_path`，训练期和样本外验证都会只在该股票池内计算 IC / 分层收益。

当前示例：

- `validation_config/value.yaml` -> `poolsel/value.pkl`
- `validation_config/duration.yaml` -> `poolsel/duration.pkl`

---

## 现有方法

样本内挖掘方法都挂在 `mining_methods.py`：

- `factor_mad`
  - 参考 FactorMAD 的双 agent debate / critique / correction
- `alpha_agent`
  - 参考 AlphaAgent 的 idea / factor / eval 分工
- `alpha_jungle_mcts`
  - 参考 Alpha Jungle 的 MCTS + FSA

这些方法共用同一套执行层和验证层，便于在统一口径下比较不同挖掘策略。

---

## 快速启动

安装依赖：

```bash
conda activate reorg_proj_py310
pip install openai pandas numpy numba joblib tqdm pyyaml pyarrow statsmodels matplotlib scipy
```

配置 API Key（首次运行前必做）：

本项目调用 DeepSeek API，需要在环境变量 config.py 中设置自己的 API Key。



API Key 申请地址：<https://platform.deepseek.com/>

运行示例：

```bash
conda activate reorg_proj_py310
cd reorg_proj

python run.py
python run.py --max_cycles 1

python run.py --method_config ./method_config/factor_mad.yaml
python run.py --method_config ./method_config/alpha_agent.yaml
python run.py --method_config ./method_config/alpha_jungle_mcts.yaml

python run.py --validation_config ./validation_config/full_domain.yaml
python run.py --validation_config ./validation_config/value.yaml
python run.py --validation_config ./validation_config/duration.yaml

python run.py --method_config ./method_config/factor_mad.yaml --validation_config ./validation_config/value.yaml
```

只检查初始化是否正常：

```bash
python run.py --validation_config ./validation_config/value.yaml --max_cycles 0
```

---
