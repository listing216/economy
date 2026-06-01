# 因子表达式可解释性增强方案

## 1. 改造背景

原系统主要围绕因子生成、样本内评估、MMR 去相关、样本外验证展开。因子解释主要来自 LLM 生成文本，存在解释粒度粗、结构化不足、难以复盘的问题。

本次改造目标是在保留原始计算逻辑的前提下，为每个候选因子补充结构化解释信息，并允许解释性评分作为 MMR 的辅助筛选目标。

## 2. 改造目标

- 将因子表达式解析为 AST；
- 提取字段、算子、窗口、复杂度、子表达式；
- 引入字段和算子语义字典；
- 生成结构化解释；
- 计算 `interpretability_score` 和 `complexity_penalty`；
- 在 MMR 中可选加入解释性评分；
- 保存解释性 JSON 和 Markdown 报告。

## 3. 改造范围

### 新增模块

- `explainability/expression_ast.py`
- `explainability/semantics.py`
- `explainability/factor_explainer.py`
- `explainability/scoring.py`
- `explainability/report.py`

### 新增语义字典

- `explainability/semantics/field_semantics.yaml`
- `explainability/semantics/operator_semantics.yaml`

### 修改文件

- `factor_mining.py`
- `run.py`
- `method_config/alpha_jungle_mcts.yaml`
- `method_config/alpha_agent.yaml`
- `method_config/factor_mad.yaml`

## 4. 保留原始逻辑说明

本次改造不改变以下逻辑：

- 不改变 DSL 因子表达式计算逻辑；
- 不改变 IC、IR、分层收益、多空收益等指标计算逻辑；
- 不改变样本内和样本外时间切分；
- 不改变原始样本外验证条件；
- 不改变 baseline 因子读取方式；
- 不重写任何具体挖掘方法。

当 `explainability.enabled=false` 或未传入解释性配置时，MMR 仍使用原始公式：

```text
mmr_score = lambda_param * quality_score - (1 - lambda_param) * max_correlation
```

## 5. 改造后流程

```text
LLM / 挖掘方法生成候选因子
        ↓
样本内指标计算
        ↓
进入统一 MMR 入口 alpha_mmr_selection()
        ↓
解释性模块解析表达式 AST
        ↓
提取字段、算子、窗口、复杂度、子表达式
        ↓
语义字典生成结构化解释
        ↓
计算 interpretability_score 与 complexity_penalty
        ↓
MMR 使用 IC、相关性、解释性评分综合筛选
        ↓
样本外验证
        ↓
保存因子值、指标、解释性 JSON、Markdown 报告
```

## 6. 解释性字段

改造后，候选因子记录中新增以下字段：

| 字段 | 含义 |
|---|---|
| `explainability` | 完整结构化解释字典 |
| `interpretability_score` | 解释性评分，范围 0 到 1 |
| `complexity_penalty` | 复杂度惩罚，范围 0 到 1 |
| `semantic_tags` | 规则推断的金融语义标签 |
| `explainability_parse_status` | AST 解析状态 |

## 7. MMR 增强公式

当 `explainability.mmr.use_interpretability_score=true` 时，使用增强公式：

```text
mmr_score =
    quality_weight * quality_score
  + explainability_weight * interpretability_score
  - correlation_weight * max_correlation
  - complexity_weight * complexity_penalty
```

默认权重：

```yaml
quality_weight: 0.70
explainability_weight: 0.15
correlation_weight: 0.10
complexity_weight: 0.05
```

设计原则：IC 质量仍是主导，解释性只作为辅助目标，避免系统偏向复杂但偶然有效的表达式。

## 8. 输出结果

每个通过样本外验证并入库的因子会额外保存：

```text
explainability/
  cycle_xxxx_yyyymmdd_hhmmss/
    alpha_xxx_explainability.json
    alpha_xxx_report.md
```

其中 JSON 用于程序化分析，Markdown 用于人工复盘和汇报。

## 9. 验证方式

建议用以下命令做最小验证：

```bash
python run.py --method_config method_config/alpha_jungle_mcts.yaml --max_cycles 1
```

重点检查：

1. 日志中是否出现“MMR启用解释性评分”；
2. 最终 CSV 是否包含 `interpretability_score`、`semantic_tags` 等字段；
3. 是否生成 `explainability/*.json` 和 `*.md` 报告；
4. 设置 `explainability.enabled=false` 后，系统能否退回原始 MMR 逻辑。
