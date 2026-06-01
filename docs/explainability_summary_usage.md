# 因子解释性统计汇总说明

## 1. 改造目的

前两轮改造已经为单个因子生成解释性结构，包括 AST、字段、算子、窗口、语义标签、复杂度和 `interpretability_score`。第三步的目标是把单因子报告提升为实验级统计汇总，便于回答以下问题：

- 本次实验共有多少因子生成了解释性记录；
- AST 解析成功率是多少；
- 高 IC 因子是否也具有较高解释性；
- 哪些字段、算子和语义标签最常出现；
- 自适应 UCT 是否导致表达式复杂度上升；
- 是否存在 metrics 有记录但缺少 explainability 报告的因子。

## 2. 新增文件

```text
scripts/summarize_explainability_results.py
```

该脚本不修改原始挖掘流程，只在运行结束后扫描已有结果目录。

## 3. 输入目录

默认扫描：

```text
results/explainability/**/*_explainability.json
results/metrics/**/*.{csv,json,jsonl}
```

如果项目结果目录不是 `results`，可通过 `--results-dir` 指定。

## 4. 输出文件

默认输出到 `results/`：

```text
results/explainability_summary.csv
results/explainability_summary.json
results/explainability_summary.md
```

其中：

- `csv`：适合 Excel、pandas、后续画图；
- `json`：保留结构化字段，便于程序继续处理；
- `md`：适合直接放入汇报文档。

## 5. 使用方式

在项目根目录执行：

```bash
python scripts/summarize_explainability_results.py
```

指定结果目录：

```bash
python scripts/summarize_explainability_results.py --results-dir results
```

指定输出目录：

```bash
python scripts/summarize_explainability_results.py --results-dir results --out-dir results/summary
```

## 6. 汇总字段

核心字段包括：

```text
alpha_id
cycle
expression
parse_status
interpretability_score
complexity_penalty
num_nodes
max_depth
num_fields
num_operators
fields
operators
windows
semantic_tags
ic_mean
ic_ir
long_excret
long_sharpe
ic_5_mean
ic_20_mean
source_explainability_file
source_metrics_file
```

如果 metrics 文件中存在额外字段，脚本会尽量保留。

## 7. 结果解读

重点查看：

1. `parse_status` 是否大多为 `success`；
2. `interpretability_score` 是否集中在合理区间；
3. `complexity_penalty` 是否能识别复杂表达式；
4. `fields` 和 `operators` 是否过度集中；
5. `semantic_tags` 是否过于单一；
6. 高 `long_sharpe` / 高 `ic_mean` 因子是否同时具有较好的解释性。

## 8. 下一步用途

该汇总表可用于下一阶段实验：

- 对比固定 UCT 与自适应 UCT 的表达式复杂度；
- 比较解释性 MMR 开启前后的入选因子质量；
- 设置解释性硬过滤阈值；
- 生成汇报中的字段分布、算子分布、解释性评分分布表。
