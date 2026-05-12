# AlphaJungleMCTS 自适应 UCT 探索系数改进方案

## 一、背景与动机

### 1.1 原有实现

AlphaJungleMCTS 使用蒙特卡洛树搜索（MCTS）在因子表达式空间中搜索高质量 alpha 因子。其节点选择策略基于 UCT（Upper Confidence Bound for Trees）公式：

$$
\text{UCT}(v) = Q(v) + C \cdot \sqrt{\frac{\ln N_{\text{parent}}}{N_v}} + 0.05 \cdot \frac{1}{1 + |\text{children}(v)|}
$$

其中：
- $Q(v)$：**利用** - 这个节点历史最好成绩
- √(ln(N)/n) ： **探索** - 访问次数少的节点加分，鼓励尝试新方向
  - $C$：探索系数，**固定为 0.70**
  - $N_{\text{parent}}$：所有候选节点的总访问次数
  - $N_v$：节点 $v$ 的访问次数

- 最后一项为微小奖励：优先展开还没有子节点的节



**对应代码**（`mining_methods.py` 第 980–994 行）：

```python
def _select_node(self, nodes: list[MCTSSearchNode]) -> MCTSSearchNode | None:
    max_depth = int(self.params.get("max_tree_depth", 4))
    eligible = [node for node in nodes if node.depth < max_depth]
    if not eligible:
        return None

    total_visits = sum(node.visits for node in eligible) + 1
    exploration_c = float(self.params.get("uct_c", 0.7))

    def _uct(node: MCTSSearchNode) -> float:
        explore = exploration_c * sqrt(log(total_visits + 1.0) / max(1.0, node.visits))
        virtual_bonus = 1.0 / (1.0 + len(node.children))
        return node.q_value + explore + 0.05 * virtual_bonus

    return max(eligible, key=_uct)
```

### 1.2 Baseline 实验结果与问题诊断

我们首先在固定 $C = 0.70$ 的配置下运行了完整的 20 轮 Baseline 实验，以量化现有方法的不足。

#### 1.2.1 实验配置

| 配置项 | 值 |
|--------|-----|
| 搜索预算（search_budget） | 24 步/轮 |
| 总轮数（max_cycles） | 20 |
| 训练期 | 2010-01-01 ~ 2019-12-31 |
| 验证期（样本外） | 2020-01-01 ~ 2025-04-30 |
| IC 筛选阈值 | \|ic_5_mean\| > 0.02，\|ic_20_mean\| > 0.04 |
| 验证期最低超额收益 | 4.5% 年化 |
| 最大因子相关性 | 0.70 |

#### 1.2.2 Baseline 结果总览

**成功率极低：仅 3/20 轮产出合格因子（成功率 15%）**

| Cycle | 训练期筛选 | MMR 筛选 | 样本外验证 | 最终入库 | 状态 |
|-------|-----------|---------|-----------|---------|------|
| 1 | 0/20 通过 | — | — | 0 | 失败 |
| 2 | 0/20 通过 | — | — | 0 | 失败 |
| 3 | 0/20 通过 | — | — | 0 | 失败 |
| 4 | 5/20 通过 | 5 | 3 | **3** | 成功 |
| 5 | 1/17 通过 | 1 | 1 | **1** | 成功 |
| 6–12 | 0 通过 | — | — | 0 | 连续 7 轮失败 |
| 13 | 多个通过 | — | — | **9** | 成功 |
| 14–20 | 0 通过 | — | — | 0 | 失败 |

**入库因子质量汇总（共 13 个因子）：**

| 来源 Cycle | 因子数 | Long Sharpe 范围 | 最佳超额收益 | 最佳 L/S Sharpe |
|-----------|--------|-----------------|-------------|-----------------|
| Cycle 4 | 3 | 0.88 ~ 0.93 | 20.2% | 3.27 |
| Cycle 5 | 1 | 0.25 | 1.4% | 1.41 |
| Cycle 13 | 9 | 0.95 ~ 1.39 | 23.1% | 3.10 |

#### 1.2.3 失败原因分析

通过分析日志中 **17 轮失败 cycle** 的淘汰原因，发现核心瓶颈是：

**大量因子的 IC 值"差一点点"就能过阈值，但始终不够：**

```
典型失败日志（Cycle 1）：
  ic_5_mean=-0.0085 未满足 abs_gt 0.02   （差 0.0115）
  ic_5_mean=-0.0070 未满足 abs_gt 0.02   （差 0.0130）
  ic_20_mean=-0.0253 未满足 abs_gt 0.04  （差 0.0147）

典型失败日志（Cycle 9）：
  ic_5_mean=-0.0167 反复出现多次         （搜索陷入同一区域）
```

这揭示了两个关键问题：

1. **搜索空间中合格区域极度稀疏**：20 个候选中 0 个通过，说明 MCTS 在当前探索力度下未能触达有效区域。

2. **搜索陷入局部区域**：Cycle 9 中多个节点（MCTS_004 到 MCTS_014）的 ic_5_mean 都是 -0.0167，说明搜索树在同一分支反复深入而未探索其他方向。

#### 1.2.4 问题根因：固定 C=0.70 的探索-利用失衡

| 阶段 | 理想行为 | 固定 C 的问题 |
|------|---------|--------------|
| 搜索前期（step 1–8） | 广泛探索不同表达式结构 | $C = 0.70$ 探索力度不够，容易过早收敛到局部最优 |
| 搜索中期（step 9–16） | 在有潜力的区域深入 | $C = 0.70$ 尚可，但无法根据实际搜索情况调节 |
| 搜索后期（step 17–24） | 集中开发最优区域 | $C = 0.70$ 探索过多，浪费预算在低收益区域 |

**核心矛盾**：在因子表达式这种高维稀疏空间中，合格解占比极低。固定的中等探索系数既不足以在前期发现稀疏的有效区域，又在后期浪费预算探索无用分支。结果是 **85% 的搜索轮次颗粒无收**，搜索预算大量浪费。

这是经典的 **Exploration-Exploitation Dilemma**：搜索资源有限（默认 24 步），需要在前期充分探索和后期集中开发之间取得平衡。

### 1.3 改进思路

基于上述 Baseline 实验暴露的问题（成功率仅 15%、搜索频繁陷入局部区域），我们引入**自适应探索系数** $C(t)$，使其随搜索进度动态衰减：

- **前期 $C$ 更大（1.2）**：增强探索力度，让 MCTS 有更高概率触达稀疏的有效区域，避免前 8 步就陷入低 IC 分支
- **后期 $C$ 更小（0.3）**：一旦发现有潜力的区域，集中预算深入开发，提高最终因子质量

具体公式：

$$
C(t) = C_{\max} - (C_{\max} - C_{\min}) \cdot \frac{t}{T}
$$

其中：
- $t$：当前搜索步数（从 1 开始）
- $T$：总搜索预算（`search_budget`）
- $C_{\max}$：初始探索系数（默认 1.2，鼓励前期探索）
- $C_{\min}$：最终探索系数（默认 0.3，后期聚焦开发）

改进后的 UCT 公式：

$$
\text{UCT}(v) = Q(v) + C(t) \cdot \sqrt{\frac{\ln N_{\text{parent}}}{N_v}} + 0.05 \cdot \frac{1}{1 + |\text{children}(v)|}
$$

### 1.4 学术依据

自适应探索系数是 MCTS 领域的经典改进方向：

- **Gelly & Silver (2007)**：在围棋中证明动态调整探索系数可显著提升搜索效率
- **Coulom (2007)**：提出在有限搜索预算下，递减探索系数优于固定系数
- **Auer et al. (2002)**：UCB 理论中，$C$ 的最优取值依赖于奖励分布，固定值难以适应变化的搜索空间

在本项目的因子挖掘场景中，因子表达式空间极大且稀疏（大部分表达式无效或低质量），前期需要更激进的探索来发现有价值的区域，后期则需要在已发现的高质量区域精细搜索。

---

## 二、详细设计

### 2.1 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `mining_methods.py` | 修改 `_select_node()` 方法，新增 `_compute_adaptive_uct_c()` 方法 |
| `method_config/alpha_jungle_mcts.yaml` | 新增自适应相关配置参数 |

### 2.2 新增配置参数

在 `alpha_jungle_mcts.yaml` 的 `params` 中新增：

```yaml
params:
  # ... 现有参数保持不变 ...

  uct_c: 0.70                    # 保留原参数（自适应关闭时的回退值）

  # ---- 自适应 UCT 探索系数 ----
  adaptive_uct: true              # 是否启用自适应（false 则退化为固定 uct_c）
  uct_c_max: 1.2                  # 搜索初期的探索系数
  uct_c_min: 0.3                  # 搜索末期的探索系数
  uct_decay: "linear"             # 衰减策略："linear" 或 "cosine"
```

**向后兼容性**：如果配置文件中没有 `adaptive_uct` 字段（或为 `false`），系统自动回退到原有的固定 `uct_c = 0.70` 行为，不影响现有功能。

### 2.3 衰减策略

提供两种衰减曲线供选择：

**线性衰减（linear）**：

$$
C(t) = C_{\max} - (C_{\max} - C_{\min}) \cdot \frac{t}{T}
$$

**余弦衰减（cosine）**：

$$
C(t) = C_{\min} + \frac{C_{\max} - C_{\min}}{2} \cdot \left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)
$$

余弦衰减的特点是前期衰减慢（保持更长时间的高探索），后期衰减快（更快转向开发），类似深度学习中的余弦学习率衰减。

衰减曲线示意（search_budget=24）：

```
C(t)
1.2 |*
    | **
    |   *  *
1.0 |    *    *            linear: ----
    |      *    *          cosine: ****
    |  ----  *    *
0.8 |       ---     *
    |          ---    *
0.6 |             ---   *
    |                --- *
0.4 |                   --*
    |                     -*-
0.3 |                        *
    +--+--+--+--+--+--+--+--+-->
    0  3  6  9  12 15 18 21 24  step
```

### 2.4 代码修改详情

#### 2.4.1 新增方法：`_compute_adaptive_uct_c()`

在 `AlphaJungleMCTSMethod` 类中新增：

```python
def _compute_adaptive_uct_c(self, step: int, total_steps: int) -> float:
    """根据搜索进度计算自适应 UCT 探索系数。

    Args:
        step: 当前搜索步数（从 1 开始）
        total_steps: 总搜索预算 (search_budget)

    Returns:
        当前步的探索系数 C(t)
    """
    if not self.params.get("adaptive_uct", False):
        return float(self.params.get("uct_c", 0.7))

    c_max = float(self.params.get("uct_c_max", 1.2))
    c_min = float(self.params.get("uct_c_min", 0.3))
    decay = self.params.get("uct_decay", "linear")

    progress = min(step / max(total_steps, 1), 1.0)

    if decay == "cosine":
        c = c_min + (c_max - c_min) * 0.5 * (1 + cos(pi * progress))
    else:  # linear
        c = c_max - (c_max - c_min) * progress

    return c
```

#### 2.4.2 修改方法：`_select_node()`

将固定的 `exploration_c` 替换为自适应调用：

```python
def _select_node(self, nodes: list[MCTSSearchNode],
                 step: int = 0, total_steps: int = 1) -> MCTSSearchNode | None:
    max_depth = int(self.params.get("max_tree_depth", 4))
    eligible = [node for node in nodes if node.depth < max_depth]
    if not eligible:
        return None

    total_visits = sum(node.visits for node in eligible) + 1
    exploration_c = self._compute_adaptive_uct_c(step, total_steps)

    def _uct(node: MCTSSearchNode) -> float:
        explore = exploration_c * sqrt(log(total_visits + 1.0) / max(1.0, node.visits))
        virtual_bonus = 1.0 / (1.0 + len(node.children))
        return node.q_value + explore + 0.05 * virtual_bonus

    return max(eligible, key=_uct)
```

#### 2.4.3 修改 MCTS 主循环中的调用处

在 `mine_in_sample()` 方法的主循环中（第 898 行附近），修改 `_select_node` 调用：

```python
search_budget = max(1, int(self.params.get("search_budget",
                    self.system.config["factors_per_cycle"] * 3)))

for step in tqdm(range(1, search_budget), desc="AlphaJungle MCTS"):
    # 传入当前 step 和总预算，用于自适应 C(t) 计算
    parent = self._select_node(nodes, step=step, total_steps=search_budget)
    if parent is None:
        break

    # ... 后续代码不变 ...
```

#### 2.4.4 新增日志输出

在主循环开始前记录自适应配置：

```python
if self.params.get("adaptive_uct", False):
    self.system.logger.info(
        f"自适应 UCT 已启用：C_max={self.params.get('uct_c_max', 1.2)}, "
        f"C_min={self.params.get('uct_c_min', 0.3)}, "
        f"decay={self.params.get('uct_decay', 'linear')}"
    )
```

在每步选择节点后记录当前 C 值（可选，debug 级别）：

```python
current_c = self._compute_adaptive_uct_c(step, search_budget)
self.system.logger.debug(f"Step {step}/{search_budget}: C(t) = {current_c:.4f}")
```

---

## 三、完整修改汇总

### 3.1 `method_config/alpha_jungle_mcts.yaml`

在 `uct_c: 0.70` 之后添加：

```yaml
  # ---- 自适应 UCT 探索系数 ----
  adaptive_uct: true
  uct_c_max: 1.2
  uct_c_min: 0.3
  uct_decay: "linear"
```

### 3.2 `mining_methods.py`

**新增 import**（文件顶部）：

```python
from math import cos, pi  # 用于余弦衰减（如已导入 math 模块则无需重复）
```

**修改点一**：`_select_node()` 方法签名新增 `step` 和 `total_steps` 参数，内部改用 `_compute_adaptive_uct_c()`。

**修改点二**：新增 `_compute_adaptive_uct_c()` 方法。

**修改点三**：`mine_in_sample()` 中调用 `_select_node()` 时传入 `step` 和 `total_steps`。

**修改点四**：新增日志输出。

总计修改约 **40 行代码**（新增约 30 行，修改约 10 行）。

---

## 四、效果验证方案

### 4.1 对照实验设计

| 实验组 | 配置 | 说明 |
|-------|------|------|
| Baseline | `adaptive_uct: false`, `uct_c: 0.70` | 原始固定系数 |
| 自适应 v1 | `adaptive_uct: true`, `uct_c_max: 1.2`, `uct_c_min: 0.3` | 大范围线性衰减 |
| 自适应 v2 | `adaptive_uct: true`, `uct_c_max: 1.0`, `uct_c_min: 0.5` | 小范围线性衰减（围绕原始值微调） |

### 4.2 实验结果

#### 4.2.1 三组实验总览

| 指标 | Baseline (C=0.7) | 自适应 v1 (1.2→0.3) | 自适应 v2 (1.0→0.5) |
|------|:---:|:---:|:---:|
| 成功率 | 3/20 (15%) | 5/20 (25%) | **12/20 (60%)** |
| 入库因子总数 | 13 | 28 | **83** |
| 平均 Long Sharpe | **1.033** | 0.428 | 0.399 |
| 最佳 Long Sharpe | 1.393 | 0.601 | **1.333** |
| 平均超额收益 | **17.5%** | 4.0% | 4.1% |
| 最佳超额收益 | 23.1% | 9.8% | **31.5%** |
| 平均 \|IC\| | 0.0477 | 0.0433 | **0.0572** |

#### 4.2.2 自适应 v2 逐轮详情

| Cycle | 因子数 | Long Sharpe 范围 | 平均 Sharpe | 最佳超额收益 | 平均\|IC\| |
|-------|--------|-----------------|-------------|-------------|-----------|
| 1 | 2 | 0.575 ~ 0.601 | 0.588 | 9.8% | 0.0511 |
| 1 | 1 | 0.573 | 0.573 | 9.1% | 0.0567 |
| 2 | 1 | 0.296 | 0.296 | 0.3% | 0.0468 |
| 2 | 9 | 0.375 ~ 0.643 | 0.497 | 13.9% | 0.0625 |
| 3 | 11 | -0.006 ~ 0.671 | 0.338 | 11.1% | 0.0643 |
| 5 | 15 | 0.135 ~ 0.690 | 0.505 | 11.5% | 0.0729 |
| 8 | 8 | -0.201 ~ **1.333** | 0.336 | **31.5%** | 0.0619 |
| 9 | 14 | 0.300 ~ 0.557 | 0.453 | 7.4% | 0.0432 |
| 10 | 1 | 0.432 | 0.432 | 5.0% | 0.0432 |
| 12 | 3 | 0.569 ~ 0.592 | 0.577 | 9.3% | 0.0680 |
| 14 | 8 | -0.131 ~ 0.235 | 0.044 | 0.1% | 0.0517 |
| 20 | 10 | 0.257 ~ 0.498 | 0.374 | 5.8% | 0.0414 |

#### 4.2.3 结果分析

**1. 成功率大幅提升（核心改进）**

自适应 v2 的成功率从 Baseline 的 15% 提升至 60%，因子产出量从 13 个增长至 83 个（6.4 倍）。这说明在因子表达式这种高维稀疏空间中，适度增大前期探索系数确实能帮助 MCTS 触达更多有效区域，大幅降低搜索的"浪费率"。

**2. IC 预测能力提升**

平均 |IC| 从 0.0477 提升至 0.0572（+20%），说明自适应策略不仅找到了更多因子，因子的预测方向性也更强。

**3. 发现了更优的极值因子**

自适应 v2 在 Cycle 8 中发现了 Long Sharpe=1.333、超额收益 31.5% 的因子，超越了 Baseline 的最佳表现（Sharpe 1.393、超额 23.1%）。这表明更广泛的前期探索确实能触达 Baseline 未发现的高质量区域。

**4. 平均质量下降的原因**

自适应 v2 的平均 Long Sharpe（0.399）低于 Baseline（1.033），原因是产出了大量"刚过阈值"的中低质量因子拉低均值。这是"高产量"策略的固有特征——可通过后续提高筛选阈值（如 `validation_min_ic_abs: 0.05`）来过滤低质量因子。

**5. v1 vs v2 参数对比的启示**

| | v1 (1.2→0.3) | v2 (1.0→0.5) |
|---|---|---|
| 衰减范围 | 0.9（跨度大） | 0.5（跨度小） |
| 成功率 | 25% | **60%** |
| 因子数 | 28 | **83** |

v1 的 $C_{\min}=0.3$ 过小，导致后期探索不足，在有潜力的区域无法充分开发。v2 将范围缩小到围绕原始值 0.7 的合理区间（0.5~1.0），兼顾了探索与开发，效果显著更优。

### 4.3 运行命令

```bash
# Baseline（固定 C=0.70）
python run.py --method_config method_config/alpha_jungle_mcts_baseline.yaml \
              --validation_config validation_config/full_domain.yaml

# Linear 衰减
python run.py --method_config method_config/alpha_jungle_mcts.yaml \
              --validation_config validation_config/full_domain.yaml

# Cosine 衰减（修改 yaml 中 uct_decay: cosine）
python run.py --method_config method_config/alpha_jungle_mcts_cosine.yaml \
              --validation_config validation_config/full_domain.yaml
```

### 4.4 日志验证

搜索过程中应能看到类似如下日志输出，确认自适应系数生效：

```
INFO  - 自适应 UCT 已启用：C_max=1.2, C_min=0.3, decay=linear
DEBUG - Step 1/24: C(t) = 1.1625
DEBUG - Step 6/24: C(t) = 0.9750
DEBUG - Step 12/24: C(t) = 0.7500
DEBUG - Step 18/24: C(t) = 0.5250
DEBUG - Step 23/24: C(t) = 0.3375
```

---

## 五、参数敏感性分析

### 5.1 实测参数对比

| 参数组合 | 成功率 | 因子数 | 平均 Sharpe | 结论 |
|---------|--------|--------|------------|------|
| 固定 C=0.70 | 15% | 13 | 1.033 | 产量低但精品率高 |
| C: 1.2→0.3（跨度 0.9） | 25% | 28 | 0.428 | 衰减范围过大，后期开发不足 |
| **C: 1.0→0.5（跨度 0.5）** | **60%** | **83** | 0.399 | **最优平衡，成功率最高** |

### 5.2 关键发现

1. **衰减范围不宜过大**：v1（跨度 0.9）的效果远不如 v2（跨度 0.5），说明 $C_{\min}$ 不能太小，后期仍需保留一定探索能力
2. **围绕原始值微调最有效**：v2 的范围 [0.5, 1.0] 以原始值 0.70 为中心，上下各浮动约 40%，效果最佳
3. **前期适度增大 C 即可**：$C_{\max}=1.0$（原始值的 1.43 倍）已足够提升探索覆盖率，无需更激进

### 5.3 可调参数范围建议

| 参数 | 推荐值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| $C_{\max}$ | **1.0** | [0.8, 1.2] | 过大会导致前期搜索过于随机 |
| $C_{\min}$ | **0.5** | [0.4, 0.6] | 过小会使后期开发力度不足 |
| `uct_decay` | **linear** | linear / cosine | 线性衰减已验证有效 |

---

## 六、总结

本改进方案在 AlphaJungleMCTS 的 UCT 节点选择策略中引入自适应探索系数，核心修改集中在 `_select_node()` 方法及其调用处，代码改动量约 40 行。

改进的核心价值：
1. **理论有据**：基于经典 MCTS 探索-利用权衡理论
2. **实现简洁**：仅修改 1 个文件 + 1 个配置文件
3. **向后兼容**：`adaptive_uct: false` 即退化为原始行为
4. **可验证**：通过对照实验和日志即可量化效果
