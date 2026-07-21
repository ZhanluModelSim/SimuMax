<p align="center">
  <a href="design_simu_fsdp2_gap_analysis.md">English</a>|
  <a href="design_simu_fsdp2_gap_analysis-zh.md">中文版本</a>
</p>

# FSDP2 建模 vs 实际行为：比对分析与修正方案

- Status: **Proposal (待讨论)**
- Date: 2026-07-21
- Reference: [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- Scope: 当前 layer-wise FSDP 建模与 FSDP2 实际行为的逐项比对，识别差异，
  设计修正方案。

---

## 1. FSDP2 实际行为总结

基于 [官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)，
FSDP2 的核心行为如下。

### 1.1 通信原语

| 阶段 | 操作 | 通信类型 |
|------|------|---------|
| Forward 前 | unshard params | all-gather |
| Forward 后 | reshard params | 无通信（释放显存） |
| Backward 前 | unshard params（仅 FULL_SHARD） | all-gather |
| Backward 内 | shard grads | reduce-scatter |

### 1.2 Sharding Strategy

| Strategy | `reshard_after_forward` | Forward 后 | Backward 前需 AG？ |
|----------|------------------------|-----------|-------------------|
| **FULL_SHARD**（默认） | True | 参数释放 | 是（重新 unshard） |
| **SHARD_GRAD_OP** | False | 参数保留 | 否（仍 unsharded） |

### 1.3 隐式预取（核心性能特性）

**Forward：**
- CPU thread 在 layer i 的 forward 前发起 `all-gather(i+1)`，进入独立 CUDA stream
- `compute(i)` 在 default stream 运行
- → `AG(i+1)` 与 `compute(i)` 重叠
- 第一个 `AG(0)` **完全暴露**（无前驱），除非显式 `model.unshard()` 提前发起

**Backward（逆序）：**
- Backward 的 all-gather 按 **forward 的逆序** 发起
- `AG(i-1)` 与 `compute_bwd(i)` 重叠
- 第一个 backward block（= 最后一个 forward block）的 AG **完全暴露**

### 1.4 完整时间线

**FULL_SHARD + 隐式预取：**
```
Forward:
  L0: [post AG(0)][wait AG(0)][post AG(1)][compute_fwd(0)]
  L1:                       [wait AG(1)][post AG(2)][compute_fwd(1)]
  ...

Backward (逆序):
  LN: [post AG(N)][wait AG(N)][post AG(N-1)][compute_bwd(N)][post RS(N)]
  LN-1:                       [wait AG(N-1)][post AG(N-2)][compute_bwd(N-1)][post RS(N-1)]
  ...
  L0:                                          [wait AG(0)][compute_bwd(0)][post RS(0)]

Optimizer:
  [wait all RS complete][optim_step]
```

`RS(N)` 与 `compute_bwd(N-1)` 重叠；`AG(N-1)` 与 `compute_bwd(N)` 重叠。

**SHARD_GRAD_OP：**
```
Forward: 同 FULL_SHARD（AG + compute，无 reshard）
Backward:
  LN: [compute_bwd(N)][post RS(N)]
  LN-1:   [RS(N) overlaps][compute_bwd(N-1)][post RS(N-1)]
  ...
  L0: [compute_bwd(0)][post RS(0)]
Optimizer:
  [wait all RS complete][optim_step]
```
Backward 不需要 AG（参数仍 unsharded），只发 RS。

### 1.5 显式预取

- `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`
- 可预取 **2+ 层**：`AG(i+1) + AG(i+2)` 同时与 `compute(i)` 重叠
- 代价：更高显存（同时 unshard 多层参数）

---

## 2. 当前建模与实际行为的逐项比对

### 2.1 比对总表

| # | FSDP2 实际行为 | 当前建模 | 差异 | 严重度 |
|---|---------------|---------|------|--------|
| 1 | Forward: AG → compute → reshard(no-op) | `[wait AG(this)] → [post AG(next)] → [compute]` | 一致（reshard 是 no-op） | ✅ |
| 2 | Forward 预取: AG(i+1) 与 compute(i) 重叠 | `post AG(next)` 在 dp_comm lane 并行 | 一致 | ✅ |
| 3 | 首层 AG 暴露 | 首块 `[post AG(0)] → [wait AG(0)]` 无重叠窗口 | 一致 | ✅ |
| 4 | **FULL_SHARD: Backward 前需 AG** | `async_all_gather` 只有 `fwd_cost`，bwd 阶段 no-op | **缺失 backward AG** | 🔴 严重 |
| 5 | **FULL_SHARD vs SHARD_GRAD_OP 区分** | 无 `reshard_after_forward` 配置 | **缺少策略区分** | 🔴 严重 |
| 6 | Backward RS: compute → post RS (async, 不阻塞下一层) | `[compute] → [wait RS(next)] → [post RS(this)]` | **多了 wait RS(next) 同步点** | 🟡 中等 |
| 7 | Backward AG 预取: AG(i-1) 与 compute_bwd(i) 重叠 | 无 backward AG → 无预取 | 缺失（同 #4） | 🔴 严重 |
| 8 | 末层 RS 完成需在 optimizer 前同步 | layer-wise tail 只有 optim_step，无 RS 完成等待 | RS 可能不同步 | 🟡 中等 |
| 9 | 显式预取 2+ 层 | 只预取 1 层 | 缺少多层预取 | 🟢 低 |
| 10 | `model.unshard()` 缓解首层暴露 | 无机制 | 缺少优化选项 | 🟢 低 |

### 2.2 关键差异详解

#### 差异 #4/#5/#7：Backward 缺失 all-gather（FULL_SHARD 模式）

**FSDP2 实际**：在 `FULL_SHARD`（默认策略）下，forward 后参数被 reshard。
Backward 前需要 **重新 all-gather** 来 unshard 参数：

```
Backward of layer i:  AG(i) → compute_bwd(i) → RS(i)
                     AG(i-1) prefetched during compute_bwd(i)
```

**当前建模**：`async_all_gather` 只有 `fwd_cost`，backward 阶段 `_bwd()`
返回 True（no-op）。`async_reduce_scatter` 只有 `bwd_cost`，forward 阶段
`_step()` 返回 True（no-op）。

**结论**：当前建模隐式只覆盖了 `SHARD_GRAD_OP` 策略（forward 后不 reshard，
backward 不需要 AG），但缺少 `reshard_after_forward` 配置项来区分两种策略。
用户无法选择 `FULL_SHARD`（FSDP2 的默认策略）。

#### 差异 #6：Backward 多了 wait RS(next) 同步点

**FSDP2 实际**：RS 在 backward compute 之后 **异步 post**，在独立 comm
stream 上运行，**不阻塞下一层的 backward compute**。RS 的完成只在 optimizer
step 前需要同步。

**当前建模**：
```
compute_bwd(i) → wait RS(i+1) → post RS(i)
```

`wait RS(next)` 在两层 backward 之间插入了一个同步点。这个 wait 检查前一层
backward post 的 RS 是否完成——这在真实 FSDP2 中 **不存在**。RS 在
`dp_comm` queue 里异步排队运行，不需要在两层之间显式 wait。

**影响**：当 RS 通信时间 > backward compute 时间时，当前建模会 **高估**
end_t（多了一个 stall 窗口）。

#### 差异 #8：末层 RS 完成未被同步

**FSDP2 实际**：optimizer step 前需要等待所有 RS 完成（梯度归约完毕才能
更新参数）。

**当前建模**：layer-wise 模式下，`OptimizerSimulator` tail 只有
`optim_step`（无 barrier/RS wait）。最后一个 backward block post 的 RS 在
`dp_comm` lane 上异步运行，`optim_step` 紧跟其后开始——**没有 wait RS
完成**。如果末层 RS 没跑完 optimizer 就开始了，会 **低估** end_t。

---

## 3. 修正方案

### 3.1 新增配置

```python
# StrategyConfig 新增字段
reshard_after_forward: bool = True   # True=FULL_SHARD, False=SHARD_GRAD_OP
fsdp_prefetch_layers: int = 1        # 预取层数（v1=1 隐式预取, v2 支持 2+）
```

- `reshard_after_forward=True`（默认，对应 FSDP2 `FULL_SHARD`）：
  Forward 后 reshard → Backward 前需 AG
- `reshard_after_forward=False`（对应 `SHARD_GRAD_OP`）：
  Forward 后不 reshard → Backward 不需 AG
- `fsdp_prefetch_layers`：预取层数，默认 1。v1 先实现 1，v2 再支持 2+。

### 3.2 Backward AG 注入

在 `reshard_after_forward=True` 时，每个 LLMBlock 需要 **独立的 backward AG
ops**（与 forward AG ops 分离，因为 `_posted` flag 不能复用）。

新增 `_build_fsdp_bwd_ag_ops()` 方法，构建 `async_all_gather` op，设置
`fwd_cost=0, bwd_cost=ag_cost`（只在 backward 阶段 post）：

```python
def _build_fsdp_bwd_ag_ops(self, args, com_buff):
    """Per-LLMBlock FSDP backward unshard POST ops. Only created when
    reshard_after_forward=True (FULL_SHARD). These post AG in the backward
    phase, mirroring the forward AG but with bwd_cost instead of fwd_cost."""
    # ... same comm cost computation as _build_fsdp_ag_ops ...
    ops.append(async_all_gather(
        ..., fwd_cost=0, bwd_cost=dense_ag_cost,  # bwd-phase post
        stream="dp_comm", ...))
```

### 3.3 `async_all_gather` / `async_reduce_scatter` 修改

当前 `_posted` flag 在 forward 和 backward 间共享。需要改为 **per-phase
posted flag**：

```python
class async_all_gather(LeafModel):
    def __init__(self, ..., fwd_cost=0, bwd_cost=0, ...):
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        self._posted_fwd = False
        self._posted_bwd = False

    def _step(self, t, ctx):     # forward
        if self.fwd_cost > 0 and not self._posted_fwd:
            return self._post(t, ctx, "fwd")
        return True, None

    def _bwd(self, t, ctx):      # backward
        if self.bwd_cost > 0 and not self._posted_bwd:
            return self._post(t, ctx, "bwd")
        return True, None

    def _post(self, t, ctx, phase):
        posted_attr = f"_posted_{phase}"
        if getattr(self, posted_attr):
            return True, None
        ...
        setattr(self, posted_attr, True)
        return False, ("yield_keep", gid)
```

### 3.4 Forward 交织（不变）

保持当前设计：
```
[wait AG(this)] → [post AG(next)] → [compute_fwd]
```
首块特殊处理：`[post AG(0)] → [wait AG(0)] → [post AG(1)] → [compute]`

### 3.5 Backward 交织修正

**当前（有缺陷）**：
```
[compute_bwd] → [wait RS(next)] → [post RS(this)]
```

**修正后 — FULL_SHARD**：
```
[wait AG(this)] → [post AG(next_bwd)] → [compute_bwd] → [post RS(this)]
```

**修正后 — SHARD_GRAD_OP**：
```
[compute_bwd] → [post RS(this)]
```

**关键变化**：
1. 删除 `wait RS(next)` 同步点（RS 异步 post，不需要层间 wait）
2. FULL_SHARD 模式增加 `[wait AG(this)] → [post AG(next_bwd)]` 前缀

**BwdStk LIFO 顺序设计**：

BwdStk `pop(-1)`（从末尾 pop）。temporal order = pop order = list 末尾→开头。

期望 temporal order（FULL_SHARD）：
`wait_AG(1st) → post_AG(2nd) → compute(3rd) → post_RS(4th=last)`

对应 stk（开头→末尾）= `post_RS → compute → post_AG → wait_AG`

```python
def prefill_bwd(self):
    bwd = super().prefill_bwd()   # children bwd 在 stk 中
    if not self._is_layer_wise_fsdp or not self._fsdp_rs_ops:
        return bwd

    reshard = self.strategy.reshard_after_forward

    # Bottom (stk[0:0]): post_RS → pops last → temporal last ✓
    bwd.stk[0:0] = [op.prefill_bwd() for op in self._fsdp_rs_ops]

    if reshard:
        # FULL_SHARD: top (extend): wait_AG, post_AG_next_bwd → pops first → temporal first ✓
        ag_this = self._fsdp_bwd_ag_ops   # backward-specific AG ops
        ag_next_bwd = self._prev_block._fsdp_bwd_ag_ops if self._prev_block else []
        head = [async_wait_collective(ag_this, call_stk=...+'-fsdp_bwd_ag_wait')]
        if self._next_block is None:
            # First backward block (last forward): self-post AG (exposed)
            head = [op.prefill_bwd() for op in ag_this] + head
        if self._prev_block is not None:
            head += [op.prefill_bwd() for op in ag_next_bwd]  # post AG for next bwd
        bwd.stk.extend(head)
    # SHARD_GRAD_OP: no AG, no wait — just compute → post_RS
    return bwd
```

**Block 链接语义（backward 方向）**：

| 属性 | forward 语义 | backward 语义 |
|------|-------------|-------------|
| `_next_block` | forward 下一层 | backward 前一层（已执行 backward） |
| `_prev_block` | forward 前一层 | backward 下一层（将执行 backward） |

- `wait AG(this)`：AG 由 `_next_block`（backward 前一层）的 prefetch post
- `post AG(next_bwd)`：post `_prev_block`（backward 下一层）的 AG
- 首个 backward block（`_next_block is None`）：自 post AG（完全暴露）
- 末个 backward block（`_prev_block is None`）：不 post AG（无后继）

### 3.6 OptimizerSimulator tail 修正

在 layer-wise 模式下，tail 需要增加 **RS 完成等待**：

```python
# pipeline_schedule.py — layer-wise 分支
if fsdp_layer_wise:
    # Wait for all per-block RS to complete before optimizer step
    rs_wait = async_wait_collective(
        all_rs_ops_from_last_block,  # or a global barrier
        call_stk='fsdp_rs_complete')
    optim_step = AtomModel(fwd_cost=opt_info['optim_time'], ...)
    self._step_only_layers = [rs_wait, optim_step]
```

或更简单地，在 `simu_runner.py` 拼接时插入一个 dp_comm lane 的 drain barrier。

### 3.7 Analytical 估算修正

当前 analytical 估算（`perf_llm.py`）：
```python
fwd_exposed = Σ max(0, AG_block - compute_{prev}_fwd)
bwd_exposed = Σ max(0, RS_block - compute_{next}_bwd)
```

修正后（FULL_SHARD）：
```python
fwd_exposed = Σ max(0, AG_block - compute_{prev}_fwd)   # 不变
bwd_exposed = Σ max(0, AG_block - compute_{next}_bwd)   # AG 也有 exposed
             + Σ max(0, RS_block - compute_{prev}_bwd)  # RS 也有 exposed
# 首个 backward AG + 末个 backward RS 完全暴露
```

SHARD_GRAD_OP（无 backward AG）：
```python
fwd_exposed = Σ max(0, AG_block - compute_{prev}_fwd)
bwd_exposed = Σ max(0, RS_block - compute_{prev}_bwd)
# 首个 forward AG 完全暴露；末个 backward RS 完全暴露
```

---

## 4. Phased Implementation

### Phase 1 — FULL_SHARD backward AG + 交错修正

- `reshard_after_forward` 配置字段 + 校验
- `async_all_gather` / `async_reduce_scatter` per-phase `_posted` flag
- `_build_fsdp_bwd_ag_ops()` 构建 backward AG ops
- `prefill_bwd()` 修正：FULL_SHARD 交错 + 删除 `wait RS(next)`
- OptimizerSimulator tail 增加 RS 完成等待
- Analytical 估算修正
- 文档刷新

**验证**：
- `reshard_after_forward=False`（SHARD_GRAD_OP）：与当前行为一致（删除
  `wait RS(next)` 后 end_t 应 ≤ 当前值）
- `reshard_after_forward=True`（FULL_SHARD）：backward 出现 per-block AG
  span，与 compute 重叠
- `zero_state ≤ 1`：golden 不变

### Phase 2 — 显式预取 2+ 层

- `fsdp_prefetch_layers` 配置字段
- `prefill_fwd` / `prefill_bwd` 支持 post AG(next+k) for k in 1..N
- Analytical 估算适配
- 文档刷新

**验证**：
- `fsdp_prefetch_layers=1`：与 Phase 1 一致
- `fsdp_prefetch_layers=2`：trace 中可见 2 个 AG span 同时在 dp_comm lane 飞行

---

## 5. 设计决策待确认

1. **`reshard_after_forward` 默认值**：建议 `True`（FULL_SHARD 是 FSDP2 默认）。
   当前行为相当于 `False`（SHARD_GRAD_OP），改为默认 True 会改变现有
   layer-wise 的 end_t。是否接受这个变化？

2. **`wait RS(next)` 删除**：当前 backward 有 `wait RS(next)` 同步点。
   删除后 RS 纯异步，两层 backward 间无同步。这会降低 end_t（更准确），
   但也意味着 RS 完全靠 dp_comm queue 排队。是否接受？

3. **RS 完成同步位置**：是在 OptimizerSimulator tail 加 `wait all RS`，
   还是在 simu_runner 拼接时插入 dp_comm drain？前者更显式，后者更轻量。

4. **显式预取是否 v1 就做**：还是 v1 只做 FULL_SHARD backward AG + 交错
   修正，显式预取留到 v2？

---

## 6. Decisions Log

1. **`reshard_after_forward` default = `True`** (FULL_SHARD, FSDP2 default).
   Current behavior implicitly modeled SHARD_GRAD_OP; switching default to
   True changes existing layer-wise end_t — accepted.
2. **Delete `wait RS(next)` sync point** — accepted. RS is purely async post,
   no inter-layer wait. Completion is only synchronized before optimizer step.
3. **RS completion sync: explicit** — add `async_wait_collective` in
   OptimizerSimulator tail (layer-wise mode), before `optim_step`.
4. **Explicit prefetch 2+ layers: do now** — `fsdp_prefetch_layers` config
   field, default 1. Both Phase 1 and Phase 2 implemented in this cycle.
