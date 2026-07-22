<p align="center">
  <a href="design_simu_fsdp_mem_mfu_fix.md">English</a>|
  <a href="design_simu_fsdp_mem_mfu_fix-zh.md">中文版本</a>
</p>

# FSDP 内存估计修正 + overlap_mfu_requirements 输出方案

- 状态：**提案 v3（待讨论）**
- 日期：2026-07-21（v3 更新——per-block 异构 compute + FLOPS）
- 范围：(A) 修正 FSDP 配置间内存估计无差异的问题（分析路径 + DES 路径）；
  (B) 新增 `overlap_mfu_requirements` 输出——per-block 的 compute-phase MFU
  阈值，支持层间异构（dense/MoE 混合、不同参数量）。

---

## Part A: FSDP 内存估计修正

（Part A 内容与 v2 相同，此处省略重复。关键点：）

1. **FULL_SHARD layer-wise AG buffer = (1+prefetch) × per_block**：
   当前 block + 预取 block 的 buffer 同时存活。
2. **SHARD_GRAD_OP layer-wise AG buffer = full_chunk**：
   forward 后不 reshard，逐层累积。
3. **Model-wise AG buffer = full_chunk**。
4. **AG buffer 纳入 peak_mem**（保守叠加）。
5. **DES 路径**：`SimuMemoryTracker` 新增 `alloc_transient`/`free_transient`。

预期效果：

| 配置 | peak_mem (first_stage) | AG buffer |
|------|----------------------|-----------|
| ZeRO-1 | 113.2 GB | 0 |
| model-wise FSDP | ~109 GB | full chunk (~8GB) |
| layer-wise FULL_SHARD (prefetch=1) | ~106 GB | 2 blocks (~3GB) |
| layer-wise SHARD_GRAD_OP | ~109 GB | full chunk (~8GB) |

---

## Part B: overlap_mfu_requirements 输出

### B.1 背景：层间异构建模现状

调研发现，当前 `_compute_layer_wise_fsdp_exposed_time` 中：

| 维度 | 是否 per-block | 数据来源 |
|------|--------------|---------|
| AG/RS 通信量 | ✅ per-block | `chunk.layer_i.get_model_info()` 已正确读取 |
| AG/RS 通信时间 | ✅ per-block | `ag_list[i]`, `rs_list[i]` 已 per-block 计算 |
| **compute time** | ❌ **均匀分配** | `fwd_per_block = phase["fwd_compute"] / n` |
| **FLOPS** | ❌ 未获取 | 需要 `chunk.layer_i.get_compute_info()` |

对于 dense+MoE 混合模型（如 DeepSeek-V2：1 dense + 59 MoE），dense 层
和 MoE 层的 compute time 和 FLOPS 差异巨大。均匀分配导致：
- dense 层的 overlap window 被高估（实际 compute < 均值）→ 低估 exposed
- MoE 层的 overlap window 被低估（实际 compute > 均值）→ 高估 exposed

**但 per-block compute time 和 FLOPS 数据已经可用**：
- `chunk.layer_i.get_cost_info()` 返回 per-block 的 `ModuleCostInfo`
  （含 `fwd_compute_time`, `bwd_compute_time`, `fwd_net_time` 等）
- `chunk.layer_i.get_compute_info()` 返回 per-block 的 `ModuleComputeInfo`
  （含 `fwd_flops`, `bwd_flops`）

不需要新建层间异构建模支持——**只需使用已有的 per-block API 替换均匀分配**。

### B.2 前置改动：per-block compute time

在 `_compute_layer_wise_fsdp_exposed_time` 中，当 `hasattr(chunk, 'layer_0')`
（live model，非 profile cache）时，同时收集 per-block compute time：

```python
# 替代 fwd_per_block = phase["fwd_compute"] / n
fwd_per_block_list = []
bwd_per_block_list = []
if hasattr(chunk, 'layer_0'):
    for i in range(layer_num):
        block = getattr(chunk, f'layer_{i}', None)
        if block is None:
            fwd_per_block_list.append(0.0)
            bwd_per_block_list.append(0.0)
            continue
        ci = block.get_cost_info()
        fwd_per_block_list.append(ci.fwd_compute_time + ci.fwd_net_time)
        bwd_per_block_list.append(
            ci.bwd_compute_time + ci.bwd_net_time
            + ci.recompute_compute_time + ci.recompute_net_time
        )
else:
    # CachedChunkProfile fallback: uniform
    phase = self._compute_single_batch_phase_inputs(model_name)
    fwd_per_block_list = [phase["fwd_compute"] / n] * n
    bwd_per_block_list = [phase["bwd_compute"] / n] * n
```

然后所有使用 `fwd_per_block` / `bwd_per_block` 的地方改为使用
`fwd_per_block_list[i]` / `bwd_per_block_list[i]`。

这也修正了 exposed time 的计算——per-block overlap window 更精确。

### B.3 前置改动：per-block FLOPS

同样，收集 per-block FLOPS：

```python
fwd_flops_list = []
bwd_flops_list = []
if hasattr(chunk, 'layer_0'):
    for i in range(layer_num):
        block = getattr(chunk, f'layer_{i}', None)
        if block is None:
            fwd_flops_list.append(0)
            bwd_flops_list.append(0)
            continue
        fi = block.get_compute_info()
        fwd_flops_list.append(fi.fwd_flops)
        bwd_flops_list.append(fi.bwd_flops)
else:
    # Fallback: uniform from chunk-level
    fi = chunk.get_compute_info()
    fwd_flops_list = [fi.fwd_flops / n] * n
    bwd_flops_list = [fi.bwd_flops / n] * n
```

### B.4 掩盖 pairing（per-block）

根据实际 schedule（FULL_SHARD, prefetch=1）：

**Forward AG**：AG(N+1) 与 compute_fwd(N) 重叠
- block N 的 overlap window = fwd_per_block_list[N]
- block N+1 的 AG = ag_list[N+1]
- per-block ratio: fwd_per_block_list[N] / ag_list[N+1]
  - 注意：最后一个 block 没有 AG(N+1)，用前一个 block 的 AG 做近似

**Backward AG**（FULL_SHARD only）：AG(N-1) 与 compute_bwd(N) 重叠
- block N 的 overlap window = bwd_per_block_list[N]
- block N-1 的 AG = ag_list[N-1]

**Backward RS**：RS(N) 与 compute_bwd(N-1) 重叠
- block N-1 的 overlap window = bwd_per_block_list[N-1]
- block N 的 RS = rs_list[N]

### B.5 per-block overlap_mfu 公式

```
fwd_ag_overlap_mfu[i] = fwd_flops_list[i] / (ag_list[i] × peak_TFLOPS × 1e12)
bwd_ag_overlap_mfu[i] = bwd_flops_list[i] / (ag_list[i] × peak_TFLOPS × 1e12)   # FULL_SHARD
bwd_rs_overlap_mfu[i] = bwd_flops_list[i] / (rs_list[i] × peak_TFLOPS × 1e12)
```

**不依赖 duration，不依赖 current_mfu，无循环依赖。**
每个 block 有独立的 overlap_mfu 值，反映该层的 compute/comm ratio。

### B.6 输出结构

在 `_compute_layer_wise_fsdp_exposed_time()` 的返回 dict 中新增：

```python
{
    ...
    "overlap_mfu_requirements": {
        "peak_tflops": <float>,
        "per_block": [
            {
                "block_idx": 0,
                "fwd_flops": <int>,
                "bwd_flops": <int>,
                "fwd_ag_overlap_mfu": <float>,
                "bwd_ag_overlap_mfu": <float or null>,   # null for SHARD_GRAD_OP
                "bwd_rs_overlap_mfu": <float>,
            },
            ...
        ],
        # Aggregate (weighted average for quick scan)
        "fwd_ag_overlap_mfu_avg": <float>,
        "bwd_rs_overlap_mfu_avg": <float>,
    },
    ...
}
```

per-block 列表让用户看到 dense 层（低 overlap_mfu，AG 小）vs MoE 层
（高 overlap_mfu，AG 大）的差异。Aggregate avg 便于快速判断。

### B.7 预期输出示例

以 DeepSeek-V2（1 dense + 5 MoE per stage）为例：

```json
{
    "overlap_mfu_requirements": {
        "peak_tflops": 312.0,
        "per_block": [
            {
                "block_idx": 0,
                "fwd_flops": 2.1e9,
                "bwd_flops": 4.2e9,
                "fwd_ag_overlap_mfu": 0.12,
                "bwd_ag_overlap_mfu": 0.24,
                "bwd_rs_overlap_mfu": 0.08
            },
            {
                "block_idx": 1,
                "fwd_flops": 5.4e10,
                "bwd_flops": 1.08e11,
                "fwd_ag_overlap_mfu": 0.54,
                "bwd_ag_overlap_mfu": 0.48,
                "bwd_rs_overlap_mfu": 0.29
            },
            ...
        ],
        "fwd_ag_overlap_mfu_avg": 0.48,
        "bwd_rs_overlap_mfu_avg": 0.27
    }
}
```

Block 0（dense 层）：AG 小（只有 dense params），compute 也小（无 MoE
routing/grouped GEMM）→ overlap_mfu 低，说明 compute 容易被 AG 淹没。
Block 1+（MoE 层）：AG 大（dense + expert params），compute 也大
（grouped GEMM）→ overlap_mfu 高，AG 更容易被 compute 掩盖。

### B.8 数据依赖关系

```
overlap_mfu[i] = fwd_flops_list[i] / (ag_list[i] × peak_TFLOPS × 1e12)
                  ↑                      ↑                 ↑
       get_compute_info()      get_model_info()     system.accelerator
       (per-block, live)       (per-block, ✓)       (global, ✓)
```

- `get_compute_info()` 需要在 `run_estimate()` 之后调用（`_compute_info` 已填充）
- `_compute_layer_wise_fsdp_exposed_time` 在 `_compute_dp_time` 中调用，
  而 `_compute_dp_time` 在 `_analysis_gbs_comm_time` 中调用，这在 `run_estimate()`
  之后——所以 `get_compute_info()` 已就绪。
- `CachedChunkProfile` 路径：per-block `get_compute_info()` 不可用，
  fallback 到 chunk 级均匀分配。

---

## Part C: per-block compute time 对 exposed time 的影响

### C.1 当前问题

当前 `_compute_layer_wise_fsdp_exposed_time` 用均匀分配的 compute time：

```python
fwd_per_block = phase["fwd_compute"] / n  # 均匀
fwd_exposed = ag_list[0]  # block 0 的 AG（dense 层，小）
for i in range(1, n):
    fwd_exposed += max(0.0, ag_list[i] - fwd_per_block)  # 用均值减
```

对于 DeepSeek-V2（1 dense + 59 MoE），dense 层的 AG 远小于 MoE 层的 AG，
但 compute time 也不同。用均值会导致 MoE 层的 exposed 被高估（均值 compute
< MoE compute），dense 层的 exposed 被低估（均值 compute > dense compute）。

### C.2 修正

用 per-block compute time：

```python
# Forward: AG(i) overlaps with compute_fwd(i-1)
fwd_exposed = ag_list[0]  # block 0's AG is fully exposed (no predecessor)
for i in range(1, n):
    overlap_window = fwd_per_block_list[i - 1]  # previous block's compute
    fwd_exposed += max(0.0, ag_list[i] - overlap_window)

# Backward RS: RS(i) overlaps with compute_bwd(i-1)
bwd_rs_exposed = rs_list[0]  # last backward block's RS fully exposed
for i in range(1, n):
    overlap_window = bwd_per_block_list[i - 1]
    bwd_rs_exposed += max(0.0, rs_list[i] - overlap_window)
```

注意 pairing 的变化：AG(i) 与 compute_fwd(i-1) 重叠（block i-1 的 forward
post 了 block i 的 AG）。用 `fwd_per_block_list[i-1]` 而非 `fwd_per_block`。

---

## 分阶段实施

### Phase 1 — 分析路径内存修正 + per-block compute time + overlap_mfu

**内存**（Part A）：
- `_analysis_mem_impl`: 区分 fsdp_mode/reshard/prefetch 的 AG buffer
- AG buffer 纳入 peak_mem

**per-block compute time**（Part C）：
- `_compute_layer_wise_fsdp_exposed_time`: 用 `get_cost_info()` 获取
  per-block fwd/bwd compute time，替换均匀分配
- 同时修正 exposed time 的 pairing（AG(i) vs compute(i-1)）

**overlap_mfu**（Part B）：
- `_compute_layer_wise_fsdp_exposed_time`: 用 `get_compute_info()` 获取
  per-block FLOPS，计算 per-block overlap_mfu

验证：
- DeepSeek-V2 对比报告中 dense 层和 MoE 层的 overlap_mfu 应不同
- exposed time 用 per-block compute 后应更精确（与 DES 对比）
- peak_mem 4 配置应有差异

### Phase 2 — DES 路径内存追踪

- `SimuMemoryTracker`: alloc_transient / free_transient
- `async_all_gather._post` → alloc_transient
- `async_reduce_scatter._post` → free_transient
- model-wise `all_gather` op: 补充 size_bytes + alloc/free

---

## 决策记录

1. **AG buffer 纳入 peak_mem**：保守叠加。——已确认
2. **SHARD_GRAD_OP 的 AG buffer = full_chunk**：逐层累积。——已确认
3. **FULL_SHARD 的 AG buffer = (1+prefetch) × per_block**。——已确认
4. **overlap_mfu 用 compute-phase FLOPS / (comm_time × peak_TFLOPS)**：
   不依赖 duration，无循环依赖。——已确认
5. **per-block compute time 和 FLOPS**：使用已有的 `get_cost_info()` /
   `get_compute_info()` API，不需要新建层间异构建模支持。CachedChunkProfile
   路径 fallback 到均匀分配。——v3 新增
6. **overlap_mfu 输出 per-block 列表 + aggregate avg**：让用户看到 dense vs
   MoE 层的差异。——v3 新增
7. **exposed time pairing 修正**：AG(i) 与 compute(i-1) 重叠，用
   `fwd_per_block_list[i-1]` 而非均值。——v3 新增
