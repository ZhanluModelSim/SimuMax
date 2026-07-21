<p align="center">
  <a href="design_simu_zero3_fsdp.md">English</a>|
  <a href="design_simu_zero3_fsdp-zh.md">中文版本</a>
</p>

# 设计方案：ZeRO-3 / FSDP 建模

- 状态：**草案 v0.1**（讨论已定稿，尚未实现）
- 日期：2026-07-17
- 范围：ZeRO-3（参数分片）、per-layer 与 per-model 的 FSDP 通信、
  overlap 建模。建立在仓库已有的 ZeRO-1 基础设施之上。

## 1. 背景

ZeRO-1（优化器状态分片）已完整接线：显存分片
（`state_bytes /= group`）、优化器时间按分片缩减、DES 的
`OptimizerSimulator` 发出 RS-grad → optim-step → AG-param 的
单体尾块。`zero_state=2/3` 已声明但 warn-ignore
（`config.py:810`）；叶子模块的显存分片分支（`>=2` 分片
grad_bytes、`>=3` 分片 weight_bytes）存在但通信序列错误或缺失。
FSDP 的 per-layer all-gather 参数（forward 前）与 reduce-scatter
梯度（backward 后）在层计算路径中完全不存在。
`overlap_grad_reduce` 是死配置；Phase 2 的 post/wait 机制已建好
但未接 FSDP。

## 2. 目标 / 非目标

目标：

1. `zero_state=3` 激活 FSDP（参数分片）；去掉 warn-ignore。
2. 新 `fsdp_mode` 字段选择通信模式：
   - `"model-wise"`：训练步开始前 all-gather 全部参数（unshard），
     结束后 reduce-scatter 全部梯度（reshard）；基本无 overlap。
   - `"layer-wise"`：逐 LLMBlock unshard/reshard，与邻接层 overlap；
     通信超出 overlap 窗口时有 exposed time。
3. 快速解析路径与 DES `simulate()` 路径都支持——用户自选；
   `simulate()` 通过 post/wait 调度给出精确 overlap 值。
4. MoE 块同样遵循粒度规则（layer-wise = per-block，model-wise =
   per-model），dense 参数走 dp_cp 组、expert 参数走 edp 组。
5. 向后兼容：`zero_state ≤ 1` 不变；无 `fsdp_mode` 时对
   zero_state=3 默认 model-wise（最安全的基线）。

非目标：flat vs per-tensor 参数布局区分（性能建模无差异）；
ZeRO-2 作为独立阶段（它坍缩为 zero_state=3 的同通信配方、仅 shard
大小不同）；真实 FSDP 检查点。

## 3. 配置

```json
{
    "zero_state": 3,
    "fsdp_mode": "layer-wise"
}
```

- `zero_state`：0/1/2/3（现有字段）。3 激活 FSDP。`config.py:810`
  对值 3 的 warn-ignore 去掉。（值 2 语义上是子集，如需后续单独
  去掉；暂仍 warn。）
- `fsdp_mode: str = "model-wise"` — 新 StrategyConfig 字段；合法值
  `{"model-wise", "layer-wise"}`。仅在 `zero_state >= 3` 时有意义；
  配更低 zero_state 时校验并 warn。

## 4. Model-wise FSDP（`fsdp_mode = "model-wise"`）

最简模式——结构上最接近现状尾块，仅重新定位与修正大小。

### 4.1 解析路径

- `_compute_dp_time`：AG size = 分片参数字节（叶子 `>=3` 分片后的
  `weight_bytes`，按 model chunk 汇总）；RS size = 分片梯度字节。
  桶化可保留（一个大的 AG/RS 按桶化走）或去掉（每 dense/MoE 家族
  一次不桶化调用）。v1 保留桶化以一致；仅修正 AG size 推导。
- `dp_comm_exposed_time = dp_comm_time`（无 overlap——设计如此）。
- `_compute_optim_time` 已正常工作（消费分片后的 state_bytes）。

### 4.2 DES

`OptimizerSimulator` 尾块变为：

```
AG(dense params, dp_cp_group) → AG(moe params, edp_group)
  → [PP schedule fwd/bwd 以全量参数运行]
  → RS(dense grads, dp_cp_group) → RS(moe grads, edp_group)
  → optim_step
```

AG 前置到 PP schedule 作业之前；RS 和 optim_step 后置。世界
all_reduce 屏障保留（若组已覆盖所有 rank 则吸收进 RS）。
`run_simulation` 将前置的 AG 接入 `PpSchedule.prefill_batch` 之前的
job 列表。

### 4.3 显存

峰值 = 静态（分片参数 + 分片梯度 + 分片状态）+ AG buffer（全量
未分片参数）+ 激活。由于 静态 + AG buffer = 全量参数 + 分片梯度 +
分片状态，峰值约等于 zero_state=1（参数始终全量）。Model-wise
FSDP 不比 ZeRO-1 省峰值显存；其节省在 ZeRO-1 已捕获的优化器状态
分片里。

## 5. Layer-wise FSDP（`fsdp_mode = "layer-wise"`）

逐 LLMBlock unshard/reshard，与邻接层 overlap。

### 5.1 解析路径（快速）

每 block 成本：

- `AG_block` = 该 block 的分片参数字节 /（dp 组带宽）
- `RS_block` = 该 block 的分片梯度字节 /（dp 组带宽）
- `compute_block_fwd` = 该 block 现有 fwd 计算时间
- `compute_block_bwd` = 现有 bwd 计算时间

Overlap 估计（前向）：

```
fwd_exposed = Σ_blocks max(0, AG_block - compute_{prev_block}_fwd)
bwd_exposed = Σ_blocks max(0, RS_block - compute_{next_block}_bwd)
dp_comm_exposed_time = fwd_exposed + bwd_exposed
```

第一个 block 无前驱可 overlap → AG 全 exposed。此公式是不可
overlap 部分的保守上界；DES 路径给出精确值。

### 5.2 DES（精确路径）

逐 LLMBlock，job 列表通过 post/wait 交织 AG/RS 与计算
（Phase-2 机制）：

```
post AG(params for block N+1) → compute block N fwd
  → wait AG → compute block N+1 fwd → ...
  → compute block N bwd → post RS(grads for block N)
  → compute block N+1 bwd → wait RS → ...
```

- `all_gather` / `reduce_scatter` 在 block 的 `prefill_fwd` /
  `prefill_bwd`（`language_model.py` 的 LLMBlock，不在
  `OptimizerSimulator`）中创建，走 dp_cp 组（dense）与 edp 组
  （MoE expert）。
- Post/wait 用阻塞集合通信的 post/wait 路径：`Com._step` 发 entry
  （post 标记）并 yield；wait 是独立 op，在 comm_entry 完成时阻塞。
  kind-resource 设计 Phase 2 的 post/wait 语义适用。
- `OptimizerSimulator` 尾块缩减为仅 `optim_step`（RS 在 bwd 时
  per-layer 发生；AG 在 fwd 时 per-layer 发生）。

### 5.3 显存

峰值 = 静态（分片）+ 一个 block 的未分片参数 buffer + 激活。远低于
model-wise（同一时刻只有一个 block 的参数 gathered，而非整个模型）。

### 5.4 MoE

Layer-wise 模式下，每个 MoE LLMBlock 获得：

- Fwd：`AG(dense params, dp_cp_group)` + `AG(expert params,
  edp_group)`（两个 AG，可都 post 后都 wait）
- Bwd：`RS(dense grads, dp_cp_group)` + `RS(expert grads,
  edp_group)`

Model-wise 模式下：步首一次大 `AG(all dense params, dp_cp)` + 一次大
`AG(all expert params, edp)`；步尾 `RS` 全量梯度。

## 6. 分阶段实施

- **Phase 1 — model-wise FSDP**：去掉 zero_state=3 的 warn-ignore；
  `fsdp_mode` 字段 + 校验；修正 `_compute_dp_time` 的 AG size；
  `OptimizerSimulator` 重定位（AG 前置 PP 前、RS 后置、optim_step 尾）；
  显存峰值含 AG buffer；文档。验证：zero_state≤1 golden 不变；
  model-wise E2E 跑通。
- **Phase 2 — layer-wise FSDP**：在 `language_model.py` 注入 per-LLMBlock
  AG/RS；DES post/wait 接线；`_compute_dp_time` 解析 overlap 估计；
  `OptimizerSimulator` 尾块缩减；显存峰值含 per-block buffer；
  MoE 双组 AG/RS；文档。验证：layer-wise E2E overlap delta；golden
  zero_state≤1 不变。

## 7. 验证

- `zero_state ≤ 1`：golden 逐事件等价（无 fsdp_mode）。
- Model-wise：analysis 跑通；DES 跑通；AG buffer 在显存中；
  OptimizerSimulator 尾块结构正确。
- Layer-wise：DES 产出 per-block AG/RS 事件；overlap 窗口在 trace 上
  可见（AG span 与 compute span 重叠）；解析 `dp_comm_exposed_time`
  与 DES end_t delta 对比报告。
- E2E：moe-8T 16384 卡 zero_state=3 model-wise vs layer-wise →
  峰值显存差异（一个 block vs 全模型 AG buffer）与 end_t 差异
  （overlap 节省）。

## 8. 决策记录

1. **FSDP-1 与 FSDP-2 统一**：性能建模无差异（flat vs per-tensor
   参数 → 通信量相同）；`zero_state=3` 是单一值。新 `fsdp_mode`
   字段（layer-wise / model-wise）选择通信模式。
2. **粒度**：layer-wise = per-LLMBlock；model-wise = per-model。
3. **MoE**：同样粒度规则——layer-wise MoE 块 per-block AG/RS（dense
   走 dp_cp 组，expert 走 edp 组）；model-wise 每组一次大 AG/RS。
4. **解析与 DES 都支持**：用户选 `analysis()`（快速，解析 overlap
   估计）或 `simulate()`（精确，DES post/wait overlap）。
