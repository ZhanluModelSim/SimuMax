<p align="center">
  <a href="strategy.md">English</a>| 
  <a href="strategy-zh.md">中文版本</a> 
</p>

# Strategy 配置
SimuMax 依赖三个核心输入文件：`system`、`strategy`、`model`。`strategy` 文件定义训练运行时选择，例如 TP / PP / EP、总卡数、batch、recompute、VPP 等。

相关文档：

- [docs/README.md](./README.md)
- [model.md](./model.md)
- [system.md](./system.md)

strategy 是 SimuMax 和 Megatron 运行时语义最直接对齐的一层。若 real run 和 strategy 在 PP / EP / TP、sequence parallel、recompute、VPP 等设置上不一致，timing 和 memory 都可能明显漂移。

## 最快起步方式

除非非常特殊，否则不要从空文件开始写。

推荐路径：

1. 从 [configs/strategy](../configs/strategy) 复制最接近的已有 JSON。
2. 先把 `seq_len`、`micro_batch_size`、`micro_batch_num` 配成最简单的版本。
3. 先保证并行规模合法，再考虑 recompute 或 VPP。
4. 只有普通 PP 跑通后，再启用 `interleaving_size > 1`。

示例起点：

- dense TP/PP 基线：
  [configs/strategy/tp1_pp2_dp4_mbs1.json](../configs/strategy/tp1_pp2_dp4_mbs1.json)
- MoE EP 基线：
  [configs/strategy/ep8_pp1_dp8_mbs1.json](../configs/strategy/ep8_pp1_dp8_mbs1.json)

## 什么时候适合先做 search

如果你已经有一份接近的 strategy JSON，通常更推荐：

1. 先固定并行策略，搜索 `micro_batch_size` / `micro_batch_num`
2. 再围绕最近的已有配置，小范围搜索 `tp/pp`

公共入口可参考：

- [tutorial.md](./tutorial.md)
- [examples/search_strategy_llama3_8b.py](../examples/search_strategy_llama3_8b.py)

补充说明：

- `gmi_error` 是按卡预留的 GiB 级显存余量，用来粗略覆盖 NCCL buffer、
  allocator/runtime 开销，以及其他没有显式建模的组件
- 在新机器上第一次做 search 时，可以先用较保守的 `10`，之后再结合 real
  显存结果收紧

## 最小可用 strategy 示例

```json
{
    "seq_len": 4096,
    "micro_batch_size": 1,
    "micro_batch_num": 8,
    "dtype": "bf16",
    "world_size": 8,
    "tp_size": 1,
    "pp_size": 1,
    "ep_size": 1,
    "etp_size": 1,
    "enable_sequence_parallel": false,
    "interleaving_size": 1,
    "zero_state": 1,
    "enable_dropout": false,
    "use_flash_sdp": true,
    "enable_recompute": false,
    "mem_factor": 0.94
}
```

## 单机 8 卡的起步心智模型

先从最简单的开始：

1. dense 模型，不开 VPP，不开 recompute
2. `world_size=8`，`tp=1`，`pp=1`，`ep=1`，`cp=1`
3. 这时剩余的并行就是纯 data parallel

然后逐步增加复杂度：

1. 单层太大，再增大 `tp_size`
2. 整体模型太大，再增大 `pp_size`
3. 只有 MoE 模型才引入 `ep_size`
4. 只有普通 PP 跑通后，才引入 `interleaving_size > 1`

## 常用必填字段和常见默认

通常需要明确设置的字段：

- `seq_len`
- `micro_batch_size`
- `micro_batch_num`
- `world_size`
- `tp_size`
- `pp_size`
- `ep_size`
- `etp_size`
- `dtype`

很多用户一开始可以保持默认的字段：

- `zero_state`
- `enable_dropout`
- `mem_factor`
- 大多数 `use_fused_*`
- 大多数 recompute 子开关

## world_size / tp / pp / ep / dp 的关系

dense 场景下最常见的关系是：

- `dp = world_size / (tp * pp * cp)`

也就是说，dense 配置至少要满足：

- `world_size` 能被 `tp * pp * cp` 整除

MoE 场景还需要满足：

- `world_size % (ep * etp * pp) == 0`

实际建议是：

- 先把 dense 的 `tp/pp/cp` 配合法
- 再加入 `ep`
- 再检查模型侧 expert 相关整除，例如 `expert_num % ep == 0`

# 参数说明
## 基础训练参数
### seq_len
序列长度（token数量）
### micro_batch_size
微批次大小（单词每次前向传播处理的样本数）
### micro_batch_num
梯度累积的微批次数量
### dtype
计算数据类型（bf16表示半精度浮点数），默认为bf16
### fp8
是否使用fp8混合精度训练，默认为false
## 分布式策略
### world_size
总GPU数量（默认8）
### tp_size
张量并行大小（Tensor Parallelism），默认为1
### pp_size
流水线大小（Pipeline Parallelism），表示按层进行纵向切分，默认为1
### ep_size
专家大小（Expert Parallelism），仅用于MOE模型，默认为1
### etp_size
专家张量大小（Expert Tensor Parallelism），默认为1
### moe_dispatcher_policy
MOE模型的路由策略， 默认为"all2all"
### enable_sequence_parallel
是否启用序列并行，默认为true，当tp_size > 1时生效
### num_layers_in_first_pipeline_stage & num_layers_in_last_pipeline_stage
控制第一个和最后一个Pipeline Parallel stage包含的层数，默认为None
### interleaving_size
虚拟 pipeline 大小。第一次起步时保持 `1` 即可。`interleaving_size > 1` 时，`pp_size` 也必须大于 `1`。
### order_of_paralielism
稠密并行维度在机器网络层级上的放置（placement）顺序，从内到外排列，默认
`"tp-cp-ep-dp-pp"`（即当前内置的 mesh 顺序）。语法：以 `-` 分隔的 token，
`tp`/`cp`/`dp` 三者各出现一次、顺序任意；`ep` token 可选、可出现在任意位置
（会被忽略——MoE mesh 的放置是固定的）；`pp` 可选、只能出现在末尾
（`pp` 出现时必须位于最外层）。示例：`"tp-cp-ep-dp-pp"`（默认）与
`"cp-tp-ep-dp-pp"`（cp 位于最内层）。它影响分层网络模型中 levels 成本路径与
net 放置所用的通信组构成 / stride 计算。约束：`pp` 始终位于最外层；
`ep`/MoE mesh 放置固定。详见 `docs/design_simu_hierarchical_network.md` 第 4 节。
### zero_state
ZeRO 优化配置。支持 `0`、`1` 和 `3`（FSDP，参数分片），默认为 `1`。`2`
已声明但尚未接入。当 `zero_state >= 3` 时，由 `fsdp_mode` 字段选择 FSDP
通信模式，见下文。对于 `zero_state <= 1`，行为保持不变（ZeRO-1 优化器状态
分片）。详见 `docs/design_simu_zero3_fsdp.md`。

### fsdp_mode
FSDP 通信模式，仅在 `zero_state >= 3` 时有意义；默认 `"model-wise"`。合法
取值：`"model-wise"`、`"layer-wise"`。

- `"model-wise"`（默认，Phase 1）：PP 前向之前一次性 all-gather 全部参数，
  PP 反向之后一次性 reduce-scatter 全部梯度并执行 optim step。几乎无重叠。
  world all_reduce 屏障仍保留在尾块。注意：model-wise FSDP 相比 ZeRO-1
  并不节省峰值显存——all-gather 缓冲区持有完整的未分片参数，因此峰值近似为
  `完整参数 + 分片梯度 + 分片状态 + 激活`（与 `zero_state = 1` 相同）；其收益
  仅为 ZeRO-1 已捕获的优化器状态分片。
- `"layer-wise"`（Phase 2）：按 `LLMBlock` 在该 block 前向之前 all-gather
  参数、在该 block 反向之后 reduce-scatter 梯度。稠密参数/梯度走 `dp_cp` 组
  （`dp_size * cp_size`，`dp_net`）；当 block 含专家时，MoE 专家参数/梯度走
  `edp` 组（`edp_size`，`edp_net`）。在 DES（`simulate()`）路径中，这些通信以
  **阻塞式**集合通信注入（暂无异步重叠），因此 trace 中呈现的是按 block 顺序
  排布、与各 block 计算串行的非重叠 AG/RS 区间。快速解析路径（`analysis()`）
  给出重叠估计 `dp_comm_exposed_time = Σ_blocks max(0, comm_block -
  prev_compute_block)`（前向 + 反向）；DES 给出精确的非重叠值，异步 post/wait
  重叠为后续工作。峰值显存为 `静态（分片）+ 一个 block 的未分片参数 + 激活`——
  远低于 model-wise，因为同一时刻只 gathering 一个 block 的参数而非整模型。当
  `zero_state <= 1` 或 `fsdp_mode != "layer-wise"` 时，`LLMBlock` 的前向/反向
  保持不变。

详见 `docs/design_simu_zero3_fsdp.md` 第 4 节（model-wise）与第 5 节
（layer-wise）。
## 内存优化
### grad_reduce_in_bf16
梯度归约是否使用bf16，默认为false
### use_accm_weight
是否使用累加权重融合（减少临时变量）, 默认为true
### cache_groupgemm_col_fp8_inputs
是否缓存groupgemm的FP8输入，默认为false
### offload_groupgemm_col_inputs
是否卸载groupgemm的输入到CPU，默认为false


## 重计算相关
#### enable_recompute
recompute全局开关，默认为true
#### recompute_granularity
recompute的粒度，可选为"full_block"和"selective_recompute"，默认为None
#### recompute_layer_num
recompute的层数，默认为0
#### attn_recompute
对attention模块进行重计算，默认为false
#### mla_rms_recompute
对mla的rmsnorm和q/k up-projection进行重计算，默认为false
#### mlp_recompute
对MLP和groupedgemm进行重计算，默认为false
#### mlp_rms_recompute
对rmsnorm+router+sharedExpert进行重计算，默认为false
#### recompute_variance
recompute checkpoint的最后一个module是否去掉冗余的前向计算，默认为false, 当recompute_granularity为"selective_recompute"时，建议设置为true以节省计算时间

#### megatron_recompute
Megatron-LM 0.14 引入了基于 `discard_output` 的 selective recompute。使用
`megatron_recompute=true` 开启，并在 `megatron_recompute_modules` 中列出
被 discard output 的模块。

示例：

```json
{
    "enable_recompute": true,
    "recompute_granularity": "selective_recompute",
    "recompute_layer_num": 12,
    "megatron_recompute": true,
    "megatron_recompute_modules": ["layernorm", "mlp"]
}
```

当前支持的模块名包括 `layernorm`、`mla_up_proj`、`moe_act`、`mlp`、`moe`。
`core_attn` 是预留名，但暂不支持。该模式不能和旧的 selective flags
（例如 `attn_recompute`、`mlp_recompute`）混用；目前也不通过 search helper
自动搜索，建议显式配置后单独评估。

## 计算优化
### attention_sparse_ratio
注意力稀疏比例（0.0为密集注意力），默认为0.0
### use_flash_sdp
使用FlashAttention加速
### cross_entropy_loss_fusion
是否启用 SimuMax 里的 fused cross entropy，默认是 `false`。

与 Megatron 的对应关系：

- SimuMax strategy 字段：`cross_entropy_loss_fusion=true`
- 本仓库常用简称：`ce_fusion`
- 结果表里常见的 case 后缀：`_cef`

对 Megatron real run，这个简称对应同时开启：

- `--cross-entropy-loss-fusion`
- `--cross-entropy-fusion-impl te`

因此仓库里的 `ce_fusion` / `_cef` 可以理解为：

- `cross_entropy_loss_fusion=True`
- 使用 TE 的 fused CE 实现
### use_fused_*
各种融合内核优化
### enable_dropout
是否启用Dropout正则化，默认为false


## 网络策略
### tp_net, pp_net, dp_net等
各种并行维度的网络通信策略，默认为"auto"，根据集群规模和并行策略自动选择

## 其他
### dispatch_probs
Megatron-LM 相关参数，用来决定 MoE 里 probs 的归属口径。

- Megatron-LM 0.14 及之后：用 `dispatch_probs=True`
- Megatron-LM 0.12 及更早：用 `dispatch_probs=False`

如果是中间过渡版本或本地 patch 版本，先确认实际走到的 MoE 路径，再决定这个开关。
### mem_factor
内存使用系数（0.94表示留6%的余量），用于估算reserve_memory（=max_memory / mem_factor），默认为0.94。

## 模拟资源选项（可选）

以下字段控制 `PerfLLM.simulate()` 背后的 DES 资源 lane 模型（见
[design_simu_kind_resource_model-zh.md](./design_simu_kind_resource_model-zh.md)）。
全部为可选字段；缺省时与当前行为一致。

### compute_engine_map
把计算类别映射到 `system.engines` 中声明的引擎 lane，例如
`{"gemm": "cube", "elementwise": "vector"}`。仅当 system 配置声明了对应
引擎时才生效；未映射的类别仍走默认 compute lane。

### fused_ops
通算融合算子列表。每项形如
`{"pattern": ..., "policy": "serial" | "max_overlap" | "chunked_pipeline", "chunks": n}`。
例如 AG+GEMM 融合 kernel 可配置为
`[{"pattern": "tp_ag_gemm", "policy": "chunked_pipeline", "chunks": 4}]`。
`chunks` 仅对 `chunked_pipeline` 策略生效。不配置时保持当前的串行建模。

### fused_mem_mode
融合算子的显存记账模式。`"steady_state"`（默认）在算子开始时按闭式
稳态峰值记账。`"ramp"` 为忠实的逐 chunk 爬升曲线预留，目前会告警并
回退到 `"steady_state"`。

### collective_skew
DES 中本地集合通信的跨 rank skew 模型（见
[design_simu_network_fabric-zh.md](./design_simu_network_fabric-zh.md) 第 8
节 Phase C）。默认 `None` 表示关闭。`"virtual_waiters"` 会按
`estimate_straggler_increase_ratio(集合通信 group 的节点数)`
放大每个本地集合通信的完成时间——确定性的、节点粒度的 skew。它只影响
DES `simulate()` 路径；而 `enable_straggler_model` 缩放的是解析估计
`run_estimate()` 的结果。

## 效率覆盖（可选）

### efficiency_overrides
用于临时 what-if 调优的按算子效率覆盖（
[design_simu_cost_model_tunability-zh.md](./design_simu_cost_model_tunability-zh.md)
Phase 1）。key 语法与
[system-zh.md](./system-zh.md#operator_efficiency可选) 中的
`operator_efficiency` 相同：class key（如 `"LinearCol"`）或 path key
（如 `"layer_3.mlp"`），值为标量或 `{"default", "shapes"}` 字典。

例如，只调第 3 层的 MLP：

```json
"efficiency_overrides": {
    "layer_3.mlp": 0.48
}
```

优先级（从高到低）：

1. 通过 `PerfBase.configure(..., efficiency_overrides={...})` 传入的
   API overrides
2. strategy 的 `efficiency_overrides`（本字段）
3. system 的 `operator_efficiency`

strategy/API overrides 适合临时调优；实测得到的稳定值应写入
`system.json`。key 会在 `run_estimate()` 构建模型后校验，匹配不到任何
模块的 key 会抛出 `ValueError`。
