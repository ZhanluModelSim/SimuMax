<p align="center">
  <a href="system.md">English</a>| 
  <a href="system-zh.md">中文版本</a> 
</p>


# System 配置
SimuMax 依赖三个核心输入文件：`system`、`strategy`、`model`。`system` 文件描述机器侧能力：

- 加速器算力
- 访存带宽
- 机内 / 机间通信带宽与 latency
- shape 级别的算子效率

一个完整的 `system.json` 总是包含三部分：

- 基本信息：系统名、单机卡数
- `accelerator`：计算和访存侧描述
- `networks`：机内和机间通信描述

某些机器族还会带额外字段，比如 `FC8`，但第一次理解时先抓住上面三部分即可。

相关入口：

- 总览：[README.md](./README.md)
- model 字段：[model.md](./model.md)
- strategy 字段：[strategy.md](./strategy.md)
- 机器测速流程：[simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md)

实际使用时还有一个很重要的口径：

- 共享公共流程可以自动生成算子效率
- 在支持的 CUDA/MUSA 硬件上，共享流程也会尝试自动补上 `accelerator.backend`、当前可见 `num_per_node` 和 `accelerator.mem_gbs`
- 但通信拟合结果目前仍需要人工写回 `networks`
- `accelerator.bandwidth` 的默认值仍然只是起步模板，做 timing 分析前需要人工确认
- 所以新生成的 `system.json` 只有在你检查过机器侧字段并把 starter network 数值替换成实测通信参数之后，才适合做 timing 分析

## 什么时候需要自己实测

以下场景通常可以直接使用已有 system 配置：

- 目标机器和仓库中的示例机器接近
- 通信拓扑没有本质差别
- 目标 case 的主要算子 shape 已经有 `accurate_efficient_factor`

以下场景建议先做自己的机器实测：

- 机器是新的
- 机内或机间带宽 / latency 与已有配置差异明显
- `system.miss_efficiency` 不是空，而且你要解释 timing

经验规则：

- 如果只是做 OOM feasibility 判断，缺失 efficiency 还可以暂时容忍
- 如果要解释 `perf vs simulator` 或 `perf vs real` 的 timing，先补齐 efficiency

## 最快起步方式

除非非常特殊，否则不要从空文件开始写。

推荐路径：

1. 从 [configs/system](../configs/system) 复制最接近的已有配置。
2. 修改 `sys_name`。
3. 修改 `num_per_node`、`accelerator.backend`、`accelerator.mem_gbs`。
4. 先把 `networks` 改成最接近你机器拓扑的版本。
5. 最后再补 `accurate_efficient_factor` 和拟合得到的通信参数。

如果你只是做近似分析，复制最近的已有配置通常比从空文件新建更稳妥。

## 最小可用模板

下面这个例子是一个真正完整、可作为起点的最小模板，包含了 `networks` 骨架。

```json
{
    "sys_name": "my_system",
    "num_per_node": 8,
    "accelerator": {
        "backend": "cuda",
        "mem_gbs": 80,
        "op": {
            "default": {
                "tflops": 312,
                "efficient_factor": 0.75
            }
        },
        "bandwidth": {
            "default": {
                "efficient_factor": 0.9,
                "gbps": 1600,
                "latency_us": 40
            }
        },
        "mode": "roofline"
    },
    "networks": {
        "intra_with_pcie": false,
        "low_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 300,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        },
        "high_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 300,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        },
        "inter_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 200,
                "latency_us": 30
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        }
    }
}
```

## 必填字段与常见默认

建议视为必填的字段：

- `sys_name`
- `num_per_node`
- `accelerator.backend`
- `accelerator.mem_gbs`
- `accelerator.op.default`
- `accelerator.bandwidth.default`
- `networks.intra_with_pcie`
- `networks` 下对应的网络组

很多用户一开始可以沿用已有配置的字段：

- `accelerator.mode`（通常是 `roofline`）
- `processor_usage`（当前公共配置里基本都是保留字段）
- `accelerator.bandwidth` 下算子级访存微调
- `networks.*.op` 下算子级通信微调

如果只是做近似分析，可以：

- 先复制最近的已有 system config
- 暂时保留很多默认效率
- 用最接近机器的通信参数先跑通

如果要 timing 更准确，建议自己测：

- 主要 `matmul`、`group_matmul`、attention shape
- 机内 / 机间通信带宽与 latency

换句话说：

- `accelerator.op.*.accurate_efficient_factor` 用来补齐算子效率
- `networks.*` 用来补齐通信 timing
- `num_per_node`、`accelerator.mem_gbs`、`accelerator.bandwidth.*` 这些机器侧字段，在依赖 timing 结果前也需要人工确认

共享测速流程见 [simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md)。

## accelerator
accelerator部分包含了显存、访存带宽、算力、各个算子的计算效率等。

### backend 
后端描述，仅用于标识。


### mem_gbs
显存大小，单位为GB。

### op
该部分定义了各个算子使用的默认算力和不同shape下准确的计算效率。

SimuMax的核心之一是实现了shape级别计算效率建模，这是性能准确建模的关键，因此，SimuMax支持用户自定义多个核心算子在不同shape下的计算效率描述, 并且定义了一套shape表达规则， 用户需要按照该规则来新增算子不同shape的计算效率。

目前支持的算子列表和其shape表达规则为：


|key|算子|格式|示例|备注|
|---|---|---|---|---|
|matmul|矩阵乘法| b={batch_size}, m={m}, k={k}, n={n}, layout={layout}, accumulate={accumulate}, out_dtype={out_dtype}|`b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16`|accumulate：是否进行梯度累加，反向对w求导时该项为True|
|fp8_matmul|FP8矩阵乘法|同上|同上|同上|
|sdp_fwd|SDP前向计算|batch={batchh_size}, seq_len={seq_len}, head_num={head_num}, kv_head_num={kv_head_num}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}, qkv_contiguous={qkv_contiguous}|`batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0729673001633662`| qkv_contiguous：输入qkv是否在内存上连续，该项影响计算性能，因此单独描述，A100上一般为连续输入|
|sdp_bwd|SDP反向计算|同上|同上|同上|
|group_matmul| MOE模型的分组matmul|ng={num_groups}, M={fwd_M}, N={fwd_N}, K={fwd_k}, dtype={dtype}, out_dtype={out_dtype}, main_grad_dtype={main_grad_dtype}, stage={stage}, grad={grad}, accumulate={accumulate}, use_split_accumulator=False, single_output={single_output}| fwd stage：<br> `ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True": 0.6313438865579614`|1. fwd、bwd_grad_act、bwd_grad_w三个阶段的groupedgemm shape描述的M,N,K都等于fwd阶段的M,N,K，用stage来区分不同阶段<br>2. single_output只有fwd stage时为True<br>3. accumulate只有bwd_grad_w stage时为True<br>4. grad和use_split_accumulator只有在bwd_grad_w stage时为True|




例如，对于NVIDIA A100，其各个算子使用的算力和不同shape的计算效率描述为:

```json

"op": {
    "default" : {
        "tflops": 312,
        "efficient_factor": 0.75
    },
     "matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16": 0.7876672065615554,
                    "b=1, m=4096, k=1536, n=5120, layout=NN, accumulate=False, out_dtype=bf16": 0.737505124681297
                },
    },
     "fp8_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {},
     },
     "sdp_fwd" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0729673001633662,
                    "batch=1, seq_len=4096, head_num=64, kv_head_num=64, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0544285429372056
                },
     },
     "sdp_bwd" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 0.8018473732899901,
                    "batch=1, seq_len=4096, head_num=64, kv_head_num=64, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 0.7942592665301026
                },
     },
     "group_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True": 0.6313438865579614,
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=bwd_grad_act, grad=True, accumulate=False, use_split_accumulator=True, single_output=False": 0.6790978664070304,
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=bwd_grad_w, grad=True, accumulate=True, use_split_accumulator=True, single_output=False": 0.5196854178569805
                },
     },
     "fp8_group_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {},
     },
}
```
其中，`default`表示默认算力，不支持的算子类型使用该算力；每个算子下面，`tflops`表示标称算力，`efficient_factor`表示默认计算效率，`accurate_efficient_factor`表示各个算子在不同shape下的实际计算效率。


### bandwidth
访存带宽描述，包含各个访存类型的带宽。例如，对于NVIDIA A100，其访存带宽描述为:

```json
"bandwidth": {
    "default" : {
        "efficient_factor": 0.91,
        "gbps": 1600,
        "latency_us": 40
    },
    "permute_fwd":{
        "efficient_factor": 0.1879,
        "gbps": 1600,
        "latency_us": 40
    },
    "permute_bwd":{
        "efficient_factor": 0.1879,
        "gbps": 1600,
        "latency_us": 40
    },
    "ce":{
        "efficient_factor": 0.808,
        "gbps": 1600,
        "latency_us": 40
    }
}
```
其中，default表示默认访存带宽及其效率；除了默认访存带宽，我们新增3个memory bound算子的算子微调效率，用户可以自定义：
- permute_fwd表示permute前向的访存带宽及其效率
- permute_bwd表示permute反向的访存带宽及其效率
- ce表示cross entropy的访存带宽及其效率

## networks
### FC8
是否为FC8互联。
### intra_with_pcie
机内是否是pcie互连。
- intra_with_pcie=True，则表示机内是pcie互连，networks还需包含以下网络带宽配置
```json
"intra_node_pcie_8x": {
},
"intra_node_pcie_4x": {
},
"intra_node_pcie_2x": {
},
"inter_node": { 
}
```
- intra_with_pcie=False， 否则表示机内是nvlink高速互连， networks还需包含以下网络带宽配置。  
```json
"low_intra_node": {
},
"high_intra_node": {
},
"inter_node": { 
}
```

### intra_link_type
机内互连类型选择（可选，默认 `"nvlink"`）。支持的取值：
- `"nvlink"` — NVIDIA NVLink（默认；等同于 `intra_with_pcie: false`）
- `"pcie"` — PCIe（等同于 `intra_with_pcie: true`）
- `"ublink"` — 华为 UBLink 高速互连，地位等同于 NVLink；使用与 NVLink
  相同的 `low_intra_node` / `high_intra_node` / `inter_node` 网络配置结构
  和相同的二元分析路径

当 `intra_link_type` 存在时优先于 `intra_with_pcie`。若仅有
`intra_with_pcie`，则链接类型推导为 `"pcie"`（true）或 `"nvlink"`（false）。
两个字段保持同步：`intra_with_pcie` 始终在
`intra_link_type == "pcie"` 时为 `True`。这保证了已有配置的行为完全不变，
同时允许新配置声明 `"intra_link_type": "ublink"` 而无需设置布尔字段。

### intra_node_pcie_8x/intra_node_pcie_4x/intra_node_pcie_2x/low_intra_node/high_intra_node/inter_node   
每一种网络带宽配置，包含以下参数：
- processor_usage: unused, 保留字段     
- bandwidth: 网络带宽配置，包含以下参数
    - efficient_factor: 网络带宽效率
    - gbps: 网络带宽
    - latency_us: 网络延迟
- overlay_bandwidth_gbps: 可选，可叠加的并行 fabric 带宽（GB/s）。仅在
  `topology.levels` 成本路径中对 **p2p 和集合通信**
  （all_reduce/all_gather/reduce_scatter/all2all）生效，加到
  `bandwidth.gbps` 上：`有效带宽 = gbps + overlay_bandwidth_gbps`。默认 0。
  示例：UBLink mesh（56 GB/s）设 `overlay_bandwidth_gbps: 224`（SU Clos），
  所有通信模式有效带宽 = 280 GB/s。
- op: 网络带宽效率，包含以下参数
    - all_reduce: all_reduce操作的网络带宽效率
        - scale: 2，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - all_gather: all_gather操作的网络带宽效率
        - scale: 1，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - reduce_scatter: reduce_scatter操作的网络带宽效率
        - scale: 1，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - p2p: p2p操作的网络带宽效率
        - scale: 1，固定
        - offset: 0， 固定
        - efficient_factor， 可选
        - latency_us，可选
    - all2all: all2all操作的网络带宽效率
        - scale: 1，固定
        - offset: -1， 固定
        - efficient_factor， 可选
        - latency_us，可选 

例如A100_PCIE相邻两卡通信带宽详细配置：
```json
"intra_node_pcie_2x": {
            "processor_usage": 0.00,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 30,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1,
                    "efficient_factor": 0.6965,
                    "latency_us": 15.51
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6866,
                    "latency_us": 24.84
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6419,
                    "latency_us": 131.30
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6969,
                    "latency_us": 24.07
                }
            }

        },
```

## engines

可选字典，为 DES 资源模型声明额外的硬件引擎 lane，例如：

```json
"engines": {
    "cube": {"peak_tflops": 320},
    "vector": {"peak_tflops": 80}
}
```

- 缺省表示单引擎，与当前行为一致。
- 引擎名必须是合法标识符，且不能与保留 lane 名 `comp`、`comm`、
  `pp_fwd`、`pp_bwd`、`off` 冲突。
- vector lane 的成本目前使用按峰值折算的解析估计（见
  [design_simu_kind_resource_model-zh.md](./design_simu_kind_resource_model-zh.md)
  决策记录 9.1），因此每个引擎条目只携带 `peak_tflops` 等标量峰值；
  之后可以在不改变接口的前提下补充实测效率表。

## fabric_model / topology（Preview）

可选字段，用于启用 DES 的网络 fabric 竞争建模（见
[design_simu_network_fabric-zh.md](./design_simu_network_fabric-zh.md)
第 6 节）：

```json
"fabric_model": "nic",
"topology": {
    "tor_capacity_gbps": 1600,
    "tor_node_share": "auto"
}
```

- `fabric_model`：缺省/`null`（默认）表示关闭 fabric 建模，与当前行为一致；
  `"nic"` 启用每 GPU 的 NIC 服务器，使 `inter_node` 通信在所属 rank 的
  NIC 上排队；`"nic+tor"` 在此基础上进一步激活 ToR（top-of-rack）服务器
  （Preview）。
- `fabric_model: "nic+levels"`（Preview）保留每 GPU 的 NIC 服务器，并额外
  为 `topology.levels` 中声明的每个 (level, unit) 激活一个逻辑链路服务器
  ——`(pod, pod_id)`、`(rack, rack_id)` 等——容量取自该层级的 net 配置，
  由该单元内的活跃成员共享（`node_share` 泛化为 `level_share`：在
  `merge_lanes` 下每层的放大系数 = 单元内活跃 rank 数 / 被模拟 rank 数，
  即单元跨度）。该模式要求声明 `topology.levels`（有校验）。链路占用采用
  与 NIC/ToR 服务器相同的按大小计费公式——一个算子按其容量份额占用链路
  的整个传输时长，因此同样存在超额计费的注意事项。详见
  [design_simu_hierarchical_network-zh.md](./design_simu_hierarchical_network-zh.md)
  第 8 节。
- `topology.tor_capacity_gbps`：ToR 服务器容量，默认取 `inter_node.gbps`
  （节点上行带宽）。
- `topology.tor_node_share`：`"auto"` 或不小于 1 的数字。`"auto"` 在
  `merge_lanes` 下解析为 `num_per_node`（此时每个节点只模拟一个 rank，
  否则 ToR 服务器只能看到节点真实流量的 1/num_per_node），否则解析为 `1`。
- `topology` 只有与 `fabric_model` 一起设置才有意义；在 `fabric_model`
  缺省时设置 `topology` 会触发警告。

## topology.levels / composition_policy（Preview）

可选的多级网络拓扑声明（
[design_simu_hierarchical_network-zh.md](./design_simu_hierarchical_network-zh.md)
的 Phase 1）。它用于建模具有多层链路层级的集群——节点内 N 张 GPU 走链路
A、每个 pod 内 M 个节点走链路 B、每个 rack 内 P 个 pod 走链路 C——并让每个
通信域按其实际跨越的层级计费。当 `topology.levels` 缺省时，结果与扁平的
两级模型逐比特一致。同一声明也支撑 `fabric_model="nic+levels"` 的 fabric
层级服务器（见上一节及设计文档第 8 节）；单独声明时仅作用于解析成本路径。

```json
"topology": {
    "levels": [
        {"name": "node", "size": 8,   "net": "high_intra_node"},
        {"name": "pod",  "size": 32,  "net": "inter_node"},
        {"name": "rack", "size": 256, "net": "inter_rack"}
    ],
    "composition_policy": {"all2all": "max", "collectives": "serial", "p2p": "serial"}
}
```

Schema 规则：

- `levels` 按从内到外的顺序排列。`size` 表示本层级的一个单元包含多少个
  上一层级的单元（`node.size=8` ⇒ 每节点 8 张 GPU；`pod.size=32` ⇒ 每
  pod 32 个节点 = 256 张 GPU；`rack.size=256` ⇒ 每 rack 256 个 pod）。
  第一层级的"单元"是一张 GPU。
- 第一层级必须是节点层级，其 `size` 必须等于 `num_per_node`（有校验），
  从而保持所有已有的基于 `num_per_node` 的修正逻辑一致。
- `net` 指向现有 `networks` 字典中的条目——`networks` 的 schema 不变；
  新增诸如 `inter_rack` 的链路配置只是数据。每一层的带宽/延迟/拟合算子
  系数都来自该 net 条目。
- `composition_policy` 设置按集合通信类型的成本组合方式；未指定的条目
  取图中所示默认值（`all2all` → `max`，层级集合通信 → `serial`，
  `p2p` → `serial`）。

net 字段语义 C（对每个 strategy 通信族 `tp_net`、`pp_net` 等生效）：

| strategy net 字段 | 有 `topology.levels` 时 | 无 `topology.levels` 时 |
|---|---|---|
| `"auto"`（默认） | 解析为伪 net `"levels"`：按通信组的层级构成逐级组合成本 | 现有的二值解析（`high_intra_node` / `inter_node`，或 pcie 变体） |
| 显式设置（如 `"inter_node"`） | 该通信族走 legacy 单 net 路径——作为逃生舱（强制最坏情况分析或模拟 rank 重映射） | 与之前一致 |

组合策略：

- `serial`（层级集合通信 `all_reduce` / `all_gather` /
  `reduce_scatter`）：集合通信被分解为每层级一个相位，总时间为各相位
  时间之和，每个相位使用该层级的 net 配置与该层级的子组大小——与分层
  NCCL 行为一致（节点内 reduce → pod 内 all_reduce → rack 内
  all_reduce → …）。
- `max`（`all2all`）：每对通信的时间受限于其路径上最慢的链路，因此总
  时间为各层级传输时间的最大值。
- `p2p`：在两个端点路径上的各层级按 `serial` 累加。

示例——使用上述三个层级时，一个 32 成员的通信组（如默认放置下的
`dp=32`）分解为 `[2, 8, 2]`：每节点 2 个成员、每 pod 8 个节点、每 rack
2 个 pod（2 × 8 × 2 = 32）。对该组的 `serial` all_reduce 会把三个相位
相加（节点内 k=2、pod 内 k=8、rack 内 k=2）；`all2all` 则取三个层级
时间的最大值。每个通信域的层级比例来自放置策略（默认顺序
`tp-cp-ep-dp-pp`，从内到外）；映射与成本数学详见设计文档第 3、5–7 节。

## operator_efficiency（可选）

面向解析成本模型的按算子效率表（
[design_simu_cost_model_tunability-zh.md](./design_simu_cost_model_tunability-zh.md)
Phase 1）。用于在不改动共享的、按 op 名组织的
`accurate_efficient_factor` 条目的前提下，对单个算子的效率进行调优。

每个 key 可以是 **class key**（模块类名，如 `"LinearCol"`、
`"ParallelCE"`，或实例级 `cost_op_key`，若已设置），也可以是
**path key**（从模型根开始的模块路径，如 `"layer_3.mlp"`）。key
对应的值可以是标量（该 key 的默认效率），也可以是带 `default`
和逐 shape `shapes` 条目的字典：

```json
"operator_efficiency": {
    "ParallelCE": 0.52,
    "LinearCol": {
        "default": 0.60,
        "shapes": {
            "b=1, m=4096, k=8192, n=10240, layout=TN, accumulate=False, out_dtype=bf16": 0.66
        }
    },
    "layer_3.mlp": {"default": 0.55}
}
```

### Shape 描述符

逐 shape 条目（`shapes` 块以及第 1/3/5 级查找）以各模块在运行时
生成的 shape 描述符为键。当前产生的格式有：

- GEMM（LinearBase 子类）：现有的
  `b=, m=, k=, n=, layout=, accumulate=, out_dtype=` 格式，例如
  `b=1, m=4096, k=8192, n=10240, layout=TN, accumulate=False, out_dtype=bf16`。
- sdp / 注意力（CoreAttention、MLA 变体）：
  `batch=, seq_len=, head_num=, kv_head_num=, qk_head_dim=, v_head_dim=, qkv_contiguous=`，
  例如 `batch=1, seq_len=4096, head_num=32, kv_head_num=8, qk_head_dim=128, v_head_dim=128, qkv_contiguous=True`。
  这与 `simu_tools/efficency_test/test_fa_efficiency.py` 生成的
  shape key 完全一致，因此实测条目可以原样粘贴回去并真正命中。
- 以 `default` op 计成本的 elementwise / norm 算子（LayerNorm、
  Swiglu、Gelu）：轻量 `b=, s=, h=` 描述符，例如
  `b=1, s=4096, h=8192`。其中 `h` 对 LayerNorm 是 hidden size，
  对 Swiglu/Gelu 是其所作用的 ffn/中间维度。

### 查找链

成本计算时首个命中的生效：

| 层级 | 键 | 来源 |
|---|---|---|
| 1 | `(path_key, shape_desc)` | overrides 链 |
| 2 | `path_key` | overrides 链 |
| 3 | `(class_key, shape_desc)` | overrides 链 |
| 4 | `class_key` | overrides 链 |
| 5 | `(op_name, shape_desc)` | `accurate_efficient_factor`（现有） |
| 6 | `op_name` | `efficient_factor`（现有） |
| 7 | `default` op | 现有兜底 |

"overrides 链" 内部的优先级（从高到低）：

1. API overrides —— `PerfBase.configure(..., efficiency_overrides={...})`
2. strategy 的 `efficiency_overrides`（见
   [strategy-zh.md](./strategy-zh.md#efficiency_overrides)）
3. system 的 `operator_efficiency`（本字段）

override key 会在 `run_estimate()` 构建模型后校验；匹配不到任何模块的
key 会抛出 `ValueError`，而不是静默无效。

### 命中 / 未命中归因与实测闭环

每次查找都会在 `hit_efficiency` 中记录命中的层级和来源；未命中的记录在
`miss_efficiency` 中，按 class key / path key 分组并带层级标签，因此报告
直接指出应该实测哪个算子。闭环流程：运行 → 查看 `miss_efficiency` → 用
[simu_tools/efficency_test](../simu_tools/efficency_test/README.md)
实测被点名的算子 → 把结果填入 `operator_efficiency`（op 名级别的条目则填入
`accurate_efficient_factor`）。

由于模块现在会在运行时生成 shape 描述符，`miss_efficiency` 中记录的
sdp 未命中带有真实 shape key，格式与 `test_fa_efficiency.py` 的实测
格式完全一致：把未命中的 sdp shape key 复制到测量脚本的 shape 列表中
运行，再把得到的条目粘贴回 `accurate_efficient_factor`（或某个
`shapes` override）——模拟器下次运行会产生完全相同的 key，实测值
必然命中。
