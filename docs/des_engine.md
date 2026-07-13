# DES 引擎：多资源队列过程级性能预测

## 目录

- [1. 概述](#1-概述)
- [2. 架构设计](#2-架构设计)
- [3. 资源队列模型](#3-资源队列模型)
- [4. 数据依赖与调度](#4-数据依赖与调度)
- [5. 1F1B 多微批次调度](#5-1f1b-多微批次调度)
- [6. 掩盖可观测性](#6-掩盖可观测性)
- [7. Chrome Tracing 导出](#7-chrome-tracing-导出)
- [8. 使用手册](#8-使用手册)
- [9. API 参考](#9-api-参考)

---

## 1. 概述

DES（Discrete-Event Simulation）引擎是 SimuMax 的过程级性能预测子系统。它在不运行真实训练任务的前提下，通过多资源队列离散事件模拟，输出以下信息：

| 输出 | 文件 | 用途 |
|------|------|------|
| 每个 rank 的时间线 | `des_tracing_logs.json` | Chrome Tracing 可视化 |
| 掩盖可观测性报告 | `overlap_report.json` | 终端打印 + JSON 报告 |
| 每个模块的暴露通信时间 | `ModuleCostInfo.net_exposed_time` | 回填到分析路径 |

### 核心能力

- **多资源并行**：COMPUTE（计算）和 INTRA_LINK（TP 通信）/ INTER_LINK（PP 通信）作为独立队列，天然支持并行
- **数据依赖建模**：compute → comm → next_compute 的 DAG 依赖链
- **1F1B 多微批次**：复用 `calculate_1f1b_bubble` 的 warmup → 1F1B → cooldown 调度
- **PP P2P 通信**：跨 stage 的 send/recv 事件注入
- **CostModel 可插拔**：每个算子可绑定不同的成本计算模型
- **Chrome Tracing 兼容**：输出格式与现有 `tracing_logs.json` 兼容

---

## 2. 架构设计

```
用户调用入口
│
├─ PerfLLM.run_estimate_with_overlap()
│   ├─ self.run_estimate()          # 构建模型 + 计算每模块成本
│   ├─ self._apply_cost_models()    # CostModel 注入
│   ├─ self._run()                  # 执行完整的前向/反向计算
│   └─ DesBridge.from_module_costs() # 构建 DES 时间线
│       ├─ calculate_1f1b_bubble()  # 获取 1F1B 偏移 (PP > 1)
│       ├─ _schedule_leaves_pass()  # 遍历 all_leaf_nodes 调度事件
│       └─ _schedule_inter_comm_at() # 注入 PP P2P 事件
│
├─ des.compute_overlap()            # OverlapTracker 扫描线法
├─ des.export_chrome_tracing()      # 导出 Chrome Tracing JSON
└─ OverlapReport.generate()         # 导出掩盖报告
```

### 文件结构

```
simumax/core/
├── des_engine.py       # MultiResourceDES + OverlapTracker + ResourceQueue
├── des_bridge.py       # DesBridge: 从 PerfLLM 构建 DES 时间线
├── overlap_report.py   # OverlapReport: 掩盖可观测性报告生成
├── cost_model.py       # CostModel 接口 + 内置实现
└── perf_llm.py         # PerfLLM.run_estimate_with_overlap()
```

---

## 3. 资源队列模型

### 3.1 资源类型

每个 GPU rank 维护 3 条独立的资源队列：

| 资源 | 常量 | 说明 |
|------|------|------|
| 计算 | `ResourceType.COMPUTE` | GPU 计算单元 |
| 节点内通信 | `ResourceType.INTRA_LINK` | NVLink / PCIe，用于 TP all-reduce |
| 节点间通信 | `ResourceType.INTER_LINK` | IB / RoCE，用于 PP P2P |

每条队列是 FIFO 的 `ResourceQueue`，维护 `current_time` 和 `events` 列表。`schedule(duration)` 在队尾追加事件并推进 `current_time`，`advance_to(t)` 用于空转等待（表示资源空闲）。

### 3.2 同一 rank 内多条队列并行

COMPUTE 和 COMM（INTRA_LINK / INTER_LINK）是独立队列，各自独立推进时间。这意味着：

```
COMPUTE:    [LN][LC:QKV]          [CA][LR:out]          [LN][LC:fc1]    ...
COMM:             [all-reduce]           [all-reduce]          [all-reduce]
```

当计算和通信在不同的队列上调度时，它们的时间线可以重叠。关键在于数据依赖是否允许这种重叠——详见下一节。

### 3.3 多 rank

`MultiResourceDES(num_ranks=N)` 为每个 rank 创建独立的资源队列集合：

```
rank0: { COMPUTE: Queue, INTRA_LINK: Queue, INTER_LINK: Queue }
rank1: { COMPUTE: Queue, INTRA_LINK: Queue, INTER_LINK: Queue }
...
```

跨 rank 的通信同步通过 `_advance_all_lanes_to` 和 `_schedule_inter_comm_at` 实现。

---

## 4. 数据依赖与调度

### 4.1 单 Micro-Batch 内的 DAG

Llama 解码器层的前向 DAG：

```
LayerNorm → LinearCol(QKV) → all-reduce → CoreAttention → LinearRow → all-reduce
                                 │
                                 └──→ LayerNorm → LinearCol(fc1) → all-reduce → Swiglu → LinearRow(fc2) → all-reduce
```

每个 `LinearCol` / `LinearRow` 的 all-reduce 输出是下一个算子的输入。这是严格的 `compute → comm → next_compute` 依赖链。

### 4.2 `_schedule_leaves_pass` 调度算法

```python
pending_comm_end = 0.0

for leaf in all_leaf_nodes:       # forward order
    comp_time = leaf.fwd_compute_time
    net_time  = leaf.fwd_net_time

    # 决策：是否需要等上一个通信？
    if not (allow_overlap and leaf in OVERLAP_TYPES):
        comp_q.advance_to(pending_comm_end)   # ← 阻塞判断

    # 1. 调度计算
    comp_q.schedule(comp_time)

    # 2. 如果有通信
    if net_time > 0:
        comm_q.advance_to(comp_q.current_time)  # comm 等 compute
        comm_q.schedule(net_time)
        pending_comm_end = comm_q.current_time  # 延迟屏障
    else:
        pending_comm_end = 0.0
```

**关键点**：

- `comp_q.advance_to(pending_comm_end)` 是**延迟屏障**：上一个 leaf 的通信完成后，才让当前 leaf 的 compute 开始
- `comm_q.advance_to(comp_q.current_time)` 确保通信不先于计算开始
- `pending_comm_end` 只在有通信的 leaf 后才设为非零值，下一个 leaf 用它判断是否需要等待

### 4.3 Overlap 模式

当 `allow_overlap=True` 且当前 leaf 类型在 `_OVERLAP_LEAF_TYPES = {"Swiglu", "LayerNorm"}` 中时，跳过 `advance_to(pending_comm_end)`。这使得 Swiglu 的 compute 可以与上一个 LinearCol 的 all-reduce 并行执行。

**适用场景**：Parallel MLP 架构（如 PaLM）中，gate 分支和 up 分支的 Swiglu 互不依赖对方的 all-reduce 输出。

### 4.4 调度日志

`_schedule_leaves_pass` 内置详细日志，每个 leaf 打印一行：

```
[BLOCK] CoreAttention    comp_q 2 → 2 μs  (waited 0 μs for comm at 2 μs)
[SKIP] Swiglu           comp_q=5 μs  ignoring pending_comm_end=6 μs  (overlap leaf)
[COMM] LinearCol         compute_end=5 μs  comm: 4 → 6 μs  pending_comm_end=6 μs
```

| 标签 | 含义 |
|------|------|
| `[BLOCK]` | comp_q 被 `advance_to(pending_comm_end)` 阻塞 |
| `[SKIP]` | 因 overlap leaf 跳过了阻塞 |
| `[SYNC]` | comp_q 已 ≥ pending_comm_end，无需阻塞 |
| `[COMM]` | 调度了 TP 通信事件 |
| `[COMP]` | 纯计算 leaf，无通信 |

---

## 5. 1F1B 多微批次调度

### 5.1 调度流程

当 `pp_size > 1` 且 `micro_batch_num > 1` 时，`from_module_costs` 使用两级调度：

1. **1F1B 偏移计算**：调用 `PerfLLM.calculate_1f1b_bubble()` 获取 per-rank per-mb per-kind 的 start 时间
2. **按时间排序所有 op**：将 (rank, mb, kind) 按 start 时间排序
3. **逐个调度**：每个 (rank, mb, kind) 在指定时间点注入对应的 `_schedule_leaves_pass`
4. **PP P2P 注入**：在所有 compute op 调度完成后，对每对相邻 stage 的相同 (mb, kind) 注入 P2P 事件

### 5.2 示例（PP=2, mbc=4）

```
rank0 (stage 0):  F0  F1  B0  F2  B1  F3  B2  B3
rank1 (stage 1):      F0  B0  F1  B1  F2  B2  F3  B3
```

warmup → 1F1B 稳态 → cooldown 的三段式结构反映在偏移表中：
- rank0 的 F0 在 t=0（warmup）
- rank0 的 B3 在最后（cooldown）

### 5.3 TP + PP 混合

当同时启用 TP 和 PP 时（如 TP=2, PP=2），共 4 个 rank：

- rank 0, 1：first_stage（同一个 TP group，共享 1F1B 偏移）
- rank 2, 3：last_stage（同一个 TP group）

同一 stage 内的 TP rank 处理相同的模型 chunk（因为 TP 拆分的是张量，不是层）。

---

## 6. 掩盖可观测性

### 6.1 扫描线法

`OverlapTracker.compute_overlap()` 对每个 rank 的事件时间线执行扫描线算法：

1. 取出所有 COMPUTE 事件和所有 COMM 事件（INTRA_LINK + INTER_LINK）
2. 按 `(start_time, +1)` 和 `(end_time, -1)` 构建端点列表
3. 扫描排序后的端点，跟踪 COMPUTE 和 COMM 的活跃计数
4. 当两者同时活跃时记录重叠区间

### 6.2 报告输出

`OverlapReport.print_summary()` 打印到终端：

```
================================================================
  Overlap Summary
================================================================
  Compute utilization:     100.0%
  Intra-link utilization:  100.0%
  Inter-link utilization:  0.0%
  Overall overlap ratio:   65.3%
  Total compute time:      0.9992 ms
  Total comm time:         0.5497 ms
  Overlapped comm time:    0.3592 ms
  Exposed comm time:       0.1905 ms
  Iteration time:          0.2780 ms
----------------------------------------------------------------
  Top exposed modules:
    self.layer_0.attention.linear_qkv: 0.0068 ms exposed
    ...
----------------------------------------------------------------
  Communication breakdown:
    LinearCol_allreduce: total=0.3262 ms, overlapped=0.2140 ms (66%)
    LinearRow_allreduce: total=0.2209 ms, overlapped=0.1441 ms (65%)
================================================================
```

`OverlapReport.generate(output_dir)` 生成 `overlap_report.json`，包含按模块和通信类型的结构化数据。

### 6.3 暴露时间回填

`backfill_exposed_times()` 将 DES 计算的 `fwd_exposed_time` / `bwd_exposed_time` 回填到 `ModuleCostInfo.net_exposed_time` 字段，供分析路径使用。

---

## 7. Chrome Tracing 导出

### 7.1 导出格式

`MultiResourceDES.export_chrome_tracing(output_dir)` 生成 Chrome Tracing 兼容的 JSON 数组：

```json
{
  "name": "LinearCol_allreduce",
  "cat": "comm",
  "ph": "X",
  "ts": 1.0,          // 微秒
  "dur": 0.8,         // 微秒
  "pid": "rank0",     // 多 rank 隔离
  "tid": "comm",      // lane: compute | comm
  "args": {
    "module_path": "self.layer_5.attention.linear_qkv",
    "stage": "fwd_mb2",
    "resource": "intra_link"
  }
}
```

### 7.2 Lane 布局（per rank）

| tid | 内容 | sort_index |
|-----|------|------------|
| `compute` | 所有 COMPUTE 事件（fwd + bwd 合并在同一泳道） | 0 |
| `comm` | 所有 INTRA_LINK + INTER_LINK 事件 | 1 |

### 7.3 输出目录

默认输出到 `./output/YYYYMMDD_HHMMSS/`，可通过 `output_dir` 参数自定义。

### 7.4 可视化

导出的 JSON 文件可直接拖入 Chrome 浏览器地址栏 `chrome://tracing` 中查看。每个 rank 显示为一个 process，内部有 compute 和 comm 两条 thread lane 并列。

---

## 8. 使用手册

### 8.1 快速开始

```python
from simumax.core.perf_llm import PerfLLM
from simumax.utils import get_simu_model_config, get_simu_strategy_config, get_simu_system_config

perf = PerfLLM()
perf.configure(
    model_config=ModelConfig.init_from_config_file(get_simu_model_config('llama3-8b')),
    strategy_config=StrategyConfig.init_from_config_file(get_simu_strategy_config('tp2_pp1_dp4_mbs1')),
    system_config=SystemConfig.init_from_config_file(get_simu_system_config('a100_pcie')),
)

# 运行 DES 估计（自动导出 tracing + 掩盖报告）
perf.run_estimate_with_overlap()
```

### 8.2 指定输出目录

```python
perf.run_estimate_with_overlap(output_dir="my_results/tp2_test")
```

### 8.3 多微批次（PP 场景）

```python
perf.run_estimate()
perf.strategy.micro_batch_num = 8  # 覆盖 micro_batch_num
perf.run_estimate_with_overlap()
```

### 8.4 自定义 CostModel

```python
from simumax.core.cost_model import OverrideCostModel, FormulaCostModel

# 固定时间覆盖
perf.set_cost_model("LinearCol", OverrideCostModel(fixed_time_ms=0.05))

# 按路径覆盖
perf.set_cost_model_by_path(
    "self.layer_0.attention.linear_qkv",
    FormulaCostModel(compute_fn=lambda ctx: ctx.flops / (400e12 * 0.8) * 1e3)
)

perf.run_estimate_with_overlap()
```

### 8.5 启用 Overlap 模式

```python
from simumax.core.des_bridge import DesBridge

# 需要手动构造 DES 并使用 allow_overlap=True
des = MultiResourceDES(num_ranks=2)
DesBridge._schedule_leaves_pass(des, rank=0, model=model, pass_dir="fwd", allow_overlap=True)
```

### 8.6 只导出 Tracing

```python
des.export_chrome_tracing(output_dir="my_traces")
```

### 8.7 查看掩盖报告

```python
summary = perf.get_overlap_report()
print(f"Overall overlap: {summary.overall_overlap_ratio:.1%}")
```

### 8.8 典型场景

| 场景 | 配置 | 关键观察 |
|------|------|----------|
| TP2 (纯 TP) | `tp2_pp1_dp4_mbs1` | 每个 rank 处理全量层，含 TP all-reduce |
| PP2 (纯 PP) | `tp1_pp2_dp4_mbs1` | stage 拆分，P2P 通信，1F1B 调度 |
| TP2+PP2 | 手动构造 | 4 rank：2 TP × 2 PP |
| Overlap demo | `allow_overlap=True` | Swiglu/LayerNorm 与 all-reduce 重叠 |

---

## 9. API 参考

### MultiResourceDES

```python
class MultiResourceDES:
    def __init__(self, num_ranks: int = 1)
    def schedule_compute(self, rank, duration, op_name, module_path, stage) -> ResourceEvent
    def schedule_intra_comm(self, ranks, duration, op_name, module_path, stage) -> None
    def schedule_inter_comm(self, ranks, duration, op_name, module_path, stage) -> None
    def sync_rank_lanes(self, rank) -> None
    def compute_overlap(self) -> OverlapSummary
    def get_iteration_time(self) -> float
    def export_chrome_tracing(self, output_dir=None, filename="des_tracing_logs.json") -> str
```

### DesBridge

```python
class DesBridge:
    @staticmethod
    def from_module_costs(perf_model, num_ranks=1) -> MultiResourceDES
    @staticmethod
    def from_simulation_log(log_path, num_ranks=1) -> MultiResourceDES
    @staticmethod
    def _schedule_leaves_pass(des, rank, model, pass_dir, mb=0, allow_overlap=False) -> None
```

### OverlapReport

```python
class OverlapReport:
    @staticmethod
    def print_summary(summary: OverlapSummary) -> None
    @staticmethod
    def generate(summary, output_dir=None, filename="overlap_report.json") -> None
```

### PerfLLM

```python
class PerfLLM:
    def run_estimate_with_overlap(self, save_path=None, output_dir=None) -> OverlapSummary
    def get_overlap_report(self) -> Optional[OverlapSummary]
    def set_cost_model(self, module_class_or_name, cost_model) -> None
    def set_cost_model_by_path(self, path, cost_model) -> None
```

### 事件数据结构

```python
@dataclass
class ResourceEvent:
    resource: ResourceType
    start_time: float       # ms
    end_time: float         # ms
    op_name: str
    module_path: str
    stage: str
    rank: int = 0
    size_bytes: int = 0
    flops: int = 0

@dataclass
class OverlapSummary:
    per_module: Dict[str, ModuleOverlapStats]
    per_comm_type: Dict[str, CommOverlapStats]
    total_compute_time: float
    total_comm_time: float
    total_exposed_comm_time: float
    overall_overlap_ratio: float
    iteration_time: float
```

---

## 依赖关系

```
cost_model.py  ──→  base_struct.py (MetaModule._cost_model)
des_engine.py  ──→  des_bridge.py  ──→  perf_llm.py
                                  └──→  overlap_report.py
```

- `cost_model.py` 可独立使用，在 `base_struct.py` 的 `_comp_cost_info_impl` 中有集成点
- `des_engine.py` 是独立的多资源队列引擎，无 SimuMax 特定依赖
- `des_bridge.py` 桥接 PerfLLM 的模块成本信息到 DES 引擎
- `overlap_report.py` 消费 `OverlapSummary` 生成可读输出
