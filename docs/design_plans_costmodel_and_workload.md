# 设计方案：算子 CostModel 可插拔化 & DES 过程级性能预测 & 模型负载接入变更

## 目录

- [Plan 1: 算子 CostModel 可插拔接口](#plan-1-算子-costmodel-可插拔接口)
  - [1.1 现状分析](#11-现状分析)
  - [1.2 设计目标](#12-设计目标)
  - [1.3 接口设计](#13-接口设计)
  - [1.4 内置实现](#14-内置实现)
  - [1.5 配置格式](#15-配置格式)
  - [1.6 模块集成方案](#16-模块集成方案)
  - [1.7 影响范围与兼容性](#17-影响范围与兼容性)
  - [1.8 分阶段交付计划](#18-分阶段交付计划)
- [Plan 2: DES 过程级性能预测与掩盖可观测性](#plan-2-des-过程级性能预测与掩盖可观测性)
  - [2.1 现状分析](#21-现状分析-1)
  - [2.2 设计目标](#22-设计目标-1)
  - [2.3 核心架构：多资源队列 DES 引擎](#23-核心架构多资源队列-des-引擎)
  - [2.4 掩盖可观测性系统](#24-掩盖可观测性系统)
  - [2.5 DP/Optimizer 并发集成](#25-dpoptimizer-并发集成)
  - [2.6 分析路径增强](#26-分析路径增强)
  - [2.7 影响范围与兼容性](#27-影响范围与兼容性)
  - [2.8 分阶段交付计划](#28-分阶段交付计划)
- [Plan 3: 模型负载接入变更](#plan-3-模型负载接入变更)
  - [3.1 现状分析](#31-现状分析)
  - [3.2 设计目标](#32-设计目标)
  - [3.3 Phase A: 结构可配置化](#33-phase-a-结构可配置化)
  - [3.4 Phase B: 外部模型描述导入](#34-phase-b-外部模型描述导入)
  - [3.5 影响范围与兼容性](#35-影响范围与兼容性)
  - [3.6 分阶段交付计划](#36-分阶段交付计划)

---

## Plan 1: 算子 CostModel 可插拔接口

### 1.1 现状分析

当前算子成本计算链路：

```
Module._comp_cost_info()                    # 硬编码 op_name（如 "matmul"）
  → _comp_cost_info_impl(fwd_op, ...)       # base_struct.py:821
    → compute_details(op_name, stage, flops, mem)
      → SystemConfig.compute_op_accuracy_time(op_name, flops, shape_desc)
        → accelerator.op[op_name] → CompOpConfig
        → CompOpConfig.accurate_efficient_factor[shape_desc] 或 efficient_factor
        → time = flops / (tflops × 1e12 × eff) × 1e3
      → SystemConfig.compute_mem_access_time(op_name, mem_bytes)
        → accelerator.bandwidth[op_name] → BandwidthConfig
      → SystemConfig.compute_end2end_time(compute_time, mem_time)
        → roofline: max(compute, mem) 或 only_compute
```

**核心问题**：

| 问题 | 位置 | 说明 |
|------|------|------|
| op_name 硬编码 | `dense_module.py:485-499`, `moe_module.py:1032-1046` 等 | 每个模块类的 `_comp_cost_info()` 写死算子名字符串 |
| 成本公式固定 | `config.py:854` | `time = flops / (tflops * eff)` 是唯一计算路径 |
| roofline 模型固定 | `config.py:1019-1035` | 只支持 `max(compute, mem)` 和 `only_compute` 两种模式 |
| 无实例级覆盖 | 全局 | 同一类的所有实例共享相同的成本查找路径 |
| Permutation 绕过接口 | `moe_module.py:495,798` | 直接调用 `system.compute_mem_access_time()`，不走 `_comp_cost_info_impl` |

**现有算子名全集**（9 种 compute + 4 种 bandwidth + 5 种 network）：

- Compute: `default`, `matmul`, `fp8_matmul`, `sdp_fwd`, `sdp_bwd`, `group_matmul`, `fp8_group_matmul`, `ce`, `ce_fusion`
- Bandwidth: `default`, `permute_fwd`, `permute_bwd`, `ce`
- Network: `all_reduce`, `all_gather`, `reduce_scatter`, `p2p`, `all2all`

### 1.2 设计目标

1. **可插拔**：每个算子可绑定不同的 CostModel 实现（查表、公式、外部回调）
2. **实例级覆盖**：同一模块类的不同实例可使用不同 CostModel
3. **向后兼容**：不修改任何现有 JSON 配置即可得到与当前完全一致的结果
4. **可序列化**：CostModel 配置可通过 JSON 描述和加载
5. **最小侵入**：不改变现有 `_comp_cost_info_impl` 的调用时机和数据流

### 1.3 接口设计

新增文件 `simumax/core/cost_model.py`：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class CostContext:
    """算子成本计算的输入上下文"""
    op_name: str
    stage: str              # "fwd" | "bwd_grad_act" | "bwd_grad_w" | "recompute"
    flops: int
    accessed_mem: int       # bytes
    shape_desc: str         # 现有形状描述字符串
    element_size: int       # 数据类型字节数
    strategy: Any           # StrategyConfig 引用
    system: Any             # SystemConfig 引用


@dataclass
class CostResult:
    """算子成本计算的输出"""
    compute_time: float     # ms, compute-bound 时间
    mem_time: float         # ms, memory-bound 时间
    end2end_time: float     # ms, 最终合并时间
    details: Dict[str, Any] # 可选的详细信息（tflops, gbps, eff 等）


class CostModel(ABC):
    """算子成本模型抽象基类"""

    @abstractmethod
    def compute(self, ctx: CostContext) -> CostResult:
        """计算算子的执行时间"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """CostModel 的名称，用于注册和日志"""
        ...
```

### 1.4 内置实现

#### 1.4.1 TableLookupCostModel（默认，等价于当前行为）

```python
class TableLookupCostModel(CostModel):
    """
    查表成本模型 — 完全等价于当前 SystemConfig 的
    compute_op_accuracy_time + compute_mem_access_time + compute_end2end_time 链路。
    """

    def __init__(self, op_name: str = None, bandwidth_op_name: str = None):
        self._op_name = op_name
        self._bw_op_name = bandwidth_op_name

    @property
    def name(self) -> str:
        return "table_lookup"

    def compute(self, ctx: CostContext) -> CostResult:
        op_name = self._op_name or ctx.op_name
        bw_name = self._bw_op_name or ctx.op_name

        compute_detail = ctx.system.compute_op_accuracy_time(
            op_name, ctx.flops, ctx.shape_desc, reture_detail=True
        )
        mem_detail = ctx.system.compute_mem_access_time(
            bw_name, ctx.accessed_mem, reture_detail=True
        )
        e2e = ctx.system.compute_end2end_time(
            compute_detail['compute_only_time'],
            mem_detail['io_time'],
        )
        return CostResult(
            compute_time=compute_detail['compute_only_time'],
            mem_time=mem_detail['io_time'],
            end2end_time=e2e,
            details={"compute": compute_detail, "io": mem_detail},
        )
```

#### 1.4.2 FormulaCostModel（公式驱动）

```python
class FormulaCostModel(CostModel):
    """
    基于用户自定义公式的成本模型。
    公式通过 lambda 或注册函数提供，接收 CostContext，返回 (compute_time, mem_time)。
    """

    def __init__(self, compute_fn, mem_fn=None, combine_fn=None, model_name="formula"):
        self._compute_fn = compute_fn
        self._mem_fn = mem_fn or (lambda ctx: 0.0)
        self._combine_fn = combine_fn or (lambda c, m: max(c, m))
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, ctx: CostContext) -> CostResult:
        ct = self._compute_fn(ctx)
        mt = self._mem_fn(ctx)
        return CostResult(
            compute_time=ct,
            mem_time=mt,
            end2end_time=self._combine_fn(ct, mt),
            details={},
        )
```

#### 1.4.3 OverrideCostModel（固定值覆盖）

```python
class OverrideCostModel(CostModel):
    """
    直接返回固定时间值，用于调试或已知延迟的场景。
    """

    def __init__(self, fixed_time_ms: float, model_name="override"):
        self._fixed_time = fixed_time_ms
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, ctx: CostContext) -> CostResult:
        return CostResult(
            compute_time=self._fixed_time,
            mem_time=0.0,
            end2end_time=self._fixed_time,
            details={"source": "fixed_override"},
        )
```

#### 1.4.4 CostModelRegistry（注册中心）

```python
class CostModelRegistry:
    """全局 CostModel 注册中心"""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, cost_model_cls: type):
        cls._registry[name] = cost_model_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> CostModel:
        if name not in cls._registry:
            raise KeyError(f"CostModel '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CostModel:
        """从 JSON 配置创建 CostModel 实例"""
        model_type = config.pop("type")
        return cls.create(model_type, **config)
```

预注册：

```python
CostModelRegistry.register("table_lookup", TableLookupCostModel)
CostModelRegistry.register("formula", FormulaCostModel)
CostModelRegistry.register("override", OverrideCostModel)
```

### 1.5 配置格式

#### 1.5.1 Strategy-level 全局默认（新增 strategy config 字段）

```json
{
    "cost_model_overrides": {
        "matmul": {
            "type": "table_lookup",
            "op_name": "matmul"
        },
        "sdp_fwd": {
            "type": "override",
            "fixed_time_ms": 0.05
        }
    }
}
```

#### 1.5.2 模块实例级覆盖（通过 PerfLLM API）

```python
perf_model = PerfLLM()
perf_model.configure(...)

# 为特定模块类设置 CostModel
perf_model.set_cost_model("LinearCol", FormulaCostModel(
    compute_fn=lambda ctx: ctx.flops / (400e12 * 0.8) * 1e3,
))

# 为特定模块路径设置 CostModel
perf_model.set_cost_model_by_path(
    "GPTModel_0.layer_3.SelfAttention.linear_qkv",
    OverrideCostModel(fixed_time_ms=0.1),
)
```

### 1.6 模块集成方案

#### 1.6.1 MetaModule 层修改

在 `base_struct.py` 的 `MetaModule` 基类中新增可选属性：

```python
class MetaModule:
    def __init__(self, ...):
        ...
        self._cost_model: Optional[CostModel] = None  # 新增

    @property
    def cost_model(self) -> Optional[CostModel]:
        return self._cost_model

    @cost_model.setter
    def cost_model(self, model: CostModel):
        self._cost_model = model
```

#### 1.6.2 `_comp_cost_info_impl` 修改

修改 `base_struct.py:821` 的核心方法，增加 CostModel 分支：

```python
def _comp_cost_info_impl(self, fwd_op="default", bwd_grad_act_op="default",
                         bwd_grad_w_op="default", enable_recompute=False):
    def compute_details(op_name, stage, flops, accessed_mem):
        if self._cost_model is not None:
            ctx = CostContext(
                op_name=op_name, stage=stage, flops=flops,
                accessed_mem=accessed_mem,
                shape_desc=self.get_input_shapes_desc(stage),
                element_size=self.element_size,
                strategy=self.strategy, system=self.system,
            )
            result = self._cost_model.compute(ctx)
            self.set_details(stage, result.details.get("compute", {}),
                             result.details.get("io", {}))
            return result.end2end_time

        # 原有逻辑完全保留作为 fallback
        compute_details_orig = self.system.compute_op_accuracy_time(...)
        io_details_orig = self.system.compute_mem_access_time(...)
        end2end_time = self.compute_end2end_time(...)
        self.set_details(stage, compute_details_orig, io_details_orig)
        return end2end_time

    self._cost_info.fwd_compute_time = compute_details(fwd_op, 'fwd', ...)
    self._cost_info.bwd_grad_act_time = compute_details(bwd_grad_act_op, 'bwd_grad_act', ...)
    self._cost_info.bwd_grad_w_time = compute_details(bwd_grad_w_op, 'bwd_grad_w', ...)
    self._cost_info.recompute_compute_time = self._cost_info.fwd_time if self.enable_recompute else 0
```

#### 1.6.3 PerfLLM 层 API

在 `perf_llm.py` 的 `PerfLLM` 类中新增方法：

```python
def set_cost_model(self, module_class_or_name, cost_model: CostModel):
    """为指定模块类设置 CostModel，在 build() 后生效"""
    self._cost_model_overrides[module_class_or_name] = cost_model

def set_cost_model_by_path(self, path: str, cost_model: CostModel):
    """为指定模块路径设置 CostModel"""
    self._cost_model_path_overrides[path] = cost_model

def _apply_cost_models(self):
    """在 build() 完成后，将 CostModel 注入到匹配的模块实例"""
    for model in self._all_models:
        for leaf in model.all_leaf_nodes:
            # 1. 路径匹配优先
            if leaf.full_name in self._cost_model_path_overrides:
                leaf.cost_model = self._cost_model_path_overrides[leaf.full_name]
                continue
            # 2. 类名匹配
            cls_name = type(leaf).__name__
            if cls_name in self._cost_model_overrides:
                leaf.cost_model = self._cost_model_overrides[cls_name]
```

### 1.7 影响范围与兼容性

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `simumax/core/cost_model.py` | **新增** | CostModel 接口 + 内置实现 + Registry |
| `simumax/core/base_struct.py` | 修改 | `MetaModule.__init__` 加 `_cost_model` 属性；`_comp_cost_info_impl` 加分支 |
| `simumax/core/perf_llm.py` | 修改 | 新增 `set_cost_model` / `set_cost_model_by_path` / `_apply_cost_models` |
| `simumax/core/config.py` | 可选修改 | `StrategyConfig` 新增 `cost_model_overrides` 字段 |
| `simumax/core/__init__.py` | 修改 | 导出 `CostModel`, `CostResult`, `CostContext` 等 |

**向后兼容保证**：

- 当 `_cost_model is None`（默认），`_comp_cost_info_impl` 走原有路径，结果完全一致
- 现有 JSON 配置无需任何修改
- 现有 example 脚本无需任何修改
- `Permutation` / `UnPermutation` 的直接调用方式不受影响（它们不走 `_comp_cost_info_impl`）

### 1.8 分阶段交付计划

#### Stage 1: 核心接口 + TableLookup（最小可用）

- 新增 `cost_model.py`，实现 `CostModel` ABC + `TableLookupCostModel` + `CostResult` + `CostContext`
- 修改 `base_struct.py`：`MetaModule` 加 `_cost_model` 属性，`_comp_cost_info_impl` 加分支
- 修改 `perf_llm.py`：新增 `set_cost_model` API
- 验证：用 `TableLookupCostModel` 替换某个模块，确认结果与原有一致
- 验证：`PYTHONPATH=. python examples/perf_llama3_8b_tp1_pp2.py` 结果不变

#### Stage 2: FormulaCostModel + OverrideCostModel + Registry

- 新增 `FormulaCostModel`, `OverrideCostModel`, `CostModelRegistry`
- 新增 `set_cost_model_by_path` API
- 新增 `StrategyConfig.cost_model_overrides` JSON 字段
- 验证：用 `FormulaCostModel` 为 `CoreAttention` 设置自定义公式
- 验证：用 `OverrideCostModel` 固定某个算子时间，确认生效

#### Stage 3: Permutation 特殊路径统一 + 文档

- 将 `Permutation` / `UnPermutation` 的直接调用改为走 `CostModel` 接口
- 新增 `MemoryAccessCostModel` 用于纯访存算子
- 更新 `docs/` 文档，说明 CostModel 使用方法
- 新增 example 脚本演示自定义 CostModel

---

## Plan 2: DES 过程级性能预测与掩盖可观测性

### 2.1 现状分析

当前 SimuMax 有两条独立的性能预测路径：

| 路径 | 入口 | 优势 | 劣势 |
|------|------|------|------|
| **分析路径** | `PerfLLM.analysis_cost()` | 快速，闭式计算 | 掩盖假设有矛盾，DP/Optimizer 不参与调度 |
| **仿真路径** | `run_simulation()` → `SimuSystem.simu()` | 事件驱动，支持 async P2P | DP/Optimizer 未并发模拟，单 comm lane |

#### 2.1.1 当前仿真引擎架构

```
SimuThread.t = {"comp": float, "comm": float, "off": float, "pp_fwd": float, "pp_bwd": float}

AtomModel._step()  → t["comp"] += fwd_cost       # 计算算子只推进 comp lane
Com._step()        → issue CommEntry on comm lane  # 通信算子走 comm lane
                     t["comm"] = max(t["comm"], end_t)
                     t["comp"] = max(t["comp"], end_t)  # 阻塞通信会合并两个 lane

SimuSystem.simu()  → priority-heap DES, 按 min(lane times) 调度 rank
```

#### 2.1.2 关键问题

| 问题 | 位置 | 说明 |
|------|------|------|
| `net_exposed_time` 始终为 0 | `model_struct.py:324-412` | `ModuleCostInfo` 设计了 exposed 字段但从未填充 |
| TP 通信假设矛盾 | `perf_llm.py:2066` vs `model_struct.py:343` | 模块级假设 100% 掩盖（exposed=0），GBS 报告用全量 net_time |
| DP 无掩盖 | `perf_llm.py:1570` | `dp_comm_exposed_time = dp_comm_time  # no overlap for now` |
| Optimizer 无掩盖 | `perf_llm.py:1503` | `optim_exposed_time = optim_time` |
| DP/Optimizer 不参与仿真 | `simu_runner.py:79-83` | 作为顺序块追加在 pipeline 之后 |
| 单 comm lane | `base_struct.py:1347` | `t["comm"]` 是单一标量，NVLink 和 IB 通信串行化 |
| 阻塞通信强制合并 lane | `base_struct.py:2272-2295` | `t["comp"] = t["comm"] = max(...)` 消除一切掩盖 |
| 无资源利用率追踪 | 全局 | 无法回答"GPU 计算单元利用率多少""NVLink 带宽利用率多少" |
| 无掩盖分解报告 | 全局 | 无法回答"TP all-reduce 被掩盖了多少""哪些通信暴露最多" |

#### 2.1.3 现有仿真能力矩阵

| 掩盖场景 | 分析路径 | 仿真路径 |
|----------|----------|----------|
| TP comm ↔ compute | 假设 100% 掩盖 | 部分：blocking 合并 lane，async 可分离 |
| PP P2P ↔ compute | 无掩盖（全暴露） | async P2P 可掩盖，blocking 不可 |
| DP grad reduce ↔ bwd compute | 无掩盖 | 未仿真 |
| Optimizer ↔ 任何 | 无掩盖 | 未仿真 |
| Comm ↔ Comm（多通信并发） | 未建模 | 单 lane 串行 |
| 部分掩盖（如 70% overlap） | 不支持 | 不支持 |

### 2.2 设计目标

1. **多资源队列**：计算、NVLink 通信、IB 通信、内存带宽作为独立资源队列，天然支持并行
2. **掩盖可观测性**：每个算子、每个模块、每个阶段都能报告掩盖比例和暴露时间
3. **DP/Optimizer 并发**：DP 梯度归约与 backward compute 并发调度，Optimizer 可与 PP bubble 重叠
4. **统一路径**：分析路径和仿真路径共享掩盖模型，消除矛盾
5. **向后兼容**：不修改现有 JSON 配置即可得到与当前一致的结果
6. **增量演进**：不替换现有 `SimuSystem`，而是在其基础上构建更高层抽象

### 2.3 核心架构：多资源队列 DES 引擎

#### 2.3.1 资源队列模型

新增文件 `simumax/core/des_engine.py`：

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import heapq


class ResourceType(Enum):
    COMPUTE = "compute"           # GPU 计算单元
    MEM_BANDWIDTH = "mem_bw"      # HBM 访存带宽
    INTRA_LINK = "intra_link"     # NVLink / PCIe（节点内）
    INTER_LINK = "inter_link"     # IB / RoCE（节点间）
    OFFLOAD = "offload"           # 卸载引擎


@dataclass
class ResourceEvent:
    """资源上的一个事件"""
    resource: ResourceType
    start_time: float             # ms
    end_time: float               # ms
    op_name: str                  # 算子标识
    module_path: str              # 所属模块路径
    stage: str                    # "fwd" | "bwd" | "recompute" | "dp" | "optim"
    size_bytes: int = 0           # 数据量（用于带宽利用率计算）
    flops: int = 0                # 计算量（用于算力利用率计算）


@dataclass
class ResourceQueue:
    """单资源 FIFO 队列，追踪该资源的时间线"""
    resource: ResourceType
    capacity: float = 1.0         # 归一化容量（1.0 = 100%）
    current_time: float = 0.0
    events: List[ResourceEvent] = field(default_factory=list)
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0

    def schedule(self, duration: float, op_name: str, module_path: str,
                 stage: str, size_bytes: int = 0, flops: int = 0) -> ResourceEvent:
        """在资源当前时间点调度一个事件，返回事件对象"""
        start = self.current_time
        end = start + duration
        event = ResourceEvent(
            resource=self.resource, start_time=start, end_time=end,
            op_name=op_name, module_path=module_path, stage=stage,
            size_bytes=size_bytes, flops=flops,
        )
        self.events.append(event)
        self.current_time = end
        self.total_busy_time += duration
        return event

    def advance_to(self, t: float):
        """推进资源时间到 t（表示空闲等待）"""
        if t > self.current_time:
            self.total_idle_time += t - self.current_time
            self.current_time = t
```

#### 2.3.2 多资源 DES 调度器

```python
class MultiResourceDES:
    """
    多资源离散事件调度器。
    每个 rank 拥有独立的资源队列集合。
    通信算子需要多个 rank 在同一资源上 rendezvous。
    """

    def __init__(self, num_ranks: int, resource_config: Dict[ResourceType, float]):
        self.num_ranks = num_ranks
        self.resource_config = resource_config
        # 每个 rank 的资源队列
        self.rank_resources: Dict[int, Dict[ResourceType, ResourceQueue]] = {}
        for r in range(num_ranks):
            self.rank_resources[r] = {
                res: ResourceQueue(resource=res, capacity=cap)
                for res, cap in resource_config.items()
            }
        # 全局事件堆（按时间排序）
        self._event_heap: List[Tuple[float, int, str]] = []
        # 通信 rendezvous 状态
        self._pending_barriers: Dict[str, Dict] = {}
        # 掩盖追踪器
        self.overlap_tracker = OverlapTracker()

    def schedule_compute(self, rank: int, duration: float, op_name: str,
                         module_path: str, stage: str, flops: int = 0):
        """调度计算事件到 rank 的 compute 资源"""
        res = self.rank_resources[rank][ResourceType.COMPUTE]
        event = res.schedule(duration, op_name, module_path, stage, flops=flops)
        self.overlap_tracker.record_event(rank, event)
        heapq.heappush(self._event_heap, (event.end_time, rank, "compute"))

    def schedule_intra_comm(self, ranks: List[int], duration: float, op_name: str,
                            module_path: str, stage: str, size_bytes: int = 0):
        """
        调度节点内通信（TP all-reduce 等）。
        所有参与 rank 的 intra_link 资源需要 rendezvous。
        """
        gid = f"intra_{op_name}_{module_path}_{stage}"
        self._barrier_arrive(gid, ranks, duration, ResourceType.INTRA_LINK,
                             op_name, module_path, stage, size_bytes)

    def schedule_inter_comm(self, ranks: List[int], duration: float, op_name: str,
                            module_path: str, stage: str, size_bytes: int = 0):
        """调度节点间通信（PP P2P、DP all-reduce 等）"""
        gid = f"inter_{op_name}_{module_path}_{stage}"
        self._barrier_arrive(gid, ranks, duration, ResourceType.INTER_LINK,
                             op_name, module_path, stage, size_bytes)

    def _barrier_arrive(self, gid, ranks, duration, resource_type,
                       op_name, module_path, stage, size_bytes):
        """通信 barrier rendezvous"""
        if gid not in self._pending_barriers:
            self._pending_barriers[gid] = {
                "expected": len(ranks), "arrived": 0,
                "ready_times": {}, "duration": duration,
                "resource_type": resource_type, "op_name": op_name,
                "module_path": module_path, "stage": stage,
                "size_bytes": size_bytes, "ranks": ranks,
            }
        barrier = self._pending_barriers[gid]
        # 每个 rank 的 ready time 是其资源队列的当前时间
        for r in ranks:
            res = self.rank_resources[r][resource_type]
            barrier["ready_times"][r] = res.current_time
        barrier["arrived"] += len(ranks)

        if barrier["arrived"] >= barrier["expected"]:
            # 所有参与者到达，计算完成时间
            max_ready = max(barrier["ready_times"].values())
            end_t = max_ready + duration
            for r in ranks:
                res = self.rank_resources[r][resource_type]
                res.advance_to(end_t)
                event = ResourceEvent(
                    resource=resource_type,
                    start_time=barrier["ready_times"][r],
                    end_time=end_t,
                    op_name=op_name, module_path=module_path,
                    stage=stage, size_bytes=size_bytes,
                )
                res.events.append(event)
                self.overlap_tracker.record_event(r, event)
            del self._pending_barriers[gid]
```

#### 2.3.3 与现有 SimuSystem 的集成策略

**不替换，而是桥接**。新增 `DesBridge` 层将现有仿真的事件流转化为多资源时间线：

```python
class DesBridge:
    """
    桥接层：从现有 SimuSystem 的 log 输出构建多资源时间线。
    也可作为独立 DES 引擎的前端，直接接收算子级事件。
    """

    @staticmethod
    def from_simulation_log(log_path: str, strategy, system) -> 'MultiResourceTimeline':
        """解析现有仿真 log，构建多资源时间线"""
        ...

    @staticmethod
    def from_module_costs(perf_model) -> 'MultiResourceTimeline':
        """从 ModuleCostInfo 直接构建（分析路径增强）"""
        ...
```

### 2.4 掩盖可观测性系统

#### 2.4.1 OverlapTracker — 核心追踪器

```python
@dataclass
class OverlapRecord:
    """单条掩盖记录"""
    rank: int
    overlapped_event: ResourceEvent    # 被掩盖的事件（如通信）
    overlapping_event: ResourceEvent   # 掩盖它的事件（如计算）
    overlap_duration: float            # 重叠时长 (ms)
    overlap_ratio: float               # 被掩盖事件被掩盖的比例 (0.0-1.0)


@dataclass
class OverlapSummary:
    """掩盖汇总统计"""
    # 按模块聚合
    per_module: Dict[str, 'ModuleOverlapStats']
    # 按资源聚合
    per_resource: Dict[ResourceType, 'ResourceOverlapStats']
    # 按通信类型聚合
    per_comm_type: Dict[str, 'CommOverlapStats']
    # 全局
    total_compute_time: float
    total_comm_time: float
    total_overlapped_comm_time: float
    total_exposed_comm_time: float
    overall_overlap_ratio: float       # 总通信被掩盖的比例
    compute_utilization: float         # 计算资源利用率
    intra_link_utilization: float      # NVLink 利用率
    inter_link_utilization: float      # IB 利用率


@dataclass
class ModuleOverlapStats:
    """模块级掩盖统计"""
    module_path: str
    fwd_compute_time: float
    fwd_comm_time: float
    fwd_overlapped_time: float
    fwd_exposed_time: float            # = comm_time - overlapped_time
    fwd_overlap_ratio: float
    bwd_compute_time: float
    bwd_comm_time: float
    bwd_overlapped_time: float
    bwd_exposed_time: float
    bwd_overlap_ratio: float


class OverlapTracker:
    """
    掩盖追踪器：记录所有资源事件，计算时间线重叠。
    """

    def __init__(self):
        self._events_by_rank: Dict[int, List[ResourceEvent]] = {}

    def record_event(self, rank: int, event: ResourceEvent):
        self._events_by_rank.setdefault(rank, []).append(event)

    def compute_overlap(self) -> OverlapSummary:
        """
        扫描所有 rank 的事件时间线，计算两两资源之间的时间重叠。
        算法：对每个 rank，按时间排序事件，扫描线法计算重叠区间。
        """
        per_module = {}
        per_resource = {}
        per_comm_type = {}

        for rank, events in self._events_by_rank.items():
            events.sort(key=lambda e: e.start_time)

            # 扫描线法：维护每个资源的活跃事件集合
            # 当 compute 和 comm 同时活跃时，记录重叠
            compute_events = [e for e in events if e.resource == ResourceType.COMPUTE]
            for comm_res in [ResourceType.INTRA_LINK, ResourceType.INTER_LINK]:
                comm_events = [e for e in events if e.resource == comm_res]
                overlaps = self._compute_pairwise_overlap(compute_events, comm_events)
                for ov in overlaps:
                    # 聚合到模块、通信类型
                    ...

        return OverlapSummary(...)

    @staticmethod
    def _compute_pairwise_overlap(events_a, events_b) -> List[OverlapRecord]:
        """扫描线法计算两组事件之间的重叠"""
        results = []
        # 合并所有端点，排序，扫描
        endpoints = []
        for e in events_a:
            endpoints.append((e.start_time, +1, 'a', e))
            endpoints.append((e.end_time, -1, 'a', e))
        for e in events_b:
            endpoints.append((e.start_time, +1, 'b', e))
            endpoints.append((e.end_time, -1, 'b', e))
        endpoints.sort()

        active_a = 0
        active_b = 0
        overlap_start = None

        for t, delta, group, event in endpoints:
            was_overlapping = active_a > 0 and active_b > 0
            if group == 'a':
                active_a += delta
            else:
                active_b += delta
            is_overlapping = active_a > 0 and active_b > 0

            if is_overlapping and not was_overlapping:
                overlap_start = t
            elif was_overlapping and not is_overlapping and overlap_start is not None:
                overlap_dur = t - overlap_start
                results.append(OverlapRecord(
                    rank=0, overlapped_event=event,
                    overlapping_event=event,
                    overlap_duration=overlap_dur,
                    overlap_ratio=overlap_dur / (event.end_time - event.start_time)
                    if (event.end_time - event.start_time) > 0 else 0.0,
                ))
                overlap_start = None

        return results
```

#### 2.4.2 可观测性输出

```python
class OverlapReport:
    """掩盖可观测性报告生成器"""

    @staticmethod
    def generate(summary: OverlapSummary, output_path: str):
        """生成 JSON 格式的掩盖报告"""
        report = {
            "global": {
                "total_compute_time_ms": summary.total_compute_time,
                "total_comm_time_ms": summary.total_comm_time,
                "overlapped_comm_time_ms": summary.total_overlapped_comm_time,
                "exposed_comm_time_ms": summary.total_exposed_comm_time,
                "overlap_ratio": f"{summary.overall_overlap_ratio:.1%}",
                "compute_utilization": f"{summary.compute_utilization:.1%}",
                "intra_link_utilization": f"{summary.intra_link_utilization:.1%}",
                "inter_link_utilization": f"{summary.inter_link_utilization:.1%}",
            },
            "per_module": {
                path: {
                    "fwd_overlap_ratio": f"{stats.fwd_overlap_ratio:.1%}",
                    "fwd_exposed_ms": stats.fwd_exposed_time,
                    "bwd_overlap_ratio": f"{stats.bwd_overlap_ratio:.1%}",
                    "bwd_exposed_ms": stats.bwd_exposed_time,
                }
                for path, stats in summary.per_module.items()
            },
            "per_comm_type": {
                comm_type: {
                    "total_time_ms": stats.total_time,
                    "overlapped_time_ms": stats.overlapped_time,
                    "exposed_time_ms": stats.exposed_time,
                    "overlap_ratio": f"{stats.overlap_ratio:.1%}",
                }
                for comm_type, stats in summary.per_comm_type.items()
            },
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    @staticmethod
    def print_summary(summary: OverlapSummary):
        """打印掩盖摘要到终端"""
        print("=" * 60)
        print("Overlap Summary")
        print("=" * 60)
        print(f"  Compute utilization:    {summary.compute_utilization:.1%}")
        print(f"  Intra-link utilization: {summary.intra_link_utilization:.1%}")
        print(f"  Inter-link utilization: {summary.inter_link_utilization:.1%}")
        print(f"  Overall overlap ratio:  {summary.overall_overlap_ratio:.1%}")
        print(f"  Exposed comm time:     {summary.total_exposed_comm_time:.4f} ms")
        print("-" * 60)
        print("Top exposed modules:")
        sorted_modules = sorted(
            summary.per_module.items(),
            key=lambda x: x[1].fwd_exposed_time + x[1].bwd_exposed_time,
            reverse=True,
        )
        for path, stats in sorted_modules[:10]:
            total_exposed = stats.fwd_exposed_time + stats.bwd_exposed_time
            print(f"  {path}: {total_exposed:.4f} ms exposed "
                  f"(fwd {stats.fwd_overlap_ratio:.0%} / bwd {stats.bwd_overlap_ratio:.0%} overlapped)")
```

#### 2.4.3 回填 ModuleCostInfo.net_exposed_time

利用 `OverlapTracker` 的结果回填 `ModuleCostInfo` 中一直为 0 的 exposed 字段：

```python
def backfill_exposed_times(perf_model, overlap_summary: OverlapSummary):
    """将 DES 计算的 exposed time 回填到 ModuleCostInfo"""
    for model in perf_model.model_chunk_dict.values():
        for leaf in model.all_leaf_nodes:
            stats = overlap_summary.per_module.get(leaf.full_name)
            if stats:
                leaf._cost_info.fwd_net_exposed_time = stats.fwd_exposed_time
                leaf._cost_info.bwd_net_exposed_time = stats.bwd_exposed_time
                leaf._cost_info.recompute_net_exposed_time = 0  # 后续扩展
```

### 2.5 DP/Optimizer 并发集成

#### 2.5.1 DP 梯度归约与 Backward 并发

当前 `OptimizerSimulator` 在所有 backward 完成后才开始 DP 通信。改造为 **bucket-based overlap**：

```python
class DpOverlapScheduler:
    """
    DP 梯度归约与 backward compute 并发调度。
    模拟 Megatron-LM 的 overlap_grad_reduce 行为：
    - 每个 bucket 的梯度在对应层 backward 完成后立即发起 reduce-scatter
    - reduce-scatter 在 inter_link 资源上执行
    - backward compute 在 compute 资源上继续执行后续层
    """

    def __init__(self, strategy, system, des: MultiResourceDES):
        self.strategy = strategy
        self.system = system
        self.des = des
        self.bucket_size = getattr(strategy, 'dp_bucket_size', None)

    def schedule_dp_with_backward(self, rank: int, backward_modules: List,
                                  dp_comm_costs: Dict[str, float]):
        """
        交错调度 backward compute 和 DP reduce-scatter。
        backward_modules: 按反向顺序排列的模块列表
        dp_comm_costs: 模块路径 → DP 通信时间
        """
        accumulated_grad_bytes = 0
        pending_dp_ops = []

        for module in backward_modules:
            # 1. 调度 backward compute
            bwd_cost = module._cost_info.bwd_grad_act_time + module._cost_info.bwd_grad_w_time
            self.des.schedule_compute(
                rank, bwd_cost / 1e3,  # ms → s
                op_name="bwd",
                module_path=module.full_name,
                stage="bwd",
            )

            # 2. 累积梯度，检查是否触发 bucket
            grad_bytes = module._model_info.dense_grad_bytes
            accumulated_grad_bytes += grad_bytes

            if self._should_trigger_bucket(accumulated_grad_bytes):
                dp_cost = self._compute_dp_cost(accumulated_grad_bytes)
                self.des.schedule_inter_comm(
                    ranks=self._dp_group_ranks(rank),
                    duration=dp_cost / 1e3,
                    op_name="reduce_scatter",
                    module_path=f"dp_bucket_{len(pending_dp_ops)}",
                    stage="dp",
                    size_bytes=accumulated_grad_bytes,
                )
                accumulated_grad_bytes = 0

        # 最后一个 bucket
        if accumulated_grad_bytes > 0:
            dp_cost = self._compute_dp_cost(accumulated_grad_bytes)
            self.des.schedule_inter_comm(...)
```

#### 2.5.2 Optimizer 与 PP Bubble 重叠

```python
class OptimizerOverlapScheduler:
    """
    Optimizer 步骤与 PP bubble 重叠调度。
    当 PP stage 处于 bubble 等待时，可以提前开始 optimizer step。
    """

    def schedule_optimizer(self, rank: int, optim_cost_ms: float,
                           pp_bubble_start: float, pp_bubble_end: float):
        """在 PP bubble 期间调度 optimizer"""
        compute_res = self.des.rank_resources[rank][ResourceType.COMPUTE]
        # 如果 compute 资源在 bubble 期间空闲，利用它执行 optimizer
        if compute_res.current_time < pp_bubble_end:
            available = pp_bubble_end - compute_res.current_time
            optim_done = min(optim_cost_ms, available)
            self.des.schedule_compute(
                rank, optim_done / 1e3,
                op_name="optimizer_step",
                module_path="optimizer",
                stage="optim",
            )
```

### 2.6 分析路径增强

#### 2.6.1 统一掩盖模型

替换分析路径中的硬编码假设，使用 `OverlapTracker` 的结果：

```python
# 旧代码 (perf_llm.py:1570)
dp_comm_exposed_time = dp_comm_time  # no overlap for now

# 新代码
if self.strategy.overlap_grad_reduce:
    overlap_info = self._des_bridge.compute_overlap()
    dp_comm_exposed_time = overlap_info.per_comm_type.get(
        "reduce_scatter", CommOverlapStats(exposed_time=dp_comm_time)
    ).exposed_time
else:
    dp_comm_exposed_time = dp_comm_time
```

#### 2.6.2 新增 PerfLLM API

```python
class PerfLLM:
    def run_estimate_with_overlap(self):
        """
        增强的性能预测：使用 DES 引擎计算掩盖后的真实迭代时间。
        替代原有的 analysis_cost() 中的硬编码假设。
        """
        # 1. 正常计算算子级成本
        self.run_estimate()

        # 2. 构建多资源时间线
        bridge = DesBridge.from_module_costs(self)

        # 3. 计算掩盖
        overlap_summary = bridge.overlap_tracker.compute_overlap()

        # 4. 回填 exposed time
        backfill_exposed_times(self, overlap_summary)

        # 5. 重新计算迭代时间（使用真实的 exposed time）
        self._analysis_cost_with_overlap(overlap_summary)

        # 6. 输出掩盖报告
        OverlapReport.print_summary(overlap_summary)
        OverlapReport.generate(overlap_summary, f"{TMP_PATH}/overlap_report.json")

    def get_overlap_report(self) -> OverlapSummary:
        """获取掩盖分析报告（需先调用 run_estimate_with_overlap）"""
        return self._overlap_summary
```

### 2.7 影响范围与兼容性

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `simumax/core/des_engine.py` | **新增** | 多资源队列 DES 引擎 + OverlapTracker |
| `simumax/core/des_bridge.py` | **新增** | 桥接层：从现有仿真/分析路径构建多资源时间线 |
| `simumax/core/overlap_report.py` | **新增** | 掩盖可观测性报告生成器 |
| `simumax/core/base_struct.py` | 可选修改 | `ModuleCostInfo.net_exposed_time` 回填支持 |
| `simumax/core/perf_llm.py` | 修改 | 新增 `run_estimate_with_overlap()` API |
| `simumax/core/simu_runner.py` | 可选修改 | 集成 DES 引擎到仿真路径 |

**向后兼容保证**：

- 所有新增功能通过新 API 入口（`run_estimate_with_overlap()`）触发
- 原有 `run_estimate()` + `analysis()` 路径完全不变
- 原有 `simulate()` 路径完全不变
- DES 引擎作为可选增强层，不影响现有结果

### 2.8 分阶段交付计划

#### Stage 1: 多资源时间线 + OverlapTracker（核心可观测性）

- 新增 `des_engine.py`：`ResourceQueue`, `ResourceEvent`, `MultiResourceDES`, `OverlapTracker`
- 新增 `des_bridge.py`：`DesBridge.from_simulation_log()` 从现有仿真 log 构建多资源时间线
- 新增 `overlap_report.py`：`OverlapReport` 报告生成
- 验证：对现有仿真输出运行 OverlapTracker，生成掩盖报告
- 验证：确认 compute/comm 重叠区间与 Chrome Tracing 可视化一致

#### Stage 2: 分析路径集成 + exposed time 回填

- `DesBridge.from_module_costs()` 从 ModuleCostInfo 构建时间线
- `backfill_exposed_times()` 回填 `net_exposed_time`
- `PerfLLM.run_estimate_with_overlap()` API
- 验证：Llama3-8B TP1PP2 的 `run_estimate_with_overlap()` 结果与 `analysis()` 一致（无掩盖时）
- 验证：DeepSeek-V2 EP4PP2 的掩盖报告合理

#### Stage 3: DP/Optimizer 并发调度

- `DpOverlapScheduler`：bucket-based DP 梯度归约与 backward 并发
- `OptimizerOverlapScheduler`：Optimizer 与 PP bubble 重叠
- 集成到 `run_estimate_with_overlap()` 流程
- 验证：开启 `overlap_grad_reduce` 后 DP 暴露时间减少
- 验证：Optimizer 在 bubble 期间执行，总迭代时间减少

#### Stage 4: 仿真路径集成 + 多 comm lane

- 扩展 `SimuThread.t` 支持 `intra_link` / `inter_link` 分离 lane
- TP 通信走 `intra_link`，DP/PP 通信走 `inter_link`
- 验证：仿真路径的 Chrome Tracing 显示多 lane 并行
- 验证：仿真路径和分析路径的掩盖报告一致

---

## Plan 3: 模型负载接入变更

### 3.1 现状分析

当前模型结构的硬编码点：

| 硬编码项 | 位置 | 说明 |
|----------|------|------|
| 只有 RMS Norm | `dense_module.py:798` | `assert norm_type in ["rms_norm"]` |
| Block 结构固定 | `language_model.py:192-197` | `norm → attn → norm → mlp`，不可变 |
| 只有 MHA/GQA + MLA | `language_model.py:140-159` | `attention_type` 只支持 `mha` 和 `mla` |
| 只有 decoder-only | `language_model.py:210-281` | 无 encoder、无 cross-attention |
| 只有 Rotary 位置编码 | `dense_module.py:1806` | `RotaryEmbedding` 是 stub，无其他位置编码 |
| MLA 不支持 TP | `dense_module.py:2583` | `assert strategy.tp_size==1` |
| QKV contiguous 按 sys_name 判断 | `dense_module.py:1154` | `'s5000' in self.system.sys_name` |
| Post-process 固定 | `language_model.py:258-280` | `LayerNorm → LinearCol → ParallelCE` |
| Dense layers 只能在前面 | `language_model.py:255` | `use_dense=(i < dense_layers)` |
| 激活函数只有 SwiGLU/GELU | `dense_module.py:2951-2965` | `use_swiglu` 布尔开关 |
| 无位置编码配置 | 全局 | 没有 `position_embedding_type` 字段 |
| 无异构层配置 | 全局 | 所有层共享同一个 `ModelConfig` |

### 3.2 设计目标

**Phase A（结构可配置化）**：

1. 将 norm 类型、block 结构、位置编码类型从硬编码改为 JSON 可配置
2. 支持更多架构变体（LayerNorm、post-norm、parallel attn+mlp 等）
3. 消除 sys_name 字符串匹配等 hack

**Phase B（外部模型描述导入）**：

4. 支持从 HuggingFace `config.json` 自动导入模型结构
5. 提供导入器注册机制，支持更多格式扩展

### 3.3 Phase A: 结构可配置化

#### 3.3.1 ModelConfig 新增字段

```python
@dataclass
class ModelConfig(Config):
    # ... 现有字段保持不变 ...

    # === 新增字段 ===
    norm_type: str = "rms_norm"
    # 可选值: "rms_norm" (默认), "layer_norm", "group_norm"

    block_structure: str = "pre_norm"
    # 可选值:
    #   "pre_norm"    — norm → attn → norm → mlp (当前默认)
    #   "post_norm"   — attn → norm → mlp → norm
    #   "parallel"    — norm → (attn ∥ mlp) → residual
    #   "deepnorm"    — DeepNorm 风格 (α·x + sublayer(norm(x)))

    position_embedding_type: str = "rotary"
    # 可选值: "rotary" (默认), "absolute", "alibi", "none"

    qkv_contiguous: bool = None
    # 显式控制 QKV 是否 contiguous，替代 sys_name 字符串匹配
    # None 时按原有逻辑自动判断

    dense_layer_positions: List[int] = None
    # 指定哪些层是 dense 层（替代 dense_layers 只能在前面）
    # None 时使用 dense_layers 的原有语义

    tie_embeddings: bool = False
    # 是否共享 embedding 和 output projection 的权重

    output_norm_type: str = None
    # 最终 LayerNorm 的类型，None 时使用与 norm_type 相同的类型

    activation_type: str = None
    # 可选值: None (按 use_swiglu 判断), "swiglu", "gelu", "relu", "geglu"
```

#### 3.3.2 Norm 类型扩展

修改 `dense_module.py` 的 `LayerNorm` 类：

```python
class LayerNorm(MetaModule):
    SUPPORTED_NORM_TYPES = ["rms_norm", "layer_norm", "group_norm"]

    def __init__(self, norm_size, norm_type="rms_norm", ...):
        assert norm_type in self.SUPPORTED_NORM_TYPES, \
            f"norm_type={norm_type} not supported, choose from {self.SUPPORTED_NORM_TYPES}"
        self.norm_type = norm_type
        ...

    def _comp_leaf_flops_info(self):
        if self.norm_type == "rms_norm":
            # 原有逻辑：5 * norm_size
            self._compute_info.fwd_flops = 5 * self.norm_size * self.batch_seq_len
        elif self.norm_type == "layer_norm":
            # LayerNorm: mean + variance + normalize = 5 * norm_size
            self._compute_info.fwd_flops = 5 * self.norm_size * self.batch_seq_len
        elif self.norm_type == "group_norm":
            self._compute_info.fwd_flops = 5 * self.norm_size * self.batch_seq_len
```

#### 3.3.3 Block 结构可配置

修改 `language_model.py` 的 `LLMBlock`：

```python
class LLMBlock(MetaModule):
    def __init__(self, ...):
        ...
        self.block_structure = getattr(config, 'block_structure', 'pre_norm')
        self.norm_type = getattr(config, 'norm_type', 'rms_norm')

        if self.block_structure == "pre_norm":
            self._build_pre_norm(config, strategy, system, ...)
        elif self.block_structure == "post_norm":
            self._build_post_norm(config, strategy, system, ...)
        elif self.block_structure == "parallel":
            self._build_parallel(config, strategy, system, ...)
        else:
            raise ValueError(f"Unknown block_structure: {self.block_structure}")

    def _build_pre_norm(self, config, strategy, system, ...):
        """当前默认结构: norm → attn → norm → mlp"""
        self.layernorm_input = LayerNorm(norm_type=self.norm_type, ...)
        self.attention = self._build_attention(config, strategy, system, ...)
        self.pre_mlp_layernorm = LayerNorm(norm_type=self.norm_type, ...)
        self.mlp = self._build_mlp(config, strategy, system, ...)

    def _build_post_norm(self, config, strategy, system, ...):
        """Post-norm: attn → norm → mlp → norm"""
        self.attention = self._build_attention(config, strategy, system, ...)
        self.post_attn_norm = LayerNorm(norm_type=self.norm_type, ...)
        self.mlp = self._build_mlp(config, strategy, system, ...)
        self.post_mlp_norm = LayerNorm(norm_type=self.norm_type, ...)

    def _build_parallel(self, config, strategy, system, ...):
        """Parallel: norm → (attn ∥ mlp) → residual"""
        self.layernorm_input = LayerNorm(norm_type=self.norm_type, ...)
        self.attention = self._build_attention(config, strategy, system, ...)
        self.mlp = self._build_mlp(config, strategy, system, ...)

    def forward(self, input_info, path_debug_context):
        if self.block_structure == "pre_norm":
            h = self.layernorm_input(input_info, path_debug_context)
            h = self.attention(h, path_debug_context)
            h = self.pre_mlp_layernorm(h, path_debug_context)
            return self.mlp(h, path_debug_context)
        elif self.block_structure == "post_norm":
            h = self.attention(input_info, path_debug_context)
            h = self.post_attn_norm(h, path_debug_context)
            h = self.mlp(h, path_debug_context)
            return self.post_mlp_norm(h, path_debug_context)
        elif self.block_structure == "parallel":
            h = self.layernorm_input(input_info, path_debug_context)
            attn_out = self.attention(h, path_debug_context)
            mlp_out = self.mlp(h, path_debug_context)
            return attn_out  # parallel 结构需要特殊合并逻辑
```

**注意**：`_build_attention` 和 `_build_mlp` 提取为独立方法，消除 `LLMBlock.__init__` 中的 if-else 嵌套。

#### 3.3.4 消除 QKV contiguous hack

修改 `dense_module.py:1154`：

```python
# 旧代码
qkv_contiguous = False if 's5000' in self.system.sys_name else True

# 新代码
qkv_contiguous = getattr(self.strategy, 'qkv_contiguous', None)
if qkv_contiguous is None:
    qkv_contiguous = getattr(self.config, 'qkv_contiguous', None)
if qkv_contiguous is None:
    qkv_contiguous = True  # 默认值
```

#### 3.3.5 Dense layer 位置灵活化

修改 `language_model.py:255`：

```python
# 旧代码
use_dense = (i < dense_layers)

# 新代码
dense_layer_positions = getattr(self.model_config, 'dense_layer_positions', None)
if dense_layer_positions is not None:
    use_dense = (i in dense_layer_positions)
else:
    use_dense = (i < dense_layers)
```

#### 3.3.6 激活函数扩展

```python
# 在 ModelConfig 中
activation_type: str = None  # None 时按 use_swiglu 判断

# 在 MLP 构建中
def _resolve_activation(config):
    act = getattr(config, 'activation_type', None)
    if act is None:
        return "swiglu" if config.use_swiglu else "gelu"
    return act
```

### 3.4 Phase B: 外部模型描述导入

#### 3.4.1 导入器接口

新增文件 `simumax/core/model_import.py`：

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ModelImporter(ABC):
    """模型配置导入器抽象基类"""

    @abstractmethod
    def import_config(self, source: str, **kwargs) -> Dict[str, Any]:
        """从外部源导入模型配置，返回兼容 ModelConfig 的字典"""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """导入源类型名称"""
        ...


class ModelImporterRegistry:
    _importers: Dict[str, type] = {}

    @classmethod
    def register(cls, source_type: str, importer_cls: type):
        cls._importers[source_type] = importer_cls

    @classmethod
    def create(cls, source_type: str) -> ModelImporter:
        return cls._importers[source_type]()

    @classmethod
    def import_config(cls, source_type: str, source: str, **kwargs) -> Dict[str, Any]:
        importer = cls.create(source_type)
        return importer.import_config(source, **kwargs)
```

#### 3.4.2 HuggingFace 导入器

```python
class HuggingFaceImporter(ModelImporter):
    """从 HuggingFace config.json 导入模型配置"""

    @property
    def source_type(self) -> str:
        return "huggingface"

    # HuggingFace 字段 → SimuMax 字段的映射
    FIELD_MAP = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "head_num",
        "num_key_value_heads": "kv_head_num",
        "head_dim": "head_size",
        "intermediate_size": "intermediate_size",
        "num_hidden_layers": "layer_num",
        "vocab_size": "vocab_size",
        "num_experts": "expert_num",
        "num_experts_per_tok": "topk",
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "hidden_act": "_activation_hint",
    }

    # 架构名 → SimuMax 配置推断
    ARCH_MAP = {
        "LlamaForCausalLM": {"attention_type": "mha", "use_swiglu": True, "norm_type": "rms_norm"},
        "MistralForCausalLM": {"attention_type": "mha", "use_swiglu": True, "norm_type": "rms_norm"},
        "Qwen2ForCausalLM": {"attention_type": "mha", "use_swiglu": True, "norm_type": "rms_norm"},
        "DeepseekV2ForCausalLM": {"attention_type": "mla", "use_swiglu": True, "norm_type": "rms_norm"},
        "DeepseekV3ForCausalLM": {"attention_type": "mla", "use_swiglu": True, "norm_type": "rms_norm"},
        "MixtralForCausalLM": {"attention_type": "mha", "use_swiglu": True, "norm_type": "rms_norm"},
        "GPT2LMHeadModel": {"attention_type": "mha", "use_swiglu": False, "norm_type": "layer_norm"},
    }

    def import_config(self, source: str, **kwargs) -> Dict[str, Any]:
        import json
        with open(source, 'r') as f:
            hf_config = json.load(f)

        result = {}
        arch = hf_config.get("architectures", [None])[0]

        # 1. 从架构名推断默认值
        if arch in self.ARCH_MAP:
            result.update(self.ARCH_MAP[arch])

        # 2. 字段映射
        for hf_key, simu_key in self.FIELD_MAP.items():
            if hf_key in hf_config:
                result[simu_key] = hf_config[hf_key]

        # 3. 特殊处理
        if "model_type" in hf_config:
            if hf_config["model_type"] in ("mixtral", "deepseek_v2", "deepseek_v3"):
                result["model_type"] = "moe"
            else:
                result.setdefault("model_type", "dense")

        # MLA 特殊字段
        if arch and "Deepseek" in arch:
            for key in ["v_head_dim", "qk_nope_head_dim", "qk_rope_head_dim",
                        "q_lora_rank", "kv_lora_rank"]:
                if key in hf_config:
                    simu_key = key
                    if key == "qk_nope_head_dim":
                        simu_key = "qk_head_dim"
                    elif key == "qk_rope_head_dim":
                        simu_key = "qk_pos_emb_head_dim"
                    result[simu_key] = hf_config[key]

        return result
```

#### 3.4.3 ModelConfig 新增加载方法

```python
class ModelConfig(Config):
    @classmethod
    def from_huggingface(cls, config_path: str, **overrides) -> 'ModelConfig':
        """从 HuggingFace config.json 创建 ModelConfig"""
        from simumax.core.model_import import ModelImporterRegistry
        config_dict = ModelImporterRegistry.import_config("huggingface", config_path)
        config_dict.update(overrides)
        return cls.init_from_dict(config_dict)

    @classmethod
    def from_external(cls, source_type: str, source: str, **overrides) -> 'ModelConfig':
        """从任意外部格式创建 ModelConfig"""
        from simumax.core.model_import import ModelImporterRegistry
        config_dict = ModelImporterRegistry.import_config(source_type, source)
        config_dict.update(overrides)
        return cls.init_from_dict(config_dict)
```

使用示例：

```python
# 从 HuggingFace 导入
model_config = ModelConfig.from_huggingface(
    "/path/to/llama-3-8b/config.json",
    model_name="my_llama3_8b",
)

# 手动覆盖某些字段
model_config = ModelConfig.from_huggingface(
    "/path/to/deepseek-v2/config.json",
    layer_num=4,  # 只模拟 4 层
)
```

### 3.5 影响范围与兼容性

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `simumax/core/config.py` | 修改 | `ModelConfig` 新增字段（全部有默认值，向后兼容） |
| `simumax/core/transformer/dense_module.py` | 修改 | `LayerNorm` 支持更多 norm 类型；消除 QKV hack |
| `simumax/core/transformer/language_model.py` | 修改 | `LLMBlock` 支持多种 block 结构；提取 `_build_*` 方法 |
| `simumax/core/model_import.py` | **新增** | 导入器接口 + HF 导入器 + Registry |
| `configs/models/*.json` | 不变 | 现有配置无需修改（新字段有默认值） |

**向后兼容保证**：

- 所有新增 `ModelConfig` 字段的默认值与当前硬编码行为一致
- `norm_type` 默认 `"rms_norm"`，`block_structure` 默认 `"pre_norm"`
- 现有 JSON 配置不包含新字段时，行为完全不变
- `use_swiglu` 继续生效，`activation_type` 只在显式设置时覆盖

### 3.6 分阶段交付计划

#### Stage 1: ModelConfig 扩展 + Norm 类型

- `ModelConfig` 新增 `norm_type`, `block_structure`, `position_embedding_type`, `qkv_contiguous`, `activation_type` 等字段
- `LayerNorm` 支持 `layer_norm` 和 `group_norm`
- 消除 `qkv_contiguous` 的 sys_name hack
- 验证：现有所有 example 结果不变

#### Stage 2: Block 结构可配置

- `LLMBlock` 支持 `pre_norm` / `post_norm` / `parallel` 三种结构
- 提取 `_build_attention` / `_build_mlp` 方法
- `dense_layer_positions` 支持任意位置 dense 层
- 验证：用 `post_norm` 构建一个模型并运行 perf

#### Stage 3: HuggingFace 导入器

- 新增 `model_import.py` + `HuggingFaceImporter`
- `ModelConfig.from_huggingface()` API
- 验证：从真实 HF config.json 导入 Llama3-8B、DeepSeek-V2，与手动 JSON 结果一致

#### Stage 4: 更多导入器 + 文档

- 支持从 Megatron-LM 参数导入
- 更新 `docs/model.md` 说明新配置字段
- 新增 example 演示 HF 导入流程

---

## 三个 Plan 的依赖关系

```
Plan 1 (CostModel)  ──→  Plan 2 (DES 过程级预测)  ──→  Plan 3 (Workload)
     独立可先行                依赖 Plan 1                  独立可并行

Plan 1 → Plan 2：DES 引擎的算子成本输入来自 CostModel 接口。
                 Plan 1 的 TableLookupCostModel 是 DES 的默认成本源。

Plan 2 → Plan 3：Plan 3 的 block 结构变更可能引入新的算子（如 parallel block
                 的合并操作），这些新算子的成本通过 Plan 1 的 CostModel 接口
                 配置，并通过 Plan 2 的 DES 引擎参与过程级调度。

Plan 1 与 Plan 3 可并行开发（无直接依赖）。

建议优先级：
  Plan 1 Stage 1 → Plan 2 Stage 1 → Plan 1 Stage 2 → Plan 2 Stage 2
  → Plan 3 Stage 1 → Plan 2 Stage 3 → Plan 3 Stage 2 → ...
```

## 验证策略

每个 Stage 完成后：

1. `bash tools/lint/pylint.sh` — lint 通过
2. `python -m compileall -q simumax app tools examples` — 编译通过
3. `PYTHONPATH=. python examples/perf_llama3_8b_tp1_pp2.py` — smoke test 结果不变
4. `cd examples && bash run_all.sh` — 所有 example 正常运行
5. 新增的 API 有对应的 example 脚本验证
