# DES Plan 3: VPP 交错调度 + 异步 P2P + PP 通信与计算重叠

## 目录

- [1. 现状分析](#1-现状分析)
- [2. 总体架构](#2-总体架构)
- [3. Task 1: VPP 交错调度](#3-task-1-vpp-交错调度)
- [4. Task 2: 异步 P2P 通信](#4-task-2-异步-p2p-通信)
- [5. Task 3: PP 通信与 Compute 重叠](#5-task-3-pp-通信与-compute-重叠)
- [6. 集成与测试](#6-集成与测试)
- [7. 里程碑](#7-里程碑)

---

## 1. 现状分析

### 1.1 当前 DES 支持的 PP 策略

| 策略 | 支持状态 | 路径 |
|------|----------|------|
| 单 MB 串行 (PP=1) | 已支持 | `from_module_costs` else 分支 |
| 单 chunk 1F1B (PP≥2, mbc≥1) | 已支持 | `from_module_costs` Path A (1F1B) |
| VPP 交错调度 | **未支持** | 无 |
| Async P2P (`pp_comm_async=True`) | **未支持** | 无 |
| PP 通信与 compute 重叠 | **未支持** | 无（P2P 在 `_schedule_inter_comm_at` 中与 compute 串行） |

### 1.2 已有基础设施

**旧 DES 引擎 (SimuSystem) 已有 VPP + Async P2P 的完整实现**，新 DES 引擎可以直接复用其概念和接口：

| 组件 | 位置 | 说明 |
|------|------|------|
| `PpSchedule._prefill_batch_interleaved()` | `pipeline_schedule.py:97` | VPP 调度器，支持阻塞/异步两种模式 |
| `_compute_interleaved_sync_schedule()` | `perf_llm.py:2387` | VPP perf timing 的 heap 调度器 |
| `async_send` / `async_recv` / `async_wait_recv` | `base_struct.py:2449-2647` | 异步 P2P 通信基类 |
| `SimuContext` 异步基础设施 | `base_struct.py:1570` | `AsyncP2PState`, `post_async_send_entry`, `ensure_async_ready` 等 |
| `_Op` 带 `bundle_ops` | `perf_llm.py:2439` | VPP 批量通信操作的中间表示 |
| `_compute_single_batch_phase_inputs()` | `perf_llm.py:2709` | 每个 stage 的 fwd/bwd 时间分解（含 P2P send/recv） |
| `vpp_chunk_dict` + `vpp_stage_chunk_names` | `perf_llm.py:852` | VPP 虚拟 chunk 的构建 |

### 1.3 新 DES 引擎当前的能力

`from_module_costs` 当前：
- 只调度物理 chunk（`model_chunk_dict`），不感知虚拟 chunk
- P2P 通信通过 `_schedule_inter_comm_at` 按绝对时间注入，与 compute 串行
- 无 `async_send`/`async_wait_recv` 概念
- 调度粒度是 `(rank, mb, "F"/"B")`（1F1B 偏移 → `_schedule_leaves_pass`）

---

## 2. 总体架构

```
新的 from_module_costs 流程:

PerfLLM
├── model_chunk_dict (物理 chunk)
│   ├── first_stage_chunk
│   └── last_stage_chunk
├── vpp_chunk_dict (虚拟 chunk, 仅在 vp > 1 时构建)
│   ├── first_stage_chunk_v0, first_stage_chunk_v1
│   └── last_stage_chunk_v0, last_stage_chunk_v1
└── vpp_stage_chunk_names (映射)

DesBridge.from_module_costs VPP 分支:
│
├─ 1. 构建虚拟 stage 列表 (vp_size × pp_size 个 stage)
│
├─ 2. 计算每个虚拟 stage 的 fwd/bwd/recv/send 时间
│     (复用 _compute_single_batch_phase_inputs 或直接从 chunk 计算)
│
├─ 3. 生成 1F1B 调度表 (复用 _compute_interleaved_sync_schedule
│     或直接调用 PpSchedule._prefill_batch_interleaved)
│
├─ 4. 按时间排序所有 (rank, mb, virtual_chunk, kind, comm_ops) 事件
│
├─ 5. 逐个调度:
│     - compute: _schedule_leaves_pass(des, rank, model, pass_dir, mb)
│     - async P2P: schedule_async_send / schedule_async_recv
│     - async wait: schedule_async_wait (在需要数据的 compute 前)
│
└─ 6. 导出 tracing
```

---

## 3. Task 1: VPP 交错调度

### 3.1 设计目标

支持 `interleaving_size > 1` 的 VPP 场景，正确调度多个虚拟 stage 在同一个物理 PP stage 上的交错执行。

### 3.2 核心逻辑

**3.2.1 虚拟 stage 列表构建**

```python
def _build_vpp_stages(perf_model):
    """构建虚拟 stage 列表: [(stage_idx, virtual_idx, model)]"""
    stages = []
    vp_size = perf_model._vp_size()
    if vp_size <= 1:
        return []  # 无 VPP

    pp_size = perf_model.strategy.pp_size
    for pp_rank in range(pp_size):
        # 确定该 pp_rank 对应的物理 stage key
        if pp_rank == 0:
            stage_key = "first_stage_chunk"
        elif pp_rank == pp_size - 1:
            stage_key = "last_stage_chunk"
        else:
            stage_key = "middle_stage_chunk"

        for v in range(vp_size):
            chunk_name = f"{stage_key}_v{v}"
            model = perf_model.vpp_chunk_dict.get(chunk_name)
            if model:
                stages.append((pp_rank * vp_size + v, pp_rank, v, model))
    return stages
```

**3.2.2 1F1B 调度表生成**

复用已有的两种方式：

- **方式 A (推荐)**：调用 `PpSchedule._prefill_batch_interleaved()` 生成 job 列表，解析为 `(start_time, rank, virtual_stage, mb, kind, comm_ops)` 序列
- **方式 B**：调用 `_compute_interleaved_sync_schedule()` 获取已有的 VPP perf timing 结果

方式 A 更精确（因为它实际运行了 `PpSchedule` 的 job 生成逻辑），方式 B 更快（纯解析计算）。

**3.2.3 调度到 DES**

```python
for start_ms, gpu_rank, model, mb, kind, comm_bundle in ordered_ops:
    _advance_all_lanes_to(des, gpu_rank, start_ms)
    pass_dir = "fwd" if kind == "F" else "bwd"
    _schedule_leaves_pass(des, gpu_rank, model, pass_dir, mb=mb)

    if comm_bundle:
        for comm_op in comm_bundle:  # send/recv pairs
            if comm_op.is_async:
                _schedule_async_p2p(des, ...)
            else:
                _schedule_sync_p2p(des, ...)
```

### 3.3 新增/修改的文件

| 文件 | 变更 |
|------|------|
| `des_bridge.py` | 新增 `_build_vpp_stages()`, `from_module_costs` 增加 VPP 分支 |
| `des_engine.py` | 新增 `schedule_async_send()`, `schedule_async_recv()`, `schedule_async_wait()` 方法 |

### 3.4 验证

- 构造 TP=1, PP=4, VP=2 场景
- 检查每个物理 rank 上的多虚拟 stage 调度顺序
- 与旧的 `PpSchedule` 生成的调度表对比时间一致性

---

## 4. Task 2: 异步 P2P 通信

### 4.1 设计目标

支持 `pp_comm_async=True` 的异步 P2P 通信模式，以 `ResourceEvent` 建模 send/recv/wait 三元组。

### 4.2 核心数据结构

**新增事件类型**：不再只用 `ph:"X"` 的 duration 事件，增加以下语义：

```python
@dataclass
class AsyncP2PGroup:
    """一组异步 P2P 操作 (batch_isend_irecv)"""
    gid: str               # 全局唯一 ID "(phase, idx, src, dst)"
    send_rank: int
    recv_rank: int
    send_start: float      # ms
    send_dur: float        # ms
    recv_start: float      # ms
    recv_dur: float        # ms
    wait_start: float      # ms (async_wait 开始时间)
    wait_end: float        # ms (数据到达时间)
```

**DES 引擎新增方法**：

```python
class MultiResourceDES:
    def schedule_async_send(self, rank, gid, start_ms, dur_ms, ...):
        """在 INTER_LINK 上调度异步发送。不阻塞 COMPUTE。"""
        ...

    def schedule_async_recv(self, rank, gid, start_ms, dur_ms, ...):
        """在 INTER_LINK 上调度异步接收。不阻塞 COMPUTE。"""
        ...

    def schedule_async_wait(self, rank, gid, target_ms, ...):
        """COMPUTE 在此阻塞直到 gid 对应的 P2P 完成。
           实际上是把 COMPUTE queue advance_to(target_ms)。"""
        comp_q = self.get_queue(rank, ResourceType.COMPUTE)
        comp_q.advance_to(target_ms)
```

### 4.3 调度流程（单个 MB 内的 P2P）

```
F0(rank0) 完成后:
  → async_send(rank0, gid_fwd_mb0, rank1)     ← 立即发出，不阻塞
  → async_recv(rank1, gid_fwd_mb0, rank0)     ← 立即发出，不阻塞

F0(rank1) 开始前:
  → async_wait(rank1, gid_fwd_mb0)            ← 如果 P2P 未完成则阻塞 COMPUTE
```

### 4.4 新增/修改的文件

| 文件 | 变更 |
|------|------|
| `des_engine.py` | `AsyncP2PGroup` dataclass + `schedule_async_*` 方法 |
| `des_bridge.py` | `from_module_costs` 增加 async P2P 分支 |

### 4.5 验证

- 对比 async vs sync 模式的迭代时间（async 应该更短）
- 验证 tracing 中 send/recv/wait 事件的位置
- TP=1, PP=2, mbc=4, async=True：send 和 recv 应该与 compute 并行

---

## 5. Task 3: PP 通信与 Compute 重叠

### 5.1 设计目标

在单个 MB 内允许 PP P2P 通信与 compute 并行执行。这是通过 `ResourceQueue` 的独立 lane 天然支持的——只需要解除 `_schedule_leaves_pass` 中 `compute → comm` 的强制串行约束。

### 5.2 重叠时机

| 场景 | 重叠方式 |
|------|----------|
| Sync P2P + compute | **无重叠**（sync send/recv 阻塞 COMPUTE） |
| Async P2P + compute | **有重叠**（async send 发出后 COMPUTE 继续，async_wait 时才阻塞） |
| 跨 MB 的 1F1B | F(N) 和 B(N-1) 在不同 stage 上重叠（已支持） |

### 5.3 实现方式

当前的 `_schedule_leaves_pass` 中，sync P2P 和 compute 在同一个 `for mb` 循环内串行调度：

```python
# 当前: fwd → p2p → bwd (全串行)
_schedule_leaves_pass(des, rank, model, "fwd")
_schedule_inter_comm_at(...)  # P2P 插入
_schedule_leaves_pass(des, rank, model, "bwd")
_schedule_inter_comm_at(...)  # P2P 插入
```

改为 async 模式后：

```python
# Async: fwd → async_send → bwd (P2P 与后续 compute 并行)
_schedule_leaves_pass(des, rank, model, "fwd")
_des.schedule_async_send(rank, gid, ...)   # 发出但不阻塞
_async_wait_if_needed(des, rank, gid, ...) # 只在必要时阻塞
_schedule_leaves_pass(des, rank, model, "bwd")  # 可以与 async_send 并行
```

### 5.4 新增/修改的文件

| 文件 | 变更 |
|------|------|
| `des_bridge.py` | `from_module_costs` 增加 async/overlap 分支 |
| `des_engine.py` | 已有 `INTER_LINK` lane 天然支持并行，无需改 |

### 5.5 验证

- Async PP2: tracing 中 P2P send/recv 与下一个 MB 的 compute 在时间线上重叠
- 对比 async vs sync 的 total_iteration_time：async 应该更短
- Overlap 报告应该显示部分 P2P 通信时间被掩盖

---

## 6. 集成与测试

### 6.1 `from_module_costs` 的最终控制流

```python
def from_module_costs(perf_model, num_ranks):
    vp_size   = perf_model._vp_size()
    pp_size   = perf_model.strategy.pp_size
    is_async  = perf_model.strategy.pp_comm_async
    mbc       = perf_model.strategy.micro_batch_num

    if vp_size > 1:
        # VPP 路径
        return _build_vpp_des(perf_model, num_ranks, is_async, mbc)
    elif pp_size > 1:
        if is_async:
            return _build_async_pp_des(perf_model, num_ranks, mbc)
        else:
            return _build_sync_pp_des(perf_model, num_ranks, mbc)  # 当前实现
    else:
        return _build_tp_only_des(perf_model, num_ranks, mbc)
```

### 6.2 测试场景

| 场景 | 配置 | 关键验证点 |
|------|------|------------|
| VPP sync | TP=1, PP=4, VP=2, mbc=8, async=False | 虚拟 stage 调度顺序，chunk 分配 |
| VPP async | TP=1, PP=4, VP=2, mbc=8, async=True | async send/recv 与 compute 重叠 |
| PP async | TP=2, PP=2, mbc=4, async=True | P2P 与 compute 并行，tracing 时间线 |
| PP sync | TP=2, PP=2, mbc=4, async=False | 与现有行为向后兼容 |

### 6.3 Chrome Tracing 可视化

- async P2P 事件在 `comm` 泳道上显示为 `async_send_fwd`, `async_recv_fwd`, `async_wait_fwd` 等
- 每个 wait 事件通过 `args.gid` 与对应的 send/recv 关联
- `cat` 字段区分：`"comm"` → `"comm_async"` 或 `"comm_wait"` 以示区分

---

## 7. 里程碑

| Stage | 内容 | 预计文件变更 | 验证方式 |
|-------|------|-------------|----------|
| M1: VPP 调度 | `_build_vpp_stages` + VPP 1F1B 调度表生成 + 调度到 DES | `des_bridge.py` +200 行 | TP1 PP4 VP2 的 tracing 与旧 PpSchedule 对比 |
| M2: Async P2P | `schedule_async_*` 方法 + tracing 导出适配 | `des_engine.py` +80 行, `des_bridge.py` +60 行 | async PP2 的 send/recv/wait 事件时序 |
| M3: PP Overlap | 解除 sync P2P 的 compute→comm 串行约束 | `des_bridge.py` +40 行 | async vs sync 迭代时间对比、overlap 报告 |

**依赖关系**：

```
M1 (VPP 调度) ──→ M2 (Async P2P) ──→ M3 (PP Overlap)
    独立可先行              依赖 M1                     依赖 M2
```

M1 可先行交付（纯 sync VPP），M2 在 M1 基础上增加 async 语义，M3 在 M2 基础上解锁重叠。

---

## 参考

- `simumax/core/transformer/pipeline_schedule.py` - PpSchedule VPP 实现
- `simumax/core/perf_llm.py` - `_compute_interleaved_sync_schedule` / `_Op` 类
- `simumax/core/base_struct.py` - `async_send` / `async_recv` / `async_wait_recv` / `SimuContext`
- `simumax/core/config.py` - `interleaving_size`, `pp_comm_async`, `microbatch_group_size_per_vp_stage`
