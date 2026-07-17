<p align="center">
  <a href="design_simu_kind_resource_model.md">English</a>|
  <a href="design_simu_kind_resource_model-zh.md">中文版本</a>
</p>

# 设计方案：显式 `simu_kind` 与资源/引擎 lane 模型

- 状态：**草案 v0.1**（讨论已定稿，尚未实现）
- 日期：2026-07-17
- 范围：`simumax/core` 的 DES 路径（`PerfLLM.simulate()`）、trace 导出、配置。
  静态解析路径（`analysis_*`）明确不在本期范围内。

## 1. 背景与问题

当前模拟器的算子分类发生在两个脱节的位置：

1. **行为分类是隐式的**。算子是计算还是通信，由实例化哪个类决定
   （`simumax/core/base_struct.py` 中的 `AtomModel` vs `Com`），没有任何
   显式声明；DES 内核只是多态地调用 `step()/bwd()`。
2. **展示分类靠字符串猜测**。trace 导出器
   （`simumax/core/generate_tracing.py:192-206`）按 `call_stk` 最后一段
   与硬编码的 `COMM_PREFIXES`（`generate_tracing.py:6-24`）匹配来分配
   cat/tid/lane。一个行为上正确的新通信算子，只要类名不命中前缀表，就
   会被静默错分成 compute。
3. **资源模型是三条硬编码 lane**。每 rank `t["comp"]/t["comm"]/t["off"]`
   （`base_struct.py:1347`），`off` 从未使用。一个 rank 同一时刻只能有
   一个计算在飞，无法表达 Cube∥Vector 引擎并行和通算融合 kernel。阻塞
   通信完成时无条件 clamp `t["comp"]`（`base_struct.py:2215-2216`），把
   "rank 内计算与集合通信互斥"写死；`overlap_grad_reduce`
   （`config.py:275`）因此成为死配置。
4. **日志是私有文本格式**。7 处写点每行一次 open/write/close
   （`base_struct.py:85,177,1988,2032,2050,2149,2167`）；不匹配解析正则
   （`generate_tracing.py:27-53`）的行被静默丢弃。

## 2. 目标 / 非目标

目标：

1. `simu_kind` 显式声明，分类的单一事实来源收敛到算子定义处。
2. 资源 lane 泛化：硬件引擎（Cube/Vector）与通信链路成为可注册资源，
   可表达多资源并行。
3. `fused` 作为可配置扩展的 kind 落地，预留通算融合与 DualPipeV 式
   F/B 交织所需的一切。
4. 结构化事件流取代文本日志。
5. 行为兼容：现有配置（单引擎 GPU 机器）产出逐事件等价的 trace。

非目标（本期不做）：

- DualPipeV 调度器本身的实现（只交付地基与 builder 插槽）。
- Vector 引擎的真机效率测量流程（`simu_tools/efficency_test` 扩展，
  另立需求）。
- 静态解析路径（`perf_llm.py` 的 `analysis_*`）的同步改造。fused 策略
  下它可能与 `simulate()` 结果分歧，这是已接受的取舍。

## 3. fused 到底 fuse 什么（分层定义）

"融合"必须分层，否则 kind 会变成什么都装的筐：

| 层次 | fuse 的对象 | 例子 | 分类归属 |
|---|---|---|---|
| F1 kernel 融合 | 计算 × 计算（同引擎） | fused swiglu、fused CE | **不是** fused，只是成本数值不同的 compute |
| F2 通算融合 | 计算 × 通信（跨资源） | AG+GEMM、RS+GEMM 分 chunk 流水 | `simu_kind="fused"`，占用 `(cube, comm)` |
| F3 引擎并行 | 计算 × 计算（跨引擎） | Cube 跑 GEMM 的同时 Vector 跑 norm/permute；DualPipeV 中 F(batch i) 的 Cube 段与 B(batch j) 的 Vector 段并行 | 不需要特殊 kind，由资源 lane 分离自然涌现 |

**定义：`fused` = 一个调度单元同时占用 ≥2 类硬件资源**。F3 不靠 kind
解决，靠 4.2 的资源模型；DualPipeV 的 F/B 交织是调度器产物，其中的
Cube∥Vector 并行是 F3，通算 chunk 是 F2。

## 4. 设计

### 4.1 `simu_kind` 显式声明

```python
class LeafModel:
    simu_kind: ClassVar[str] = "compute"         # compute | comm | wait | scope | fused
    simu_resources: ClassVar[tuple] = ("comp",)  # 占用的资源 lane
```

- `AtomModel` → `compute / ("comp",)`；`Com` 各子类 → `comm /
  ("comm",)`；async p2p post → `comm / ("pp_fwd",)` 或 `("pp_bwd",)`；
  `async_wait_recv` → `wait`。
- 导出器改为读事件对象上的 kind/resources，删除三处硬编码
  （`COMM_PREFIXES`、`_comm_lane`、scope 前缀猜测）。
- `call_stk` 保留，仅用于人类可读命名与 microbatch/chunk 提取
  （`simu_memory` 依赖），不再承担分类职能。

### 4.2 资源/引擎 lane 模型（本方案的地基）

- `SimuThread.t`（`base_struct.py:1347`）从硬编码 `{comp, comm, off}`
  改为按**资源注册表**初始化的 lane 字典；注册表来自 system 配置的
  `engines` 声明 + 内置通信 lane。
- 算子完成时**只推进自己声明的 lane**；`Com._step/_bwd` 中无条件 clamp
  `t["comp"]` 的特例删除。
- 阻塞语义统一为 **post + wait**：现有 async p2p 的 post/wait 机制
  （`base_struct.py:2419-2617`）推广为唯一的通信语义；
  `blocking = post + 紧邻的 wait`。`overlap_grad_reduce` 从此有了表达
  途径（post 后不立刻 wait）。
- 跨资源依赖（Vector op 消费 Cube op 产出、同一 microbatch 的 F→B 依赖）
  用 notify/wait token 对表达，复用 `BarrierBackend` 基础设施。
- 调度时刻语义（`cur_time` = 活跃 lane 取 min）不变，只是 lane 集合从
  3 条变 N 条。
- **默认配置（GPU、单引擎）下资源集 = `{comp, comm, pp_fwd, pp_bwd}`，
  与现状一一对应，行为不变。**

### 4.3 `fused` kind 与可插拔 fusion policy

```python
class FusedOp(LeafModel):
    simu_kind = "fused"
    simu_resources = ("cube", "comm")
    fusion_policy = ChunkedPipeline(chunks=4)
    # duration = max(Σcube_chunk, Σcomm_chunk) + 流水首尾气泡
    # 完成时两条 lane 分别推进
```

- `fusion_policy` 是可插拔对象；内置 `Serial`（等价现状）、`MaxOverlap`
  （duration=max）、`ChunkedPipeline(chunks=n)`；新策略注册即可用，不改
  内核。
- 成本模型：`SystemConfig` 新增 `compute_fused_op_cost(op_desc, policy)`
  分派入口；F2 类算子的效率条目挂 system.json（预留字段，无实测数据
  时用 policy 的解析上界公式兜底）。
- strategy 侧开关，如 `"fused_ops": [{"pattern": "tp_ag_gemm",
  "policy": "chunked_pipeline", "chunks": 4}]`；未配置则一切按现状串行
  建模。

### 4.4 DualPipeV 预留（只交付地基）

1. **资源层**：cube/vector lane 分离（4.2），F(batch i) 的 GEMM 与
   B(batch j) 的 vector 段天然并行。
2. **调度层**：`PpSchedule` 的构建逻辑抽象出 `ScheduleBuilder` 接口
   （现有 1F1B / interleaved 收编为两个 builder），`DualPipeVBuilder`
   后续注册进同一插槽。job 仍是顺序队列，交织在构建期展开，
   **DES 内核零改动**。
3. **依赖层**：同一 microbatch 跨 chunk、跨引擎的 F→B 依赖用 4.2 的
   notify/wait token 表达（接口预留，DualPipeV 实施时填充）。

注：DualPipeV 下两个 microbatch 激活共存不需要新机制——显存 cache
token 已经通过 `cache_token_scope` 按 microbatch 分池。

### 4.5 结构化事件流

- 7 处文本写点改为向 `ctx.event_sink` 追加**事件对象**（dataclass：
  rank、kind、resources、name、call_stk、ts、dur、gid、post_ts……），
  默认内存列表。
- `process_log_file` 直接消费事件流，正则解析删除——"格式不匹配静默
  丢事件"的失败模式整个消失，顺带消掉每行 fopen/fclose。
- `log.log` 文本可保留为 debug 产物（由事件流单向格式化），不再是
  中间交换格式。

### 4.6 trace 导出改造

- cat/tid/lane 全部来自事件属性。
- fused 事件渲染为**多条 lane 上的关联 slice**（每个占用资源一条，
  共享 correlation id）——这是通算融合在 Perfetto 里的正确形态。
- 顺手把 scope 检测的 O(k²) 扫描（`generate_tracing.py:320-357`）重写为
  栈式一次扫描。

### 4.7 fused 算子的显存记账（已定案，见第 9 节）

现有 tracker 只在 `FwdQue/BwdStk` 边界发事件（`simu_memory.py:175-187`）：
`phase_start` 跳到峰值，`phase_end` 记 alloc/free。chunk 流水式 fused
op 的显存是斜坡式爬升/回落（chunk 边到达边消费），边界事件表达不了。

定案模型：

- **默认（稳态闭式）**：op start 一次性记账
  `peak = 输入激活 + 输出 + 2×chunk 暂存（双缓冲）`。tracker 零改动，
  且保留融合相对 unfused 省 `权重全量 - 2×chunk` 的显存收益。
- **预留开关（忠实 ramp）**：config 开关打开后按 chunk 到达/消费逐笔
  发事件，得到真实斜坡曲线。实现延后，事件 schema 里预留接口。

## 5. 配置示例

system.json（Ascend 风格引擎声明；缺省 = 单引擎，行为同现状）：

```json
{ "engines": { "cube": { "peak_tflops": 320 }, "vector": { "peak_tflops": 80 } } }
```

strategy.json（全部可选，缺省即现状）：

```json
{
  "compute_engine_map": { "gemm": "cube", "elementwise": "vector" },
  "fused_ops": [{ "pattern": "tp_ag_gemm", "policy": "chunked_pipeline", "chunks": 4 }],
  "fused_mem_mode": "steady_state"
}
```

## 6. 分阶段实施

- **Phase 0 结构化事件流**：7 处写点 + event sink + 导出器消费侧；
  trace 逐事件 diff 验证。
- **Phase 1 kind 声明与注册表**：`LeafModel` 类属性、12 个 `Com` 子类
  标注、导出器切换、`COMM_PREFIXES` 删除；trace 逐事件 diff 验证。
- **Phase 2 资源 lane 泛化 + post/wait 统一**：lane 字典化、阻塞语义
  改造、`Com` clamp 删除；默认单引擎回归 + async PP 用例复验。
  **此阶段 trace 形态有意变化**（忠实的 post/wait 双事件，见 9.3），
  产出成为新基线。
- **Phase 3 fused 扩展点**：`FusedOp`、fusion policy 注册表、成本模型
  分派入口、`ScheduleBuilder` 接口化（DualPipeV 插槽）；用人工构造的
  AG+GEMM 用例验证多资源 lane 并行推进。

每阶段独立可合入。Phase 0/1 是纯重构；Phase 2 是语义风险集中点
（clamp 删除影响现有 overlap 行为），golden 回归重点覆盖。

## 7. 影响面

- `simumax/core/base_struct.py`：类属性、lane 字典化、post/wait 统一、
  写点改造（核心改动区）。
- `simumax/core/generate_tracing.py`：前缀分类删除、事件流消费、scope
  检测重写、fused 多 lane 渲染。
- `simumax/core/simu_runner.py`：资源注册表初始化、sink 接线。
- `simumax/core/config.py`：system `engines`、strategy
  `compute_engine_map` / `fused_ops` / `fused_mem_mode` 字段与校验。
- `simumax/core/transformer/pipeline_schedule.py`：builder 接口化
  （Phase 3）。
- `docs/`：配置字段落地时同步 `strategy.md` / `system.md` 及 `-zh` 镜像。

## 8. 验收标准

- Phase 0/1：`examples/simulator_trace_snapshot.py` 与
  `examples/perf_deepseekv2_layer4_ep4_pp2.py` 的 trace 与改造前逐事件
  等价（event id 允许重排）。
- Phase 2：默认配置 golden 回归全绿；`--no-merge-lanes` 8 rank 用例
  时间线不回归。
- Phase 3：构造用例中 fused op 在 cube/comm 两条 lane 上各出现关联
  slice，且 `end_t` 与 policy 解析值一致。

## 9. 决策记录（开放问题定案）

1. **Vector 引擎成本来源**：先用 peak 折算的解析估算占位（system.json
   的 `engines.vector` 只带标量峰值，无实测效率表）；后续可加测量流程，
   接口不变。
2. **fused 算子显存记账**：默认 4.7 的稳态闭式记账；忠实 ramp 模式由
   `fused_mem_mode` 配置开关预留。
3. **post/wait trace 形态**：接受忠实的双事件形态（post 事件 + wait
   事件）。Phase 2 的 trace 与旧形态有意不同；golden 等价只覆盖
   Phase 0/1。
