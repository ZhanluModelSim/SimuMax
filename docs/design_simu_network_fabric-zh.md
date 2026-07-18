<p align="center">
  <a href="design_simu_network_fabric.md">English</a>|
  <a href="design_simu_network_fabric-zh.md">中文版本</a>
</p>

# 设计方案：DES 网络 fabric 建模

- 状态：**草案 v0.1**（讨论已定稿，尚未实现）
- 日期：2026-07-17
- 范围：`simulate()` DES 路径——跨节点流量下集合通信/p2p 的计时。
  建立在 `design_simu_kind_resource_model.md` 的资源 lane 与
  simu_kind 地基之上。

## 1. 背景与问题

`merge_lanes=True`（默认）时只模拟 `pp_size` 个代表 rank，除
`default_group`/`send_recv-*` 外的集合通信都是 `backend_kind="local"`
（`base_struct.py:2351-2358`），经 `_pump_local_entry`
（`base_struct.py:1917-1922`）完成：

```
launch_t = max(issue_t, rank_tail(stream));  end_t = launch_t + cost
```

即只在发卡 rank 自己的 `(rank, stream)` comm lane 上串行。`cost` 是
prefill 期由 `SystemConfig.compute_net_op_time`（`config.py:973`）算出
的标量（含启发式跨节点修正）；此后 size/net/拓扑信息全部丢弃，DES
内核看不到。

跨节点通信的三个缺口：

1. **无 NIC 级资源**。DES 给每 rank 独立的 `comm`/`pp_fwd`/`pp_bwd`
   lane，同一 GPU 的 dp reduce-scatter、ep all2all、pp p2p 可以"同时"
   跑而互不干扰——真实世界它们共享该 GPU 的网卡。
2. **无跨 rank 同步**。local entry 不与组成员 rendezvous，"最慢成员
   到达"效应（skew/straggler）缺失。
3. **拓扑盲**。net 放置是 group 大小启发式
   （`analysis_high_link_net`，`perf_llm.py:419-468`），不是从
   rank↔节点映射推导；`compute_net_op_time` 的跨节点修正只覆盖
   dp/edp/p2p/all2all 调用方——落到 `inter_node` 的 TP/CP 集合通信
   没有任何修正。

## 2. 目标 / 非目标

目标：

1. 在 DES 中为所有跨节点流量建模 per-GPU NIC 竞争（规模下最主要的
   跨节点效应）。
2. 模型完全可选：默认配置结果与现状逐位一致。
3. 成本模型拓扑感知化：在目前的盲区用真实成员节点比例替代
   group-size 启发式。
4. 为节点/ToR 级竞争和跨 rank skew 预留结构，本期不实现。

非目标（本期）：

- 全 world rendezvous（`merge_lanes=False` 仍是唯一的真跨 rank 同步
  模式；skew 建模属 C 级，预留）。
- 路由级保真（rail 映射、拥塞扩散、自适应路由）。
- 改变任何默认行为。

## 3. 总体设计——三级递增

| 级别 | 内容 | 改动面 | 风险 |
|---|---|---|---|
| A | 拓扑感知静态修正 | 仅成本模型（`config.py`） | 低 |
| B | `NetworkFabric`：per-GPU NIC 服务台 + p2p 双端计费 + ToR 结构预留 | DES 内核 + Com meta + 配置 | 中 |
| C | 跨 rank skew（virtual waiters / group-representative 模拟） | DES 内核 | 预留 |

按 A → B → C 顺序独立可合入。

## 4. A 级——拓扑感知静态修正

- 新增 group→节点映射辅助函数（如 `core/utils.py`）：给定
  `group_kind`（tp/cp/dp/dp_cp/pp/ep/etp/edp）、strategy 各维度与
  `num_per_node`，解析计算（成员是等差数列：tp/ep 步长 1、cp 步长
  `tp`、dp 步长 `tp*cp`、pp 步长 `tp*cp*dp`、edp 步长 `ep`）任意集合
  通信的成员节点数与跨节点流量比例。不做全 world 枚举。
- 把 `compute_net_op_time` 的跨节点修正推广到全部 op 类型（含
  TP/CP），用辅助函数的真实跨节点比例替换 group-size 启发式。
- 现有修正公式保留以保持兼容；新比例只细化目前完全没有修正的情形。

## 5. B 级——NetworkFabric（核心）

### 5.1 资源模型

`NetworkFabric` 是由 `SimuContext` 持有的全局对象：

- **NIC 服务台**：每 GPU 一个，键为 `global_rank`。容量沿用现有约定：
  `inter_node.gbps / num_per_node`（每 GPU NIC 带宽）。NIC 属于 GPU
  而非节点，因此 per-rank NIC 服务台在 `merge_lanes` 下**无需流量
  放大**——每个被模拟的 rank 独占自己的 NIC。
- **ToR 服务台**（预留，决策 3）：每节点一个，键为
  `rank // num_per_node`。route 与 pump 结构从第一天就是多跳的；
  ToR 服务台会创建但默认不生效（直通），待节点份额放大模型落地
  （§5.5）。
- 服务台状态 = 每服务台一个尾时钟（`nic_tail[rank]`、
  `tor_tail[node]`），与 `rank_comm_tail` 同构。

### 5.2 计费对象

entry 仅当其解析后的 net 名为 `inter_node` 时计费：

- prefill：`Com` 构造新增可选 `net=`（与 `size_bytes=`）参数；模块/
  流水线调用点传入已解析的 strategy net 名（`strategy.tp_net` 等——
  在 `run_estimate` 的 `analysis_net` 中解析，早于作业构建）。默认
  `None` = 不计费 = 现状，未迁移调用点不受影响。
- issue：`net`/`size_bytes` 经 `CommEntry.meta`（现有字段）传入内核。
- 优化器的 DP reduce-scatter/all-gather（`pipeline_schedule.py`）携带
  `dp_net`/`edp_net`；world all_reduce 屏障保持名义 cost，不计费。

### 5.3 pump 与完成公式

local entry（`_pump_local_entry`，唯一改动点）：

```
launch_t = max(issue_t, rank_tail(stream), nic_tail[rank], tor_tail[node]*)
end_t    = launch_t + cost          # cost 不变，来自 compute_net_op_time
nic_tail[rank] = end_t;  （ToR 激活时 tor_tail[node] = end_t）
```

rendezvous entry（`_pump_rendezvous_entry`，`merge_lanes=False`）：
每个 waiter 的 `ready_t = max(ready_t, nic_tail[waiter])`；完成时所有
waiter 的 `nic_tail[waiter] = end_t`。

async p2p（P2PBackend，双端计费——决策 1）：每个到达的
`ready_t = max(ready_t, nic_tail[rank], tor_tail[node]*)`；
`end_t = max(ready_t + cost)` 取两端最大值（同现状）；pair 完成时
`nic_tail[send_rank] = nic_tail[recv_rank] = end_t`。收发 rank 在
`AsyncP2PState` 中已有记录。

blocking p2p（`_blocking_step_impl` + BarrierBackend，双端——决策 1）：
到达时 `ready_t = max(ready_t, nic_tail[rank])`；barrier 触发时两端
waiter 的 `nic_tail` 都设为共同的 `end_t`，与抬升 lane 时钟在同一个
drain 中处理（`SimuSystem.simu` 的 pending-completions 段）。

### 5.4 与静态修正的关系（决策 2：双层保留）

两层近似的是不同的事，有意共存：

- **静态修正**（`compute_net_op_time`）：容量恒等式与组扩散效应
  （每 GPU NIC 带宽、`(k-1)/k` 跨节点比例、dp/edp 多 NIC 分摊），
  决定 op 独占 NIC 时的*服务时长*。
- **fabric 服务台**：共享同一 NIC/ToR 的 op 之间的纯时间维度排队，
  从不修改 `cost`，只推移 `launch_t`。

在文档中明确为已知的双层近似；并注明后续可在哪里接"关闭静态修正"
的纯净 A/B 开关。

### 5.5 merge_lanes 语义与预留的 ToR 模型

- NIC 服务台是 per-GPU 的，`merge_lanes` 无需修正：每个代表 rank 的
  NIC 完全归它自己。
- ToR 服务台（预留）需要节点份额放大：`merge_lanes=True` 时每节点
  只模拟 1/`num_per_node` 个 rank，ToR 服务台只能看到节点真实流量的
  1/num_per_node。预留模型：每条 entry 记录 `meta["node_share"]`
  （merge_lanes 下 ≈ `num_per_node`，否则为 1）；ToR 激活时占用时长
  按 `cost * node_share` 计。ToR 容量默认 `inter_node.gbps`（节点上
  行），可在 `topology` 中覆盖。
- 跨 stage 的 PP p2p 两端经现有代表 rank 映射；`default_group` 保持
  纯屏障。

### 5.6 trace 呈现

local comm span 的起点是 `launch_t`（`_event_start_t`），NIC 竞争表
现为 span 右移（issue→launch 的间隙）。v1 不新增事件类型；后续可
选加 `nic_wait` 标记。

## 6. 配置

system.json（全部可选；缺省 = 现状）：

```json
{
  "fabric_model": "nic",
  "topology": {
    "tor_capacity_gbps": 1600,
    "tor_node_share": "auto"
  }
}
```

- `fabric_model`：`"nic"` 启用 NIC 服务台；`"nic+tor"` 额外激活
  ToR 服务台（Preview）；缺省 = 关闭。
- `topology.tor_capacity_gbps`：ToR 容量，默认 `inter_node.gbps`。
- `topology.tor_node_share`：`"auto"`（merge_lanes 下 =
  `num_per_node`）或显式数值。

## 7. 验证

- **回归**：`fabric_model` 缺省时，三个 golden 用例（llama
  merge_lanes、8-rank、deepseekv2 ep4_pp2）逐事件等价。
- **合成串行化**：同 rank 两条并发 `inter_node` op →
  end_t == t0 + cost1 + cost2；两端 NIC 都忙的 blocking p2p →
  end_t 反映两端 tail。
- **E2E A/B**：16384 卡 moe-8T 用例在 `fabric_model` 关 / `"nic"`
  下各跑一次，报告 end_t delta 与 span 位移（预期：end_t 只增；
  dp/ep/pp 重叠区出现串行化）。
- **A 级**：`net_info.json` 仅在原先无修正处（TP/CP inter_node）
  有差异，其余不变。

## 8. 分阶段实施

- **Phase A**（成本模型）：`core/utils.py` 的 group→节点辅助函数 +
  `compute_net_op_time` 修正推广 + A 级验证。
- **Phase B**（内核）：`base_struct.py` 的 `NetworkFabric`、`Com(net=)`
  全调用点接线、local + rendezvous + async p2p + blocking p2p（双端）
  的 pump/完成钩子、配置字段、ToR 结构（默认直通）、完整验证。
- **Phase C**（预留）：基于 `enable_straggler_model` /
  `get_effective_straggler_sample_count` 的 virtual waiters 跨 rank
  skew。

## 9. 影响面

- `simumax/core/base_struct.py`：`NetworkFabric`、pump/完成钩子、
  `Com.__init__` 参数、blocking/async p2p 的 ready_t 计算。
- `simumax/core/config.py`：`fabric_model`/`topology` 字段与校验；
  `compute_net_op_time` 的 A 级修正。
- `simumax/core/utils.py`：group→节点映射辅助函数。
- `simumax/core/simu_runner.py`：由 system/strategy 构建
  `NetworkFabric` 并挂到 `SimuContext`。
- `dense_module.py` / `moe_module.py` / `pipeline_schedule.py`：各
  `Com` 创建点传 `net=`（增量、带默认值）。
- `docs/system.md`（+zh）：新增 `fabric_model`/`topology` 字段说明。

## 10. 验收标准

- 默认关闭时全部 golden 用例逐位等价。
- 合成 NIC 串行化用例的 end_t 与解析值一致。
- 交付 16384 卡 E2E A/B 报告（end_t delta + 竞争热点）。
- 除 A 级修正外不改变任何 `cost` 公式。

## 11. 决策记录

1. **blocking p2p 计双端**：收发两端 NIC 都被占用并在完成时同时更新
   （§5.3）。
2. **保留双层近似**：静态修正与 fabric 服务台共存，职责已在文档中
   划分（§5.4）。
3. **ToR 级现在预留**：多跳 route、ToR 服务台、`node_share` 放大字段
   进入数据结构；竞争模型本身后续以 Preview 落地（§5.1、§5.5）。
