<p align="center">
  <a href="design_simu_hierarchical_network.md">English</a>|
  <a href="design_simu_hierarchical_network-zh.md">中文版本</a>
</p>

# 设计方案：分层网络拓扑

- 状态：**草案 v0.1**（讨论已定稿，尚未实现）
- 日期：2026-07-17
- 范围：网络拓扑声明、通信域→层级映射、分层成本合成、fabric 层级
  服务台、placement 配置。建立在 `design_simu_network_fabric.md`
  （Phase A–C）与 `core/utils.py` 的 group→节点解析之上。

## 1. 背景与问题

当前网络模型是平铺的两级模型：

1. system.json 的 `networks` 是平铺链路列表（`high_intra_node`、
   `inter_node`、pcie 变体），链路之间没有层级关系。
2. net 选择是二值的（`analysis_high_link_net`，
   `perf_llm.py:419-446`）：每个通信域按 `group跨度 ≤ num_per_node`
   解析为节点内/节点间。
3. `num_per_node` 是唯一拓扑参数；全库无机柜/pod 概念。
4. 一次通信只挂一个 net：`compute_net_op_time` 用单一链路的拟合
   参数算 cost，没有"使用若干层物理链路"的概念。

生产集群需要（本方案补齐）：N 卡经 A 方式互联成节点、M 节点经 B
方式互联成 pod、P pod 经 C 方式互联成机柜——每层各有带宽与时延，
通信算子按通信域排布，按真实跨越的层级与比例计费。

## 2. 目标 / 非目标

目标（对应第 12 节决策）：

1. system.json 声明多层级拓扑（node/pod/rack/…），每层独立链路
   配置。
2. 通信域→层级映射带**比例分解**：32 人组可分解为
   `[2 节点内] × [8 pod 内] × [2 rack 内]`，比例驱动每层流量份额。
3. 分层成本合成按**通信类型区分策略**：all2all 取最慢层（max），
   层次化集合通信（all_reduce/all_gather/reduce_scatter）逐层串行
   求和（serial），均可配置覆盖。
4. net 字段语义 C（显式指定回退旧路径）；placement 是一等配置。
5. fabric pod/rack 服务台**直接激活计费**（非纯结构预留）。
6. 完全向后兼容：无 `topology.levels` ⇒ 结果逐位一致。

非目标：

- 路由级保真（rail 映射、自适应路由、拥塞扩散）。
- MoE mesh 的 placement 变体（dense mesh tp/cp/dp/pp 可配置；
  MoE mesh ep/etp/edp 维持现有固定顺序，见 5.4 的说明——v1 范围）。

## 3. 拓扑声明（system.json）

```json
"topology": {
  "levels": [
    {"name": "node", "size": 8,   "net": "high_intra_node"},
    {"name": "pod",  "size": 32,  "net": "inter_node"},
    {"name": "rack", "size": 256, "net": "inter_rack"}
  ],
  "composition_policy": {"all2all": "max", "collectives": "serial"}
}
```

- `levels` 按由内向外排序。`size` = 该层包含多少个下一层单元
  （node.size=8 ⇒ 8 卡/节点；pod.size=32 ⇒ 32 节点/pod = 256 卡；
  rack.size=256 ⇒ 256 pod/rack）。第一层的"单元"是单卡。
- `net` 指向现有 `networks` 字典条目——该 dict 零结构改动，新增
  `inter_rack` 等条目只是加数据。每层的带宽/时延/拟合 op 参数取
  自该 net 条目。
- 第一层必须是 size 等于 `num_per_node` 的节点层（校验；保证所有
  现存 `num_per_node` 修正口径一致）。
- `composition_policy`：第 6 节的按类型合成策略，默认值如图
  （`all2all` → max，ring/tree 集合通信 → serial，`p2p` →
  serial），单项可覆盖。

## 4. Placement 配置（strategy）

placement 决定并行维度如何排布到物理层级上——它是每个通信域
每层比例构成的来源。

- 复活 `order_of_paralielism` 字段（当前仅文档、校验只放行一个值）
  作为 placement 字段：默认 `"tp-cp-ep-dp-pp"` = 现有硬编码 mesh
  （最内层优先）。校验扩展为文档化的 dense 维排列。
- Phase A 的 stride 表改由 placement 推导：默认顺序下 stride 保持
  tp=1、cp=tp、dp=tp·cp、pp=tp·cp·dp（与现状逐位一致）；其他
  声明排列按序重算。
- 显式 rank 重排（用户自定义映射）不在 v1 范围；v1 只覆盖顺序排列。

## 5. 通信域→层级映射（T2）

`group_level_span(group_kind, strategy, levels) -> list[LevelSpan]`
（`core/utils.py`，`group_node_stats` 的推广）：

1. 组成员是等差数列 `base + k*stride`（stride 来自 placement）。
2. 逐层走查：以累计跨度 `S_L`（到该层为止的 size 连乘）计算单个
   L 单元内的成员数与该组触及的 L 单元数——产出**比例构成**
   `[k_1, k_2, …]`，如 32 人组的 `[2, 8, 2]`（每节点 2 人、每 pod
   8 节点、每 rack 2 pod）。
3. 由构成推导每层流量份额：
   - 层次化集合通信：第 L 相在上一层单元间做 k_L 元集合；每层份额
     沿用现有 `(k-1)/k` 约定按相计算。
   - all2all：成员的对端按构成散布到各层；每层份额 =
     `(k_L 层外成员)/(k_total − 1)`。
4. p2p 域（pp send/recv）映射两端点并取路径上的层级。

映射复杂度 O(层级数)，不做全 world 枚举；levels = [node] 时精确
退化为 `group_node_stats`。

## 6. 分层成本合成（T3）

`compute_net_op_time` 新增 levels 路径（`topology.levels` 存在且
该通信域 net 字段为 `"auto"` 时启用）：

- **serial（集合通信）**：按层分解为多个相，总时 = Σ_L phase_L；
  每相使用该层 net 配置、该层子组规模（`k_L`）与该相流量。对应
  层次化 NCCL 行为（节点内 reduce → pod all_reduce → rack
  all_reduce → …）。
- **max（all2all）**：每对收发的时间受限于路径上最慢链路；总时
  = 各层传输时间取 max。
- **p2p**：路径上各层 serial 求和。
- 单项覆盖：`composition_policy` + 预留的 per-call `composition=`
  参数。
- 未声明 levels 时，现有 intra/单 net 路径完全不动。

## 7. net 字段语义（决策 C）

strategy 各通信域字段（`tp_net`、`pp_net`、…）：

- `"auto"`（默认）：有 `topology.levels` → 走 T2/T3 分层分解；
  无 → 保持现有二值解析。
- 显式指定（如 `"inter_node"`）：该通信域回退旧的单链路路径——
  文档化的逃生舱（如强制最差情况分析、模拟手工 rank 重排），行为
  与现状完全一致。

无需迁移任何存量配置；`"auto"` 只是变得更聪明。

## 8. fabric 层级服务台（T4，本期激活）

`NetworkFabric` 增加层级链路服务台，在新值
`fabric_model="nic+levels"` 下激活（`"nic"`/`"nic+tor"` 语义不变）：

- 服务台：per-GPU NIC（现有）+ 每（层级， 单元）一个逻辑链路服务
  台：`(pod, pod_id)`、`(rack, rack_id)`。
- inter-node entry 的 route：`[NIC(src), link(pod, src_pod),
  link(rack, src_rack), …, NIC(dst)]`，经过的层级由 T2 给出；
  不跨单元的层跳过其服务台。
- 服务台容量 = 层 net 带宽按单元活跃成员分摊，沿用 `node_share`
  放大机制并推广为 `level_share`（merge_lanes 下每层放大系数 =
  单元活跃 rank 数 / 被模拟 rank 数）。
- ToR（节点级）维持现状；pod/rack 计费沿用 size 占用公式，高估
  caveat 同样记录。

## 9. 验证

- 默认（无 levels）：三个 golden 用例逐事件等价；
  `group_level_span` 在单层级拓扑下等于 `group_node_stats`。
- 构成数学：打印一组（group_kind, sizes, levels）矩阵的分解结果
  ——含用户的 [2,8,2] 例子——人工核对。
- 成本：合成 3 层拓扑 + 跨 rack 的 dp 集合通信——serial 模式含
  rack 分量、max 模式含最慢层分量。
- placement：一个排列变体重算 stride 与构成，与手算一致。
- E2E：16384 卡 moe-8T 配 3 层拓扑，fabric 关 vs `"nic+levels"`
  A/B 报告。

## 10. 分阶段实施

- **Phase 1**——拓扑 + 映射 + 成本合成（T1、T2、T3、语义 C、默认
  placement）：配置字段、`group_level_span`、分层成本路径、验证。
  文档：system.md(+zh)。
- **Phase 2**——placement 排列（第 4 项目标）：
  `order_of_paralielism` 复活、stride 推导、mesh 范围校验。文档：
  strategy.md(+zh)。
- **Phase 3**——fabric 层级服务台激活（T4）：
  `fabric_model="nic+levels"`、route、level_share、E2E A/B。文档：
  system.md(+zh)。

## 11. 影响面

- `simumax/core/config.py`：`topology.levels`/`composition_policy`
  字段与校验；`compute_net_op_time` 的 levels 路径。
- `simumax/core/utils.py`：`group_level_span`、placement 推导
  stride。
- `simumax/core/base_struct.py`：fabric 层级服务台、route 处理。
- `simumax/core/simu_runner.py`：levels 的 fabric 构建。
- `simumax/core/perf_llm.py`：`analysis_net` 的语义 C。
- 文档：system.md / strategy.md（+zh 镜像）。

## 12. 决策记录

1. **合成策略**：`max` 与 `serial` 两种都留——all2all 默认 max
   （瓶颈层），层次化集合通信默认 serial（分相求和）；可配置覆盖。
2. **net 字段语义**：方案 C——`"auto"` 走分层分解，显式 net 回退
   旧单链路路径。用户补充：分解必须尊重**比例构成**（如
   [2,8,2]），且 **placement 本身也是配置**（维度在层级上的排布
   顺序）。
3. **fabric 层级服务台**：本期直接激活（pod/rack 计费生效），非
   纯预留。
