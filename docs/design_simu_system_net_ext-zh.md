<p align="center">
  <a href="design_simu_system_net_ext.md">English</a>|
  <a href="design_simu_system_net_ext-zh.md">中文版本</a>
</p>

# 设计方案：System 网络配置扩展

- 状态：**v1.0（已实现）**
- 日期：2026-07-22
- 实现：Phase 1 `384d639`（FSDP net 选择器）、Phase 2 `0ea9e31`（FullMesh/CLOS 拓扑类型）、Phase 3（fabric 激活 + 文档）。
- 范围：FSDP net 选择器、物理拓扑类型（FullMesh / CLOS）、fabric contention 在 shipped 配置中激活。
- 前置依赖：`design_simu_network_fabric.md`（Phase A–C）、`design_simu_hierarchical_network.md`（Phase 1–3）、`design_simu_zero3_fsdp.md`、`design_simu_fsdp_mem_mfu_fix.md`。

## 1. 背景与问题

在 FSDP 和层级网络的工作中发现了当前 system 网络配置的三个问题：

### 1.1 FSDP 没有独立的 net 选择器

`StrategyConfig` 有 7 个 `*_net` 字段（`tp_net`、`cp_net`、`pp_net`、`dp_net`、`ep_net`、`etp_net`、`edp_net`），默认都是 `"auto"`，**没有 `fsdp_net`**。所有 ZeRO 级别——DDP all_reduce（zero_state=0）、ZeRO-1/2 grad RS + param AG（zero_state 1–2）、FSDP/ZeRO-3 unshard AG + reshard RS（zero_state ≥ 3）——都解析到同一个 `dp_net`（dense）或 `edp_net`（MoE expert）。

实际部署中 FSDP 的 all_gather（参数 unshard）和 reduce_scatter（梯度 reshard）可能走与 DDP all_reduce 不同的物理链路。当前模型无法表达这一区别。

### 1.2 shipped 配置未启用 fabric contention

`fabric_model` 字段（`"nic"`、`"nic+tor"`、`"nic+levels"`）在 2026-07-17~21 实现。shipped system 配置（`a100_pcie.json`、`b200_bf16_ceperm.json`）最后更新于 2026-05-06，早于功能落地。`"nic"` 模式零数据依赖（复用 `num_per_node` + `networks["inter_node"]`），缺失纯粹是时间原因。

### 1.3 没有物理拓扑类型声明

`topology.levels[i]` 严格为 `{"name", "size", "net"}`，没有描述物理互联形状的字段。带宽共享与专用是隐式的：

| 路径 | 带宽共享假设 | 等价物理拓扑 |
|---|---|---|
| Legacy 单网络路径（`compute_net_op_time`） | `bw /= num_per_node`（硬编码共享上行） | CLOS（共享上行链路） |
| Levels 分析路径（`_compute_net_op_time_levels`） | 每层独立 pipe，无共享 | FullMesh（专用 per-pair 链路） |
| DES Fabric（ToR / level server） | 通过 `tor_capacity_gbps` / `level_capacities` 容量控制 | 取决于容量数值 |

用户无法声明某层是 FullMesh（专用 per-pair 链路，不共享带宽）还是 CLOS（共享交换机上行，有收敛比）。CLOS 的收敛比（oversubscription ratio）无法表达。

## 2. 目标 / 非目标

### 目标

1. **FSDP net 选择器**：`StrategyConfig` 增加 `fsdp_net` / `fsdp_moe_net` 字段，默认 `"auto"`（继承 `dp_net` / `edp_net`），完全向后兼容。
2. **Fabric 激活**：在不需要新测量数据的 shipped system 配置中启用 `fabric_model`（A100 PCIe 用 `"nic"`，B200 用 `"nic+tor"`）。
3. **物理拓扑类型**：在 `topology.levels[i]` 和 `NetworkConfig` 中增加 `kind`（`"fullmesh"` / `"clos"`）和 `convergence_ratio`，贯穿分析和 DES 路径。
4. 完全向后兼容：三项改动都是 opt-in 或默认为当前行为。

### 非目标

- DES 中的 per-pair link server（v1 用 pass-through 建模 FullMesh）。
- 路由级精度（rail mapping、自适应路由）。
- `nic+levels` 的新测量链路 profile（需要实机测量；只提供示例配置）。

## 3. Part A — FSDP net 选择器

### 3.1 新增 StrategyConfig 字段

```python
# config.py, StrategyConfig (edp_net 之后)
fsdp_net: Optional[str] = "auto"       # 继承 dp_net
fsdp_moe_net: Optional[str] = "auto"   # 继承 edp_net
```

### 3.2 解析逻辑

在 `PerfLLM.analysis_net()`（`perf_llm.py:447`）中：

- **levels 路径**（`topology.levels` 存在）：`fsdp_net == "auto"` → `"levels"`（与 `dp_net` 一致）。
- **PCIe / high-link 路径**：`fsdp_net == "auto"` → 继承已解析的 `dp_net` 值（不是 `"auto"`）。
- **显式指定**（非 `"auto"`）：直接使用。

helper 属性避免 fallback 逻辑散落：

```python
@property
def _fsdp_net_resolved(self):
    """返回 fsdp_net 如果显式设置，否则返回已解析的 dp_net。"""
    fsdp_net = getattr(self.strategy, 'fsdp_net', 'auto')
    if fsdp_net and fsdp_net != 'auto':
        return fsdp_net
    return self.strategy.dp_net
```

### 3.3 调用点修改

仅 `zero_state >= 3` 的 FSDP 通信使用新选择器。ZeRO-0/1/2 和 DDP 继续使用 `dp_net` / `edp_net`。

**分析路径**（`perf_llm.py`）：

| 位置 | 当前 | 修改后 |
|---|---|---|
| `_compute_dp_time` dense 调用 (1659) | `dp_net` | `zero_state >= 3` 时用 `fsdp_net` |
| `_compute_dp_time` moe 调用 (1660) | `edp_net` | `zero_state >= 3` 时用 `fsdp_moe_net` |
| `_compute_layer_wise_fsdp_exposed_time` AG dense (1787) | `dp_net` | `_fsdp_net_resolved` |
| `_compute_layer_wise_fsdp_exposed_time` RS dense (1791) | `dp_net` | `_fsdp_net_resolved` |
| `_compute_layer_wise_fsdp_exposed_time` AG moe (1797) | `edp_net` | `_fsdp_moe_net_resolved` |
| `_compute_layer_wise_fsdp_exposed_time` RS moe (1801) | `edp_net` | `_fsdp_moe_net_resolved` |

**DES 路径 — model-wise**（`transformer/pipeline_schedule.py`）：

| 行 | 算子 | 当前 net | 修改后 |
|---|---|---|---|
| 66 | AG dense | `dp_net` | `_fsdp_net_resolved` |
| 72 | AG moe | `edp_net` | `_fsdp_moe_net_resolved` |
| 84 | RS dense | `dp_net` | `_fsdp_net_resolved` |
| 90 | RS moe | `edp_net` | `_fsdp_moe_net_resolved` |

**DES 路径 — layer-wise**（`transformer/language_model.py`）：

| 方法 | 行 | 算子 | 修改后 |
|---|---|---|---|
| `_build_fsdp_ag_ops` | 284, 292 | AG dense | `_fsdp_net_resolved` |
| `_build_fsdp_ag_ops` | 299, 307 | AG moe | `_fsdp_moe_net_resolved` |
| `_build_fsdp_rs_ops` | 328, 336 | RS dense | `_fsdp_net_resolved` |
| `_build_fsdp_rs_ops` | 343, 351 | RS moe | `_fsdp_moe_net_resolved` |
| `_build_fsdp_bwd_ag_ops` | 376, 384 | bwd AG dense | `_fsdp_net_resolved` |
| `_build_fsdp_bwd_ag_ops` | (moe) | bwd AG moe | `_fsdp_moe_net_resolved` |

### 3.4 comm_stage / group_kind

**不变。** FSDP 和 DDP 在同一个 dp_cp / edp group 上操作，NIC contention（共享多少 NIC）相同。`fsdp_net` 只改变网络 profile（带宽 / 延迟 / 拟合参数），不改变 NIC 共享模型。

### 3.5 向后兼容

- `fsdp_net = "auto"`（默认）→ 继承 `dp_net`：bit-identical。
- `zero_state < 3` → 不受影响。

## 4. Part B — Fabric contention 激活

### 4.1 A100 PCIe：启用 `"nic"`

```json
"fabric_model": "nic"
```

不需要 `topology` 块。`"nic"` 模式复用已有的 `num_per_node` 和 `networks["inter_node"]`。激活后 DES 中跨节点通信条目会有 per-GPU NIC 排队。

### 4.2 B200：启用 `"nic+tor"`

```json
"fabric_model": "nic+tor",
"topology": {
    "tor_capacity_gbps": 1600,
    "tor_node_share": "auto"
}
```

B200 有两档 intra-node（`low_intra_node` / `high_intra_node`），ToR 建模有意义。`tor_capacity_gbps = 1600` 反映无收敛的 8 × 200 Gbps 上行。设低可建模 oversubscription。

### 4.3 `nic+levels` — 仅示例

```json
"fabric_model": "nic+levels",
"topology": {
    "levels": [
        {"name": "node", "size": 8,   "net": "high_intra_node"},
        {"name": "pod",  "size": 32,  "net": "inter_node"},
        {"name": "rack", "size": 256, "net": "inter_rack"}
    ]
}
```

不在 shipped 配置中启用，因为 `inter_rack` profile 需要实机测量数据。

### 4.4 影响

- **仅 DES 路径**：`simulate()` 获得 fabric 排队。
- **分析路径**：不变。
- **向后兼容**：`fabric_model = null` → fabric 关闭 → 与当前行为一致。

## 5. Part C — 物理拓扑类型

### 5.1 Level entry schema 扩展

`_validate_topology_levels`（`config.py:1761`）当前要求严格 `{"name", "size", "net"}`。扩展为接受可选 `kind` 和 `convergence_ratio`：

```json
{"name": "node", "size": 8, "net": "high_intra_node", "kind": "fullmesh"},
{"name": "pod",  "size": 32, "net": "inter_node", "kind": "clos", "convergence_ratio": 2.0}
```

| 字段 | 类型 | 默认 | 描述 |
|---|---|---|---|
| `kind` | `str` | `"clos"` | `"fullmesh"` = 专用 per-pair 链路；`"clos"` = 共享交换机上行 |
| `convergence_ratio` | `float` | `1.0` | 收敛比；仅 `kind="clos"` 时有意义 |

### 5.2 NetworkConfig 扩展

对 legacy 单网络路径（无 `topology.levels`），给 `NetworkConfig` 增加可选 `topology_kind` 字段：

```python
@dataclass
class NetworkConfig:
    processor_usage: float
    bandwidth: BandwidthConfig
    op: Dict[str, OpConfig]
    topology_kind: str = "clos"   # "clos"（默认）或 "fullmesh"
```

当 `topology.levels` 存在时，level 的 `kind` 覆盖 net 的 `topology_kind`。不存在时，用 net 的 `topology_kind`。

### 5.3 带宽模型

| `kind` | 分析 legacy 路径 | 分析 levels 路径 | DES fabric |
|---|---|---|---|
| `"fullmesh"` | 跳过 `bw /= num_per_node`（专用 per-pair） | `eff_bw = net.gbps`（当前行为） | ToR / level server pass-through（不绑定） |
| `"clos"` | `bw /= convergence_ratio`（替代 `bw /= num_per_node`） | `eff_bw = net.gbps / convergence_ratio` | ToR / level capacity = `net.gbps / convergence_ratio` |

### 5.4 Legacy 路径修改

在 `compute_net_op_time`（`config.py:1353`），当前 `net == "inter_node"` 时的硬编码 `bw /= self.num_per_node` 替换为拓扑类型感知的除法：

```python
if net == "inter_node":
    topo_kind, conv_ratio = self._net_topology_kind(net)
    if topo_kind == "clos":
        bw /= conv_ratio  # 替代 bw /= num_per_node
    # fullmesh: 不除
```

`_net_topology_kind(net)` 解析 kind：
1. 如果 `topology.levels` 存在且 `net` 匹配某 level 的 `net` → 返回该 level 的 `kind` / `convergence_ratio`。
2. 否则 → 返回 `networks[net].topology_kind` / `1.0`。

### 5.5 Levels 路径修改

在 `_compute_net_op_time_levels`（`config.py:1558`），获取 `bw` 后应用收敛：

```python
scale, offset, eff_factor, bw, base_latency, fixed_latency = \
    self._level_net_params(span.net, op_name, comm_num)
kind = span.kind  # LevelSpan 新字段
conv = span.convergence_ratio
if kind == "clos" and conv > 1.0:
    bw /= conv
```

### 5.6 DES Fabric 修改

在 `simu_runner.py:80-85`，`level_capacities` 计算：

```python
level_capacities = []
for level in levels:
    net_bw = perf_model.system.networks[level["net"]].bandwidth.gbps
    kind = level.get("kind", "clos")
    conv = level.get("convergence_ratio", 1.0)
    if kind == "clos" and conv > 1.0:
        net_bw /= conv
    level_capacities.append(net_bw)
```

FullMesh 层的 ToR / level server 设为不绑定（pass-through），通过让 `level_capacities[i]` 非常大或跳过该 server 实现。CLOS 层使用收敛后容量。

### 5.7 向后兼容

- `kind` 默认 `"clos"`，`convergence_ratio` 默认 `1.0` → levels 路径 `bw /= 1.0` 不变；legacy 路径需 `bw /= num_per_node` → `bw /= 1.0` 会 break。
  - **缓解**：当 `topology.levels` 不存在且 `NetworkConfig.topology_kind` 为默认 `"clos"` 时，legacy 路径保持 `bw /= num_per_node`（不是 `bw /= convergence_ratio`）。只有用户显式设置 `topology_kind` 或 `convergence_ratio` 时新公式才激活。
  - 等价地：legacy inter_node net 的默认 `convergence_ratio` 是 `num_per_node`（保持 `bw /= num_per_node`），不是 `1.0`。

## 6. 实现 Phase

### Phase 1 — FSDP net 选择器（Part A）

文件：
- `simumax/core/config.py` — `StrategyConfig` 加 `fsdp_net` / `fsdp_moe_net`。
- `simumax/core/perf_llm.py` — `analysis_net` 解析新字段；`_compute_dp_time` 和 `_compute_layer_wise_fsdp_exposed_time` 使用解析值；加 `_fsdp_net_resolved` / `_fsdp_moe_net_resolved` 属性。
- `simumax/core/transformer/pipeline_schedule.py` — model-wise FSDP AG/RS 使用解析 net。
- `simumax/core/transformer/language_model.py` — layer-wise FSDP AG/RS/bwd-AG 使用解析 net。
- `simumax/utils.py` — `create_default_strategy` 如需更新。

验证：跑 FSDP layer-wise 配置，设置 `fsdp_net` 为显式网络名，确认 DES trace 中 net 字段已切换；不设置时确认与 `dp_net` 一致。

### Phase 2 — 物理拓扑类型（Part C）

文件：
- `simumax/core/config.py` — `NetworkConfig.topology_kind`；`_validate_topology_levels` 扩展 `kind` / `convergence_ratio`；`compute_net_op_time` legacy 路径 kind 感知；`_compute_net_op_time_levels` kind 感知；`_net_topology_kind` helper。
- `simumax/core/base_struct.py` — `NetworkFabric.set_level_topology` 接收并应用 kind / convergence_ratio。
- `simumax/core/simu_runner.py` — 构造 `level_capacities` 时带收敛。
- `simumax/core/utils.py` — `LevelSpan` 增加 `kind` / `convergence_ratio` 字段。

验证：配置 FullMesh level 对比 CLOS level（`convergence_ratio=2.0`），确认分析和 DES 路径带宽除法正确。

### Phase 3 — Fabric 激活 + 文档（Part B + 文档）

文件：
- `configs/system/b200_bf16_ceperm.json` — 增加 `fabric_model: "nic+tor"` + topology tor 旋钮。
- `configs/system/a100_pcie.json` — 增加 `fabric_model: "nic"`。
- `docs/design_simu_system_net_ext.md` + `-zh.md` — 本文档。
- `docs/system.md` / `docs/system-zh.md` — 更新 system 配置文档。
- `AGENTS.md` — 如有 conventions 变化则更新。

验证：跑 A100 和 B200 `simulate()`，确认 fabric contention 生效（comm entry 有 fabric 排队），分析路径结果不变。

## 7. 开放问题与建议

### Q1：FSDP 是否需要新增 `comm_stage`？

**建议：不需要。** FSDP 和 DDP 在同一个 dp_cp / edp group 上操作，NIC contention 模型（共享多少 NIC）相同。`fsdp_net` 只改变网络 profile，不改变 NIC 共享数学。

### Q2：`NetworkConfig` 上的默认 `topology_kind`

默认 `"clos"` 与当前 legacy 行为一致（`bw /= num_per_node`）。当 `topology.levels` 存在且 `kind="fullmesh"` 时，level 的 kind 覆盖。当 `net == "inter_node"` 但无 `topology.levels` 时，用 net 的 `topology_kind`。legacy 路径的默认 `convergence_ratio` 是 `num_per_node`（保持 `bw /= num_per_node`），不是 `1.0`。

### Q3：FullMesh 在 DES 中的精度

**建议：v1 用 pass-through。** FullMesh 意味着无共享交换机带宽——DES 中对应 ToR / level server 不绑定（不推迟 `launch_t`）。per-GPU NIC server 已经建模了 per-GPU 独立。per-pair link server（N(N-1)/2 个 server）留作 future work。
