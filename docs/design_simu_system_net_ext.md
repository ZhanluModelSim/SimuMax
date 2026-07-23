<p align="center">
  <a href="design_simu_system_net_ext.md">English</a>|
  <a href="design_simu_system_net_ext-zh.md">中文版本</a>
</p>

# Design Proposal: System Network Configuration Extensions

- Status: **v1.0 (proposed)**
- Date: 2026-07-22
- Scope: FSDP net selector, physical topology types (FullMesh / CLOS),
  fabric contention activation in shipped configs.
- Builds on: `design_simu_network_fabric.md` (Phases A–C),
  `design_simu_hierarchical_network.md` (Phases 1–3),
  `design_simu_zero3_fsdp.md`, `design_simu_fsdp_mem_mfu_fix.md`.

## 1. Background and Problems

Three gaps in the current system network configuration were identified
during the FSDP and hierarchical-network work:

### 1.1 FSDP Has No Dedicated Net Selector

`StrategyConfig` has seven `*_net` fields (`tp_net`, `cp_net`, `pp_net`,
`dp_net`, `ep_net`, `etp_net`, `edp_net`), all defaulting to `"auto"`.
There is **no `fsdp_net`**. Every ZeRO level — DDP all_reduce
(zero_state=0), ZeRO-1/2 grad RS + param AG (zero_state 1–2), and
FSDP/ZeRO-3 unshard AG + reshard RS (zero_state ≥ 3) — resolves to the
same `dp_net` (dense) or `edp_net` (MoE expert).

In real deployments FSDP's all_gather (param unshard) and
reduce_scatter (grad reshard) may traverse different physical links
than DDP's all_reduce. Today the model cannot express this distinction.

### 1.2 Fabric Contention Not Activated in Shipped Configs

The `fabric_model` field (`"nic"`, `"nic+tor"`, `"nic+levels"`) was
implemented in 2026-07-17~21 (`design_simu_network_fabric.md` Phases
A–C, `design_simu_hierarchical_network.md` Phase 3). The shipped system
configs (`a100_pcie.json`, `b200_bf16_ceperm.json`) were last updated
2026-05-06 — before the feature landed. The `"nic"` tier requires zero
new data (reuses `num_per_node` + `networks["inter_node"]`), so the
omission is purely temporal.

### 1.3 No Physical Topology Type Declaration

The `topology.levels[i]` schema is strictly `{"name", "size", "net"}`
— no field describes the physical interconnect shape. The
shared-vs-dedicated bandwidth distinction is implicit:

| Path | Bandwidth sharing assumption | Equivalent physical topology |
|---|---|---|
| Legacy single-net (`compute_net_op_time`) | `bw /= num_per_node` (hardcoded shared uplink) | CLOS (shared uplink) |
| Levels analytical (`_compute_net_op_time_levels`) | Per-level independent pipe, no sharing | FullMesh (dedicated per-pair link) |
| DES Fabric (ToR / level server) | Controlled by `tor_capacity_gbps` / `level_capacities` | Depends on capacity values |

Users cannot declare whether a level uses FullMesh (dedicated per-pair
links, no bandwidth sharing) or CLOS (shared switch uplink with
convergence ratio). The CLOS convergence ratio (oversubscription
factor) is not expressible.

## 2. Goals / Non-Goals

### Goals

1. **FSDP net selector**: separate `fsdp_net` / `fsdp_moe_net` fields
   in `StrategyConfig`, defaulting to `"auto"` (inherits `dp_net` /
   `edp_net`), fully backward-compatible.
2. **Fabric activation**: enable `fabric_model` in shipped system
   configs where no new measured data is needed (`"nic"` for A100 PCIe,
   `"nic+tor"` for B200).
3. **Physical topology types**: add `kind` (`"fullmesh"` / `"clos"`)
   and `convergence_ratio` to `topology.levels[i]` entries and to
   `NetworkConfig`; wire through analytical and DES paths.
4. Full backward compatibility: all three changes are opt-in or
   default to current behavior.

### Non-goals

- Per-pair link server in DES (v1 uses pass-through for FullMesh).
- Route-level fidelity (rail mapping, adaptive routing).
- New measured link profiles for `nic+levels` (requires real hardware
  measurement; only example configs are provided).

## 3. Part A — FSDP Net Selector

### 3.1 New StrategyConfig Fields

```python
# config.py, StrategyConfig (after edp_net)
fsdp_net: Optional[str] = "auto"       # inherits dp_net
fsdp_moe_net: Optional[str] = "auto"   # inherits edp_net
```

### 3.2 Resolution Logic

In `PerfLLM.analysis_net()` (`perf_llm.py:447`):

- **Levels path** (`topology.levels` exists): `fsdp_net == "auto"` →
  `"levels"` (same as `dp_net`).
- **PCIe / high-link path**: `fsdp_net == "auto"` → inherits the
  already-resolved `dp_net` value (not `"auto"`).
- **Explicit** (non-`"auto"`): used verbatim.

A helper property avoids scattering the fallback logic:

```python
@property
def _fsdp_net_resolved(self):
    """Return fsdp_net if explicitly set, else the resolved dp_net."""
    fsdp_net = getattr(self.strategy, 'fsdp_net', 'auto')
    if fsdp_net and fsdp_net != 'auto':
        return fsdp_net
    return self.strategy.dp_net
```

### 3.3 Call-Site Changes

Only `zero_state >= 3` FSDP communications use the new selector.
ZeRO-0/1/2 and DDP continue using `dp_net` / `edp_net` unchanged.

**Analytical path** (`perf_llm.py`):

| Location | Current | After |
|---|---|---|
| `_compute_dp_time` dense call (1659) | `dp_net` | `fsdp_net` when `zero_state >= 3` |
| `_compute_dp_time` moe call (1660) | `edp_net` | `fsdp_moe_net` when `zero_state >= 3` |
| `_compute_layer_wise_fsdp_exposed_time` AG dense (1787) | `dp_net` | `_fsdp_net_resolved` |
| `_compute_layer_wise_fsdp_exposed_time` RS dense (1791) | `dp_net` | `_fsdp_net_resolved` |
| `_compute_layer_wise_fsdp_exposed_time` AG moe (1797) | `edp_net` | `_fsdp_moe_net_resolved` |
| `_compute_layer_wise_fsdp_exposed_time` RS moe (1801) | `edp_net` | `_fsdp_moe_net_resolved` |

**DES path — model-wise** (`transformer/pipeline_schedule.py`):

| Line | Op | Current net | After |
|---|---|---|---|
| 66 | AG dense | `dp_net` | `_fsdp_net_resolved` |
| 72 | AG moe | `edp_net` | `_fsdp_moe_net_resolved` |
| 84 | RS dense | `dp_net` | `_fsdp_net_resolved` |
| 90 | RS moe | `edp_net` | `_fsdp_moe_net_resolved` |

**DES path — layer-wise** (`transformer/language_model.py`):

| Method | Lines | Ops | After |
|---|---|---|---|
| `_build_fsdp_ag_ops` | 284, 292 | AG dense | `_fsdp_net_resolved` |
| `_build_fsdp_ag_ops` | 299, 307 | AG moe | `_fsdp_moe_net_resolved` |
| `_build_fsdp_rs_ops` | 328, 336 | RS dense | `_fsdp_net_resolved` |
| `_build_fsdp_rs_ops` | 343, 351 | RS moe | `_fsdp_moe_net_resolved` |
| `_build_fsdp_bwd_ag_ops` | 376, 384 | bwd AG dense | `_fsdp_net_resolved` |
| `_build_fsdp_bwd_ag_ops` | (moe) | bwd AG moe | `_fsdp_moe_net_resolved` |

### 3.4 comm_stage / group_kind

**Unchanged.** FSDP and DDP operate on the same dp_cp / edp groups, so
NIC contention (how many NICs are shared) is identical. `fsdp_net`
only changes the network profile (bandwidth / latency / fitted factors),
not the NIC sharing model.

### 3.5 Backward Compatibility

- `fsdp_net = "auto"` (default) → inherits `dp_net`: bit-identical.
- `zero_state < 3` → not affected.

## 4. Part B — Fabric Contention Activation

### 4.1 A100 PCIe: Enable `"nic"`

```json
"fabric_model": "nic"
```

No `topology` block needed. The `"nic"` tier uses `num_per_node` and
`networks["inter_node"]` which already exist. This activates per-GPU
NIC serialization for cross-node comm entries in the DES.

### 4.2 B200: Enable `"nic+tor"`

```json
"fabric_model": "nic+tor",
"topology": {
    "tor_capacity_gbps": 1600,
    "tor_node_share": "auto"
}
```

B200 has two intra-node tiers (`low_intra_node` / `high_intra_node`),
making ToR modeling meaningful. `tor_capacity_gbps = 1600` reflects a
non-oversubscribed 8 × 200 Gbps uplink. Set it lower to model
oversubscription.

### 4.3 `nic+levels` — Example Only

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

Not enabled in shipped configs because `inter_rack` profile requires
real measurement data not yet available.

### 4.4 Impact

- **DES path only**: `simulate()` gains fabric serialization.
- **Analytical path**: unchanged.
- **Backward compatibility**: `fabric_model = null` → fabric off →
  identical to current behavior.

## 5. Part C — Physical Topology Types

### 5.1 Level Entry Schema Extension

`_validate_topology_levels` (`config.py:1761`) currently requires
exactly `{"name", "size", "net"}`. Extended to accept optional `kind`
and `convergence_ratio`:

```json
{"name": "node", "size": 8, "net": "high_intra_node", "kind": "fullmesh"},
{"name": "pod",  "size": 32, "net": "inter_node", "kind": "clos", "convergence_ratio": 2.0}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | `"clos"` | `"fullmesh"` = dedicated per-pair links; `"clos"` = shared switch uplink |
| `convergence_ratio` | `float` | `1.0` | Oversubscription ratio; only meaningful with `kind="clos"` |

### 5.2 NetworkConfig Extension

For legacy single-net path (no `topology.levels`), add a
`topology_kind` field to `NetworkConfig`:

```python
@dataclass
class NetworkConfig:
    processor_usage: float
    bandwidth: BandwidthConfig
    op: Dict[str, OpConfig]
    topology_kind: str = "clos"   # "clos" (default) or "fullmesh"
```

When `topology.levels` exists, the level's `kind` overrides the net
profile's `topology_kind`. When it doesn't, the net's
`topology_kind` governs.

### 5.3 Bandwidth Model

| `kind` | Analytical legacy path | Analytical levels path | DES fabric |
|---|---|---|---|
| `"fullmesh"` | Skip `bw /= num_per_node` (dedicated per-pair) | `eff_bw = net.gbps` (current behavior) | ToR / level server pass-through (non-binding) |
| `"clos"` | `bw /= convergence_ratio` (replaces `bw /= num_per_node`) | `eff_bw = net.gbps / convergence_ratio` | ToR / level capacity = `net.gbps / convergence_ratio` |

### 5.4 Legacy Path Change

In `compute_net_op_time` (`config.py:1353`), the current hardcoded
`bw /= self.num_per_node` for `net == "inter_node"` is replaced by a
topology-kind-aware division:

```python
if net == "inter_node":
    topo_kind, conv_ratio = self._net_topology_kind(net)
    if topo_kind == "clos":
        bw /= conv_ratio  # replaces bw /= num_per_node
    # fullmesh: no division
```

`_net_topology_kind(net)` resolves the kind:
1. If `topology.levels` exists and `net` matches a level's `net` →
   return that level's `kind` / `convergence_ratio`.
2. Otherwise → return `networks[net].topology_kind` / `1.0`.

### 5.5 Levels Path Change

In `_compute_net_op_time_levels` (`config.py:1558`), after fetching
`bw` from `_level_net_params`, apply the convergence:

```python
scale, offset, eff_factor, bw, base_latency, fixed_latency = \
    self._level_net_params(span.net, op_name, comm_num)
kind = span.kind  ￼ # new field on LevelSpan
conv = span.convergence_ratio
if kind == "clos" and conv > 1.0:
    bw /= conv
```

### 5.6 DES Fabric Change

In `simu_runner.py:80-85`, `level_capacities` computation:

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

In `NetworkFabric`, FullMesh levels have their ToR / level server set to
non-binding (pass-through) by making `level_capacities[i]` very large
(or skipping the server). CLOS levels use the converged capacity.

### 5.7 Backward Compatibility

- `kind` defaults to `"clos"` and `convergence_ratio` to `1.0` →
  `bw /= 1.0` = no change for levels path; legacy path needs
  `bw /= num_per_node` → `bw /= 1.0` would break.
  - **Mitigation**: when `topology.levels` is absent AND
    `NetworkConfig.topology_kind` is the default `"clos"`, the legacy
    path keeps `bw /= num_per_node` (not `bw /= convergence_ratio`).
    Only when a user explicitly sets `topology_kind` or
    `convergence_ratio` does the new formula activate.
  - Equivalently: `num_per_node` is the default
    `convergence_ratio` for the legacy inter_node net when no explicit
    topology is declared.

## 6. Implementation Phases

### Phase 1 — FSDP Net Selector (Part A)

Files:
- `simumax/core/config.py` — `StrategyConfig` adds `fsdp_net` /
  `fsdp_moe_net`.
- `simumax/core/perf_llm.py` — `analysis_net` resolves new fields;
  `_compute_dp_time` and `_compute_layer_wise_fsdp_exposed_time` use
  resolved values; add `_fsdp_net_resolved` / `_fsdp_moe_net_resolved`
  properties.
- `simumax/core/transformer/pipeline_schedule.py` — model-wise FSDP
  AG/RS use resolved net.
- `simumax/core/transformer/language_model.py` — layer-wise FSDP
  AG/RS/bwd-AG use resolved net.
- `simumax/utils.py` — `create_default_strategy` if needed.

Validation: run an FSDP layer-wise config with `fsdp_net` set to an
explicit net name; verify DES trace shows the net field changed; unset
→ verify identical to `dp_net`.

### Phase 2 — Physical Topology Types (Part C)

Files:
- `simumax/core/config.py` — `NetworkConfig.topology_kind`;
  `_validate_topology_levels` extended for `kind` /
  `convergence_ratio`; `compute_net_op_time` legacy path kind-aware;
  `_compute_net_op_time_levels` kind-aware;
  `_net_topology_kind` helper.
- `simumax/core/base_struct.py` — `NetworkFabric.set_level_topology`
  receives and applies kind / convergence_ratio.
- `simumax/core/simu_runner.py` — constructs `level_capacities` with
  convergence.
- `simumax/core/utils.py` — `LevelSpan` gains `kind` /
  `convergence_ratio` fields.

Validation: configure a FullMesh level vs a CLOS level with
`convergence_ratio=2.0`; verify bandwidth division is correct in both
analytical and DES paths.

### Phase 3 — Fabric Activation + Documentation (Part B + docs)

Files:
- `configs/system/b200_bf16_ceperm.json` — add `fabric_model: "nic+tor"`
  + topology tor knobs.
- `configs/system/a100_pcie.json` — add `fabric_model: "nic"`.
- `docs/design_simu_system_net_ext.md` + `-zh.md` — this document.
- `docs/system.md` / `docs/system-zh.md` — update with new fields.
- `AGENTS.md` — update if conventions changed.

Validation: run A100 and B200 `simulate()`; confirm fabric contention
active (comm entries serialized on NIC/ToR servers); analytical results
unchanged.

## 7. Open Questions and Recommendations

### Q1: Does FSDP need its own `comm_stage`?

**Recommendation: No.** FSDP and DDP operate on the same dp_cp / edp
groups. The NIC contention model (how many NICs are shared) is
identical. `fsdp_net` only changes the network profile, not the NIC
sharing math.

### Q2: Default `topology_kind` on `NetworkConfig`

Default `"clos"` matches current legacy behavior (`bw /= num_per_node`).
When `topology.levels` exists with `kind="fullmesh"`, the level's kind
overrides. When `net == "inter_node"` but no `topology.levels` is
declared, the net's `topology_kind` governs. The default `convergence_ratio`
for the legacy path is `num_per_node` (preserving `bw /= num_per_node`),
not `1.0`.

### Q3: FullMesh precision in DES

**Recommendation: v1 pass-through.** FullMesh means no shared switch
bandwidth — in DES this corresponds to the ToR / level server being
non-binding (never pushes `launch_t` later). The per-GPU NIC server
already models per-GPU independence. Per-pair link servers (N(N-1)/2
servers) are deferred to future work.
