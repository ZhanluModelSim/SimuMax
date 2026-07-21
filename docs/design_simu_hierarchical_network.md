<p align="center">
  <a href="design_simu_hierarchical_network.md">English</a>|
  <a href="design_simu_hierarchical_network-zh.md">中文版本</a>
</p>

# Design Proposal: Hierarchical Network Topology

- Status: **v1.0 (Phases 1-3 implemented)**
- Implementation: phase 1 `6b9dd4d`, phase 2 `a32519a`, phase 3 `206f879`.
- Date: 2026-07-17
- Scope: network topology declaration, comm-domain→level mapping,
  per-level cost composition, fabric level servers, placement config.
  Builds on `design_simu_network_fabric.md` (Phases A–C) and the
  group→node analytics of `core/utils.py`.

## 1. Background and Problems

The current network model is a flat, two-level one:

1. `networks` in system.json is a flat list of link profiles
   (`high_intra_node`, `inter_node`, pcie variants); links have no
   hierarchy relation.
2. Net selection is binary (`analysis_high_link_net`,
   `perf_llm.py:419-446`): each comm domain resolves to intra-node or
   inter-node by `group_span <= num_per_node`.
3. `num_per_node` is the only topology parameter; no rack/pod concepts
   exist anywhere.
4. One collective hangs on exactly one net: `compute_net_op_time`
   computes cost from a single link's fitted parameters. There is no
   "use several levels of physical links" notion.

What production clusters need (and this design adds): N GPUs per node
via link A, M nodes per pod via link B, P pods per rack via link C —
each level with its own bandwidth/latency — and every comm domain
charged through the levels it actually spans, in the right proportions.

## 2. Goals / Non-Goals

Goals (per the decisions in section 12):

1. Declarative multi-level topology in system.json (node/pod/rack/…),
   each level with its own net profile.
2. Comm-domain → level mapping with **proportional decomposition**:
   a group of 32 may compose as [2 in-node] × [8 in-pod] × [2 in-rack],
   and those proportions drive the per-level traffic fractions.
3. Per-level cost composition with **per-collective-type policies**:
   all2all takes the slowest level (max), hierarchical collectives
   (all_reduce/all_gather/reduce_scatter) sum over level phases
   (serial), both overridable by config.
4. Net-field semantics C (explicit-override fallback); placement is a
   first-class configuration.
5. Fabric pod/rack servers **activated** (not just reserved).
6. Full backward compatibility: no `topology.levels` ⇒ bit-identical
   results.

Non-goals:

- Route-level fidelity (rail mapping, adaptive routing, congestion
  spreading).
- MoE-mesh placement variants (the dense mesh tp/cp/dp/pp is
  configurable; the MoE mesh ep/etp/edp keeps its current fixed order —
  see 5.4).

## 3. Topology Declaration (system.json)

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

- `levels` are ordered innermost→outermost. `size` = how many units of
  the previous level this level contains (node.size=8 ⇒ 8 GPUs/node;
  pod.size=32 ⇒ 32 nodes/pod = 256 GPUs; rack.size=256 ⇒ 256
  pods/rack). The first level's "unit" is one GPU.
- `net` points at an entry in the existing `networks` dict — zero schema
  change there; adding `inter_rack` etc. is just data. Each level's
  bandwidth/latency/fitted op factors come from that net entry.
- The first level must be the node level whose size equals
  `num_per_node` (validated; keeps every existing `num_per_node`-based
  correction consistent).
- `composition_policy`: per-collective-type composition for section 6;
  defaults shown (`all2all` → max, ring/tree collectives → serial,
  `p2p` → serial). Individual entries overridable.

## 4. Placement Configuration (strategy)

Placement decides how parallel dims are laid onto the physical
hierarchy — it determines the per-level proportions of every comm
domain.

- Reactivate `order_of_paralielism` (currently documentation-only,
  validated to a single value) as the placement field:
  default `"tp-cp-ep-dp-pp"` = today's hardcoded mesh (innermost first).
  Validation expands to documented permutations of the dense dims.
- The stride table used by `group_node_stats` (Phase A) is derived from
  the placement: with the default order the strides stay exactly
  tp=1, cp=tp, dp=tp·cp, pp=tp·cp·dp (bit-identical to today); any
  other declared permutation recomputes them accordingly.
- Explicit rank remaps (user-defined placements) are out of v1 scope;
  v1 covers order permutations only.

## 5. Comm-Domain → Level Mapping (T2)

`group_level_span(group_kind, strategy, levels) -> list[LevelSpan]` in
`core/utils.py`, generalizing `group_node_stats`:

1. Members of the group form an arithmetic progression
   `base + k*stride` (stride from the placement).
2. Walk the levels: with cumulative span `S_L` (product of sizes up to
   that level), compute how many members sit inside one L-unit and how
   many L-units the group touches — producing the **composition**
   `[k_1, k_2, …]`, e.g. `[2, 8, 2]` for a 32-member group (2 per node,
   8 nodes per pod, 2 pods per rack).
3. From the composition derive per-level traffic fractions:
   - hierarchical collectives: phase L collectively involves `k_L`
     units of the previous level; fraction modeled per level following
     the existing `(k-1)/k` convention per phase.
   - all2all: a member's peers distribute across levels per the
     composition; the per-level share of each member's traffic is
     `(k_L remaining)/(k_total − 1)`.
4. p2p domains (pp send/recv) map both endpoints and take the levels on
   the path between them.

The mapping is O(levels), no world-size enumeration, and reduces to
`group_node_stats` exactly when levels = [node].

## 6. Per-Level Cost Composition (T3)

`compute_net_op_time` gains a levels path (used when
`topology.levels` exists and the domain's net field is `"auto"`):

- **serial (collectives)**: the collective is decomposed into phases
  per level; total time = Σ_L phase_time_L, where each phase uses that
  level's net profile with the level's sub-group size (`k_L`) and the
  phase's traffic size. Matches hierarchical NCCL behavior (intra-node
  reduce → pod all_reduce → rack all_reduce → …).
- **max (all2all)**: each pair's time is bounded by the slowest link on
  its path; total = max over levels of the per-level transfer time.
- **p2p**: serial over the levels on the endpoint path.
- Per-op override: `composition_policy` plus an optional per-call
  `composition=` argument for future fine control.
- The existing intra-node / single-net path is untouched when no
  levels are declared.

## 7. Net-Field Semantics (decision C)

Per strategy comm family (`tp_net`, `pp_net`, …):

- `"auto"` (default): with `topology.levels` present, use the T2/T3
  level decomposition; without levels, keep today's binary resolution.
- Explicitly set (e.g. `"inter_node"`): legacy single-net path for that
  family — the documented escape hatch (e.g. force worst-case analysis
  or emulate a rank remap), working exactly as before.

No config migration needed; `"auto"` simply gets smarter.

## 8. Fabric Level Servers (T4, activated)

`NetworkFabric` gains per-level link servers, activated under
`fabric_model="nic+levels"` (new value; `"nic"`/`"nic+tor"` keep their
current meaning):

- Servers: per-GPU NIC (existing), then one logical link server per
  (level, unit): `(pod, pod_id)`, `(rack, rack_id)`.
- Route of an inter-node entry: `[NIC(src), link(pod, src_pod),
  link(rack, src_rack), …, NIC(dst)]` for the levels the entry's
  traffic crosses per T2; intra-level hops that stay within one unit
  skip that unit's server.
- Server capacity: level bandwidth per the level's net profile,
  divided among the unit's active members using the existing
  `node_share` amplification generalized to `level_share` (merge_lanes
  amplification per level = active ranks per unit / simulated ranks per
  unit).
- ToR (per-node) remains as today; pod/rack charging follows the same
  size-based occupancy formula with its own overcharge caveats
  documented.

## 9. Validation

- Default (no levels): the three golden cases stay event-for-event
  identical; `group_level_span` on a 1-level topology equals
  `group_node_stats`.
- Composition math: printed decompositions for a matrix of
  (group_kind, sizes, levels) — including the user's [2,8,2] example —
  checked by hand.
- Cost: synthetic 3-level topology, a dp collective crossing rack —
  cost contains the rack component in `serial` mode and the max-level
  component in `max` mode.
- Placement: a permuted placement reproduces hand-computed strides and
  compositions.
- E2E: 16384-GPU moe-8T with a 3-level topology, fabric off vs
  `"nic+levels"` A/B report.

## 10. Phased Implementation

- **Phase 1** — topology + mapping + cost composition (T1, T2, T3,
  semantics C, default placement): config fields, `group_level_span`,
  per-level cost path, validation. Docs: system.md(+zh).
- **Phase 2** — placement permutations (T4 of scope 4):
  `order_of_paralielism` reactivated, stride derivation, mesh-wide
  checks. Docs: strategy.md(+zh).
- **Phase 3** — fabric level servers activated (T4):
  `fabric_model="nic+levels"`, routes, level_share, E2E A/B. Docs:
  system.md(+zh).

## 11. Impact Surface

- `simumax/core/config.py`: `topology.levels`/`composition_policy`
  fields + validation; levels path in `compute_net_op_time`.
- `simumax/core/utils.py`: `group_level_span`, placement-derived
  strides.
- `simumax/core/base_struct.py`: fabric level servers, route handling.
- `simumax/core/simu_runner.py`: fabric construction for levels.
- `simumax/core/perf_llm.py`: net-field semantics C in `analysis_net`.
- docs: system.md / strategy.md (+zh mirrors).

## 12. Decisions Log

1. **Composition policy**: both `max` and `serial` kept — all2all
   defaults to max (bottleneck level), hierarchical collectives default
   to serial (phase sum); config-overridable.
2. **Net-field semantics**: option C — `"auto"` uses level
   decomposition, explicit net values fall back to the legacy
   single-net path. The user added that the decomposition must respect
   **proportional composition** (e.g. [2,8,2]) and that **placement is
   itself a configuration** (dimension ordering across the hierarchy).
3. **Fabric level servers**: activated in this round (pod/rack cost
   accounting live), not merely reserved.
