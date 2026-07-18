<p align="center">
  <a href="design_simu_network_fabric.md">English</a>|
  <a href="design_simu_network_fabric-zh.md">中文版本</a>
</p>

# Design Proposal: Network Fabric Modeling for the DES

- Status: **Draft v0.1** (design agreed in discussion, not yet implemented)
- Date: 2026-07-17
- Scope: `simulate()` DES path — collective/p2p communication timing under
  cross-node traffic. Builds on the resource-lane and simu_kind groundwork
  of `design_simu_kind_resource_model.md`.

## 1. Background and Problems

With `merge_lanes=True` (the default), only `pp_size` representative ranks
are simulated, and every collective except `default_group`/`send_recv-*`
gets `backend_kind="local"` (`base_struct.py:2351-2358`). A local entry
completes via `_pump_local_entry` (`base_struct.py:1917-1922`):

```
launch_t = max(issue_t, rank_tail(stream));  end_t = launch_t + cost
```

i.e. it is serialized only against the issuing rank's own `(rank, stream)`
comm lane. The `cost` is a static scalar computed at prefill by
`SystemConfig.compute_net_op_time` (`config.py:973`) with heuristic
cross-node corrections; after that, size/net/topology information is
discarded and never reaches the DES.

Three gaps for cross-node communication:

1. **No NIC-level resource.** The DES gives each rank independent
   `comm`/`pp_fwd`/`pp_bwd` lanes, so a GPU's dp reduce-scatter, ep
   all2all, and pp p2p can all run "at the same time" with no mutual
   interference. In reality they share that GPU's NIC.
2. **No cross-rank synchronization.** Local entries do not rendezvous with
   group members; the "slowest member arrives" effect (skew/straggler) is
   absent.
3. **Topology blindness.** Net placement is a group-size heuristic
   (`analysis_high_link_net`, `perf_llm.py:419-468`), not derived from the
   rank↔node mapping; inter-node corrections in `compute_net_op_time`
   cover only dp/edp/p2p/all2all callers — TP/CP collectives assigned to
   `inter_node` get no correction at all.

## 2. Goals / Non-Goals

Goals:

1. Model per-GPU NIC contention for all inter-node traffic in the DES
   (the dominant cross-node effect at scale).
2. Keep the model opt-in: default configs produce bit-identical results
   to today.
3. Make the cost model topology-aware (real member-node ratios instead of
   group-size heuristics) where it is currently blind.
4. Reserve the structure for node/ToR-level contention and for cross-rank
   skew, without committing to them now.

Non-goals (this round):

- Full-world rendezvous (`merge_lanes=False` remains the only true
  cross-rank sync mode; skew modeling is Tier C, reserved).
- Network-route-level fidelity (rail mapping, congestion spreading,
  adaptive routing).
- Changing any default behavior.

## 3. Design Overview — Three Tiers

| Tier | Content | Changes | Risk |
|---|---|---|---|
| A | Topology-aware static corrections | cost model only (`config.py`) | low |
| B | `NetworkFabric`: per-GPU NIC servers + p2p both-end charging + ToR structure reserved | DES kernel + Com meta + config | medium |
| C | Cross-rank skew (virtual waiters / group-representative sim) | DES kernel | reserved |

Tiers are independently mergeable in the order A → B → C.

## 4. Tier A — Topology-Aware Static Corrections

- Introduce a group→node mapping helper (e.g. `core/utils.py`):
  given `group_kind` (tp/cp/dp/dp_cp/pp/ep/etp/edp), the strategy sizes and
  `num_per_node`, compute analytically (members form an arithmetic
  progression: stride 1 for tp/ep, `tp` for cp, `tp*cp` for dp,
  `tp*cp*dp` for pp, `ep` for edp) the member-node count and the
  cross-node traffic fraction of any collective. No world-size
  enumeration.
- Generalize the inter-node corrections in `compute_net_op_time` to all
  op kinds (TP/CP collectives included), replacing the group-size
  heuristics with the real cross-node ratios from the helper.
- Existing corrections stay for backward compatibility behind the same
  formulas; new ratios only refine cases that currently get nothing.

## 5. Tier B — NetworkFabric (core)

### 5.1 Resource model

`NetworkFabric` is a global object owned by `SimuContext`:

- **NIC servers**: one per GPU, keyed by `global_rank`. Capacity follows
  the existing convention: `inter_node.gbps / num_per_node` (per-GPU NIC
  bandwidth). Because a NIC belongs to a GPU (not a node), per-rank NIC
  servers need **no traffic amplification** under `merge_lanes` — each
  simulated rank owns its NIC outright.
- **ToR servers** (reserved, decision 3): one per node, keyed by
  `rank // num_per_node`. Route and pump structures are multi-hop from
  day one; ToR servers are created but default to non-constraining
  (pass-through) until the node-share amplification model lands (§5.5).
- Server state is a single tail clock per server (`nic_tail[rank]`,
  `tor_tail[node]`), mirroring `rank_comm_tail`.

### 5.2 What gets charged

An entry is fabric-charged iff its resolved net name is `inter_node`:

- prefill: `Com` gains optional `net=` (and `size_bytes=`) constructor
  args; module/pipeline call sites pass the already-resolved strategy net
  name (`strategy.tp_net` etc. — resolved by `analysis_net` during
  `run_estimate`, before job building). Default `None` = not charged =
  current behavior, so unmigrated call sites are unaffected.
- issue: `net`/`size_bytes` ride `CommEntry.meta` (existing field) into
  the kernel.
- The optimizer's DP reduce-scatter/all-gather (`pipeline_schedule.py`)
  carries `dp_net`/`edp_net`; the world all_reduce barrier keeps its
  nominal cost and is not charged.

### 5.3 Pump and completion formulas

Local entries (`_pump_local_entry`, the single change site):

```
launch_t = max(issue_t, rank_tail(stream), nic_tail[rank], tor_tail[node]*)
end_t    = launch_t + cost          # cost unchanged, from compute_net_op_time
nic_tail[rank] = end_t;  (tor_tail[node] = end_t when ToR is active)
```

Rendezvous entries (`_pump_rendezvous_entry`, `merge_lanes=False`): each
waiter's `ready_t = max(ready_t, nic_tail[waiter])`; on completion every
waiter's `nic_tail[waiter] = end_t`.

Async p2p (P2PBackend, both ends charged — decision 1): each arrival's
`ready_t = max(ready_t, nic_tail[rank], tor_tail[node]*)`;
`end_t = max(ready_t + cost)` over the two arrivals as today; at pair
finalization `nic_tail[send_rank] = nic_tail[recv_rank] = end_t`.
Send/recv ranks are already tracked in `AsyncP2PState`.

Blocking p2p (`_blocking_step_impl` + BarrierBackend, both ends —
decision 1): at arrival, `ready_t = max(ready_t, nic_tail[rank])`; when
the barrier fires, both waiters' `nic_tail` are set to the common
`end_t`, applied in the same drain that raises their lane clocks
(`SimuSystem.simu` pending-completions handling).

### 5.4 Relationship with the static corrections (decision 2: keep both)

The two layers approximate different things and deliberately coexist:

- **Static corrections** (`compute_net_op_time`): capacity identities and
  group-spread effects (per-GPU NIC bandwidth, `(k-1)/k` cross-node
  fractions, dp/edp multi-NIC spreading). They set the *service time* of
  an op that owns its NIC.
- **Fabric servers**: pure time-domain queuing between ops that share the
  same NIC/ToR. They never modify `cost`, only shift `launch_t`.

Documented as a known double-layer approximation; a config note marks
where a pure A/B (static corrections disabled) could be wired later.

### 5.5 merge_lanes semantics and the reserved ToR model

- NIC servers are per GPU, so `merge_lanes` needs no correction: each
  representative rank's NIC is fully its own.
- ToR servers (reserved) need node-share amplification: with
  `merge_lanes=True`, 1 of `num_per_node` ranks per node is simulated, so
  a ToR server would only see 1/num_per_node of the node's real traffic.
  The reserved model: `tor_node_share` (≈ `num_per_node` for merge_lanes,
  1 otherwise) amplifies each entry's ToR occupancy. The occupancy is
  `size_bytes / tor_capacity * node_share` (fallback
  `cost * node_share / num_per_node` when size/capacity are missing).
  ToR capacity defaults to `inter_node.gbps` (the node uplink), so with
  the default share the ToR occupancy equals the per-NIC service time
  and ToR never binds harder than the NIC for isomorphic node traffic;
  it only binds when `topology.tor_capacity_gbps` models oversubscription.
  (An earlier draft charged `cost * node_share` — that implicitly set the
  uplink equal to one NIC and was ~num_per_node x too pessimistic, as
  shown by the 16384-GPU A/B; the size-based formula above is the
  physically consistent one.) Caveat: `size_bytes` is the raw message
  size, so ops whose cost applies a cross-node traffic fraction
  (all2all `(k-1)/k`) are charged at ToR with their full size — a
  slight overcharge, accepted as a Preview limitation.
- PP p2p between stages maps both endpoints through the existing
  representative-rank mapping; `default_group` stays a pure barrier.

### 5.6 Trace presentation

Local comm spans start at `launch_t` (`_event_start_t`), so NIC
contention appears as spans sliding right (issue→launch gap). No new
event types in v1; an optional `nic_wait` marker may be added later.

## 6. Configuration

system.json (all optional; absent = current behavior):

```json
{
  "fabric_model": "nic",
  "topology": {
    "tor_capacity_gbps": 1600,
    "tor_node_share": "auto"
  }
}
```

- `fabric_model`: `"nic"` enables NIC servers; `"nic+tor"` additionally
  activates ToR servers (Preview); default/absent = off.
- `topology.tor_capacity_gbps`: ToR server capacity, default
  `inter_node.gbps`.
- `topology.tor_node_share`: `"auto"` (= `num_per_node` under merge_lanes)
  or an explicit number.

## 7. Validation

- **Regression**: with `fabric_model` absent, the three golden cases
  (llama merge_lanes, 8-rank, deepseekv2 ep4_pp2) stay event-for-event
  identical.
- **Synthetic serialization**: two concurrent `inter_node` ops on one
  rank → end_t == t0 + cost1 + cost2; blocking p2p with both ends busy →
  end_t reflects both NIC tails.
- **E2E A/B**: rerun the 16384-GPU moe-8T case with `fabric_model` off
  vs `"nic"`; report the end_t delta and the span shifts (expected:
  end_t only grows; dp/ep/pp overlap regions show serialization).
- **Tier A**: `net_info.json` diffs only where corrections were previously
  missing (TP/CP inter-node); everything else unchanged.

## 8. Phased Implementation

- **Phase A** (cost model): group→node helper in `core/utils.py` +
  generalized corrections in `compute_net_op_time` + Tier-A validation.
- **Phase B** (kernel): `NetworkFabric` in `base_struct.py`, `Com(net=)`
  plumbing through module/pipeline call sites, pump/completion hooks for
  local + rendezvous + async p2p + blocking p2p (both ends), config
  fields, ToR structure (pass-through default), full validation suite.
- **Phase C** (reserved): cross-rank skew via virtual waiters leveraging
  `enable_straggler_model` / `get_effective_straggler_sample_count`.

## 9. Impact Surface

- `simumax/core/base_struct.py`: `NetworkFabric`, pump/completion hooks,
  `Com.__init__` args, blocking/async p2p ready_t computation.
- `simumax/core/config.py`: `fabric_model`/`topology` fields + validation;
  Tier-A corrections in `compute_net_op_time`.
- `simumax/core/utils.py`: group→node mapping helper.
- `simumax/core/simu_runner.py`: construct `NetworkFabric` from
  system/strategy and attach to `SimuContext`.
- `dense_module.py` / `moe_module.py` / `pipeline_schedule.py`: pass
  `net=` to `Com` at creation sites (additive, defaulted).
- `docs/system.md` (+zh): new `fabric_model`/`topology` fields.

## 10. Acceptance Criteria

- Default-off bit-equivalence on all golden cases.
- Synthetic NIC serialization cases produce the analytic end_t values.
- 16384-GPU E2E A/B report delivered (end_t delta + contention hot spots).
- No change to `cost` formulas beyond Tier-A corrections.

## 11. Decisions Log

1. **Blocking p2p charges both ends**: sender and receiver NICs are both
   acquired and both updated at completion (§5.3).
2. **Double-layer approximation kept**: static corrections and fabric
   servers coexist with documented responsibilities (§5.4).
3. **ToR level reserved now**: multi-hop routes, ToR servers, and the
   `node_share` amplification field are built into the structures; the
   contention model itself ships later as Preview (§5.1, §5.5).
