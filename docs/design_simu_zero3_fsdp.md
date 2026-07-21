<p align="center">
  <a href="design_simu_zero3_fsdp.md">English</a>|
  <a href="design_simu_zero3_fsdp-zh.md">中文版本</a>
</p>

# Design Proposal: ZeRO-3 / FSDP Modeling

- Status: **Draft v0.1** (design agreed in discussion, not yet implemented)
- Date: 2026-07-17
- Scope: ZeRO-3 (param sharding), per-layer and per-model FSDP communication,
  overlap modeling. Builds on the ZeRO-1 infrastructure already in the repo.

## 1. Background

ZeRO-1 (optimizer-state sharding) is fully wired: memory sharded
(`state_bytes /= group`), optimizer time scales, and the DES
`OptimizerSimulator` emits a monolithic RS-grad → optim-step → AG-param
tail block per iteration. `zero_state=2/3` are declared but warn-ignored
(`config.py:810`); their memory sharding branches exist in the leaves
(`grad_bytes /= group` for ≥2, `weight_bytes /= group` for ≥3) but
the comm sequence and per-layer path are wrong or absent. FSDP's
per-layer all-gather of params before forward and reduce-scatter of
grads after backward is entirely missing from the layer compute path.
`overlap_grad_reduce` is dead config; the Phase-2 post/wait machinery
is built but not wired to FSDP.

## 2. Goals / Non-goals

Goals:

1. `zero_state=3` activates FSDP (param sharding); lift the warn-ignore.
2. A new `fsdp_mode` field selects the communication pattern:
   - `"model-wise"`: unshard (all-gather all params) at step start,
     reshard (reduce-scatter all grads) at step end; minimal overlap.
   - `"layer-wise"`: per-LLMBlock unshard/reshard with adjacent-layer
     overlap; exposed time when comm exceeds the overlap window.
3. Both the fast analytical path and the DES `simulate()` path are
   supported — the user chooses which; `simulate()` gives precise
   overlap from the post/wait scheduling.
4. MoE blocks get the same granularity (layer-wise = per-block,
   model-wise = per-model), with dense params on the dp_cp group and
   expert params on the edp group.
5. Backward compatibility: `zero_state ≤ 1` unchanged; `fsdp_mode`
   absent → default model-wise for zero_state=3 (safest baseline).

Non-goals: flat-vs-per-tensor FSDP param layout distinction (no
performance-modeling difference); ZeRO-2 as a separate stage (it
collapses into zero_state=3 with the same comm recipe, just different
shard sizes); real FSDP checkpointing.

## 3. Configuration

```json
{
    "zero_state": 3,
    "fsdp_mode": "layer-wise"
}
```

- `zero_state`: 0/1/2/3 (existing field). 3 activates FSDP. The
  warn-ignore at `config.py:810` is lifted for value 3. (Value 2 is
  semantically a subset and may be lifted later if needed; for now it
  still warns.)
- `fsdp_mode: str = "model-wise"` — new StrategyConfig field; valid
  values `{"model-wise", "layer-wise"}`. Only meaningful when
  `zero_state >= 3`; validated and warned if set with a lower
  zero_state.

## 4. Model-wise FSDP (`fsdp_mode = "model-wise"`)

The simplest mode — structurally closest to today's tail block, just
repositioned and size-corrected.

### 4.1 Analytical

- `_compute_dp_time`: AG size = sharded param bytes (`weight_bytes`
  after the leaf `>=3` sharding, summed per model chunk); RS size =
  sharded grad bytes. Bucketization may stay (one big AG/RS bucketized)
  or be dropped (one unbucketized call per dense/MoE family). v1: keep
  bucketization for consistency; just fix the AG size derivation.
- `dp_comm_exposed_time = dp_comm_time` (no overlap — by design).
- `_compute_optim_time` already works (consumes sharded state_bytes).

### 4.2 DES

`OptimizerSimulator` tail block becomes:

```
AG(dense params, dp_cp_group) → AG(moe params, edp_group)
  → [PP schedule fwd/bwd runs with full params]
  → RS(dense grads, dp_cp_group) → RS(moe grads, edp_group)
  → optim_step
```

The AG is prepended before the PP schedule job; RS and optim_step
appended after. The world all_reduce barrier stays (or is absorbed into
the RS if the group already covers all ranks). `run_simulation` wires
the prepended AG into the job list before `PpSchedule.prefill_batch`.

### 4.3 Memory

Peak = static (sharded params + sharded grads + sharded states) + AG
buffer (full unsharded params) + activations. Since static + AG buffer =
full params + sharded grads + sharded states, the peak is approximately
the same as zero_state=1 (params always full). Model-wise FSDP does not
save peak memory vs ZeRO-1; its savings are in the optimizer-state
sharding already captured by ZeRO-1.

## 5. Layer-wise FSDP (`fsdp_mode = "layer-wise"`)

Per-LLMBlock unshard/reshard with overlap.

### 5.1 Analytical (fast path)

Per-block costs:

- `AG_block` = sharded param bytes of this block / (dp group bandwidth)
- `RS_block` = sharded grad bytes of this block / (dp group bandwidth)
- `compute_block_fwd` = existing fwd compute time of this block
- `compute_block_bwd` = existing bwd compute time

Overlap estimate (forward):

```
fwd_exposed = Σ_blocks max(0, AG_block - compute_{prev_block}_fwd)
bwd_exposed = Σ_blocks max(0, RS_block - compute_{next_block}_bwd)
dp_comm_exposed_time = fwd_exposed + bwd_exposed
```

The first block has no previous to overlap with → AG is fully exposed.
The formula is a conservative upper bound on the non-overlappable
fraction; the DES path (below) gives the precise value.

### 5.2 DES (precise path)

Per-LLMBlock, the job list interleaves AG/RS with compute via post/wait
(Phase-2 machinery):

```
post AG(params for block N+1) → compute block N fwd
  → wait AG → compute block N+1 fwd → ...
  → compute block N bwd → post RS(grads for block N)
  → compute block N+1 bwd → wait RS → ...
```

- `all_gather` / `reduce_scatter` ops are created in the block's
  `prefill_fwd` / `prefill_bwd` (in `language_model.py` LLMBlock, not
  in `OptimizerSimulator`) over the dp_cp group (dense) and edp group
  (MoE experts).
- Post/wait uses the blocking-collective post/wait path: `Com._step`
  issues the entry (post marker) and yields; the wait is a separate op
  that blocks on the comm_entry completion. The post/wait semantics from
  Phase 2 of the kind-resource design apply.
- `OptimizerSimulator` tail shrinks to `optim_step` only (RS happens
  per-layer during bwd; AG per-layer during fwd).

### 5.3 Memory

Peak = static (sharded) + one block's unsharded param buffer +
activations. Much lower than model-wise (only one block's params
gathered at a time, not the whole model).

### 5.4 MoE

In layer-wise mode, each MoE LLMBlock gets:

- Fwd: `AG(dense params, dp_cp_group)` + `AG(expert params, edp_group)`
  (two AGs, can post both, wait both)
- Bwd: `RS(dense grads, dp_cp_group)` + `RS(expert grads, edp_group)`

In model-wise mode: one big `AG(all dense params, dp_cp)` + one big
`AG(all expert params, edp)` at step start; `RS` of all grads at step
end.

## 6. Phased Implementation

- **Phase 1 — model-wise FSDP**: lift warn-ignore for zero_state=3;
  `fsdp_mode` field + validation; fix `_compute_dp_time` AG size;
  `OptimizerSimulator` reposition (AG before PP, RS after, optim_step
  tail); memory peak with AG buffer; docs. Validation: zero_state≤1
  golden unchanged; model-wise E2E runs.
- **Phase 2 — layer-wise FSDP**: per-LLMBlock AG/RS injection in
  `language_model.py`; post/wait wiring in DES; analytical overlap
  estimate in `_compute_dp_time`; `OptimizerSimulator` tail shrink;
  memory peak with per-block buffer; MoE dual-group AG/RS; docs.
  Validation: layer-wise E2E with overlap delta; golden zero_state≤1
  unchanged.

## 7. Validation

- `zero_state ≤ 1`: golden traces event-for-event equal (fsdp_mode
  absent).
- Model-wise: analysis runs; DES runs; AG buffer in memory; OptimizerSimulator
  tail structure correct.
- Layer-wise: DES produces per-block AG/RS events in the trace; overlap
  window visible (AG span overlaps with compute span on the trace);
  `dp_comm_exposed_time` from the analytical estimate vs DES end_t
  delta reported.
- E2E: moe-8T 16384 with zero_state=3 model-wise vs layer-wise →
  memory peak difference (one-block vs full-model AG buffer) and
  end_t difference (overlap savings).

## 8. Decisions Log

1. **FSDP-1 and FSDP-2 unified**: no performance-modeling difference
   (flat vs per-tensor params → same comm volume); `zero_state=3` is
   the single value. A new `fsdp_mode` field (layer-wise / model-wise)
   selects the communication pattern.
2. **Granularity**: layer-wise = per-LLMBlock; model-wise = per-model.
3. **MoE**: same granularity rules — layer-wise MoE blocks get per-block
   AG/RS (dense on dp_cp group, expert on edp group); model-wise gets
   one big AG/RS per group.
4. **Both analytical and DES supported**: the user chooses
   `analysis()` (fast, analytical overlap estimate) or `simulate()`
   (precise, DES post/wait overlap).
