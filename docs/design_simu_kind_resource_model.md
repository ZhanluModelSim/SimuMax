<p align="center">
  <a href="design_simu_kind_resource_model.md">English</a>|
  <a href="design_simu_kind_resource_model-zh.md">中文版本</a>
</p>

# Design Proposal: Explicit `simu_kind` and Resource/Engine Lane Model

- Status: **v1.0 (Phases 0-3 implemented)**
- Implementation: phase 0 `0e83888`, phase 1 `3229d92`, phase 2 `79e4dfd`, phase 3 `602c209`.
- Date: 2026-07-17
- Scope: `simumax/core` DES path (`PerfLLM.simulate()`), trace export, configs.
  The static analytical path (`analysis_*`) is explicitly out of scope for now.

## 1. Background and Problems

The simulator currently classifies operators in two disconnected places:

1. **Behavioral classification is implicit.** Whether an op is compute or
   communication is decided by which class is instantiated (`AtomModel` vs
   `Com` in `simumax/core/base_struct.py`). There is no explicit declaration;
   the DES kernel just polymorphically calls `step()/bwd()`.
2. **Display classification is string guessing.** The trace exporter
   (`simumax/core/generate_tracing.py:192-206`) assigns `cat`/`tid`/lane by
   matching the last `call_stk` segment against the hard-coded
   `COMM_PREFIXES` list (`generate_tracing.py:6-24`). A behaviorally correct
   new comm op whose class name misses the prefix table is silently
   misclassified as compute.
3. **The resource model is three hard-coded lanes.**
   `t["comp"]/t["comm"]/t["off"]` per rank (`base_struct.py:1347`), with
   `off` never used. One rank can only have one compute op in flight, so
   Cube∥Vector engine parallelism and compute-comm fused kernels cannot be
   expressed. Blocking comm ops clamp `t["comp"]` on completion
   (`base_struct.py:2215-2216`), hard-wiring rank-level mutual exclusion
   between compute and collectives; `overlap_grad_reduce`
   (`config.py:275`) is dead config as a result.
4. **Logging is a private text format.** Seven write sites each do
   open/write/close per line (`base_struct.py:85,177,1988,2032,2050,2149,
   2167`); lines that fail the parser regex (`generate_tracing.py:27-53`)
   are silently dropped.

## 2. Goals / Non-Goals

Goals:

1. Explicit `simu_kind` declaration; single source of truth for
   classification at the op definition site.
2. Generalized resource lanes: hardware engines (Cube/Vector) and comm
   links become registerable resources; multi-resource parallelism is
   expressible.
3. `fused` as a configurable, extensible kind, reserving what compute-comm
   fusion and DualPipeV-style F/B interleaving need.
4. Structured event stream replacing the text log.
5. Behavioral compatibility: existing configs (single-engine GPU machines)
   produce event-for-event equivalent traces.

Non-goals (this round):

- Implementing the DualPipeV schedule itself (only the foundation and the
  builder slot are delivered).
- Real-machine efficiency measurement for the Vector engine
  (`simu_tools/efficency_test` extension; separate requirement).
- Migrating the static analytical path (`perf_llm.py` `analysis_*`). For
  fused strategies it may diverge from `simulate()` results; this is an
  accepted trade-off.

## 3. What Does "fused" Fuse? (Taxonomy)

"Fusion" must be layered, otherwise the kind becomes a catch-all:

| Layer | What is fused | Examples | Classification |
|---|---|---|---|
| F1 kernel fusion | compute × compute (same engine) | fused swiglu, fused CE | **not** `fused`; just a compute op with a different cost |
| F2 compute-comm fusion | compute × comm (cross-resource) | AG+GEMM, RS+GEMM chunked pipelines | `simu_kind="fused"`, occupies `(cube, comm)` |
| F3 engine parallelism | compute × compute (cross-engine) | Cube runs GEMM while Vector runs norm/permute; in DualPipeV, the Cube segment of F(batch i) parallel to the Vector segment of B(batch j) | no special kind; emerges naturally from separate resource lanes |

**Definition: `fused` = one scheduling unit that occupies >= 2 hardware
resource classes simultaneously.** F3 needs no kind, only the resource
model (section 4.2). DualPipeV's F/B interleaving is a scheduler product;
its Cube∥Vector overlaps are F3, its compute-comm chunks are F2.

## 4. Design

### 4.1 Explicit `simu_kind` Declaration

```python
class LeafModel:
    simu_kind: ClassVar[str] = "compute"         # compute | comm | wait | scope | fused
    simu_resources: ClassVar[tuple] = ("comp",)  # occupied resource lanes
```

- `AtomModel` → `compute / ("comp",)`; `Com` subclasses → `comm /
  ("comm",)`; async p2p posts → `comm / ("pp_fwd",)` or `("pp_bwd",)`;
  `async_wait_recv` → `wait`.
- The exporter reads kind/resources from the event objects; the three
  hard-coded sites (`COMM_PREFIXES`, `_comm_lane`, scope prefix guessing)
  are deleted.
- `call_stk` is kept for human-readable naming and microbatch/chunk
  extraction (`simu_memory` depends on it), but no longer carries any
  classification role.

### 4.2 Resource/Engine Lane Model (the foundation)

- `SimuThread.t` (`base_struct.py:1347`) changes from the hard-coded
  `{comp, comm, off}` lanes to a lane dict initialized from a **resource
  registry**, built from `system.engines` plus built-in comm lanes.
- An op advances **only the lanes it declares**; the unconditional
  `t["comp"]` clamp in `Com._step/_bwd` is removed.
- Blocking semantics are unified as **post + wait**: the existing async p2p
  post/wait machinery (`base_struct.py:2419-2617`) becomes the only comm
  semantics; `blocking = post + immediately-following wait`. This also
  gives `overlap_grad_reduce` an expression path (post without an immediate
  wait).
- Cross-resource dependencies (a Vector op consuming a Cube op's output,
  F→B dependencies of one microbatch) use notify/wait token pairs, reusing
  the `BarrierBackend` infrastructure.
- Scheduling semantics (`cur_time` = min over active lanes) are unchanged;
  only the lane set grows from 3 to N.
- **Default configs (GPU, single engine) get the resource set
  `{comp, comm, pp_fwd, pp_bwd}`, one-to-one with today, behavior
  unchanged.**

### 4.3 `fused` Kind and Pluggable Fusion Policies

```python
class FusedOp(LeafModel):
    simu_kind = "fused"
    simu_resources = ("cube", "comm")
    fusion_policy = ChunkedPipeline(chunks=4)
    # duration = max(sum(cube_chunks), sum(comm_chunks)) + pipeline fill/drain
    # both lanes advance independently on completion
```

- `fusion_policy` is a pluggable object; built-ins: `Serial` (equivalent to
  today), `MaxOverlap` (duration = max), `ChunkedPipeline(chunks=n)`. New
  policies register without kernel changes.
- Cost model: `SystemConfig` gains a `compute_fused_op_cost(op_desc,
  policy)` dispatch entry. F2 efficiency entries hang off system.json
  (reserved field; the policy's analytic upper-bound formula is the
  fallback when no measured data exists).
- Strategy-side switch, e.g. `"fused_ops": [{"pattern": "tp_ag_gemm",
  "policy": "chunked_pipeline", "chunks": 4}]`. Unconfigured means current
  serial modeling.

### 4.4 DualPipeV Provisions (foundation only)

1. **Resource layer**: separate cube/vector lanes (4.2) — the GEMM of
   F(batch i) and the vector segment of B(batch j) overlap naturally.
2. **Scheduler layer**: `PpSchedule`'s construction logic is abstracted
   into a `ScheduleBuilder` interface (the existing 1F1B and interleaved
   paths are folded in as two builders); `DualPipeVBuilder` is registered
   later into the same slot. Jobs remain sequential queues; interleaving is
   expanded at build time, so the DES kernel needs **zero changes**.
3. **Dependency layer**: cross-chunk/cross-engine F→B dependencies of one
   microbatch use the notify/wait tokens from 4.2 (interface reserved,
   filled in when DualPipeV lands).

Note: two microbatches' activations coexisting under DualPipeV needs no new
mechanism — memory cache tokens are already pooled per microbatch via
`cache_token_scope`.

### 4.5 Structured Event Stream

- The seven text write sites become appends of **event objects** (dataclass:
  rank, kind, resources, name, call_stk, ts, dur, gid, post_ts, ...) to
  `ctx.event_sink` (in-memory list by default).
- `process_log_file` consumes the event stream directly; the regex parser is
  deleted. The "silently dropped on format mismatch" failure mode
  disappears entirely, along with per-line fopen/fclose.
- A `log.log` text rendering may be kept as a debug artifact, formatted
  one-way from the event stream; it is no longer an interchange format.

### 4.6 Trace Export Changes

- cat/tid/lane all come from event attributes.
- A fused event renders as **correlated slices on multiple lanes** (one per
  occupied resource, sharing a correlation id) — the correct visual shape
  of compute-comm fusion in Perfetto.
- The O(k^2) scope-detection scan (`generate_tracing.py:320-357`) is
  rewritten as a single stack-based pass while we are touching the file.

### 4.7 Fused-Op Memory Accounting (decided, see section 9)

The current tracker only emits counter events at `FwdQue/BwdStk` boundaries
(`simu_memory.py:175-187`): a jump to peak at `phase_start`, alloc/free at
`phase_end`. A chunked fused op ramps memory in the middle of the op
(inbound chunks arrive while compute consumes earlier ones), which boundary
events cannot express.

Decided model:

- **Default (closed-form steady state)**: at op start, book
  `peak = input activations + output + 2 x chunk staging (double buffer)`.
  No tracker changes, and the fusion memory win over unfused
  (`full gathered weight - 2 chunks`) is preserved.
- **Reserved switch (faithful ramp)**: a config flag enables per-chunk
  alloc/consume events for a true ramp curve. Implementation deferred;
  interface reserved in the event schema.

## 5. Configuration Examples

system.json (Ascend-style engine declaration; absent = single engine,
current behavior):

```json
{ "engines": { "cube": { "peak_tflops": 320 }, "vector": { "peak_tflops": 80 } } }
```

strategy.json (all optional; defaults reproduce current behavior):

```json
{
  "compute_engine_map": { "gemm": "cube", "elementwise": "vector" },
  "fused_ops": [{ "pattern": "tp_ag_gemm", "policy": "chunked_pipeline", "chunks": 4 }],
  "fused_mem_mode": "steady_state"
}
```

## 6. Phased Implementation

- **Phase 0 — structured event stream**: 7 write sites + event sink +
  exporter consumer; trace diff-validated event by event.
- **Phase 1 — kind declaration and registry**: `LeafModel` class
  attributes, 12 `Com` subclasses annotated, exporter switched,
  `COMM_PREFIXES` deleted; trace diff-validated event by event.
- **Phase 2 — resource lanes + post/wait unification**: lane dict,
  blocking-semantics rework, `Com` clamp removal. Default single-engine
  regression; async PP cases re-validated. **Trace shape intentionally
  changes here** (faithful post/wait double events, see section 9.3); this
  phase's output becomes the new baseline.
- **Phase 3 — fused extension points**: `FusedOp`, fusion policy registry,
  cost-model dispatch entry, `ScheduleBuilder` interface (DualPipeV slot).
  Validated with a synthetic AG+GEMM case showing both lanes advancing in
  parallel.

Each phase is independently mergeable. Phase 0/1 are pure refactors; Phase
2 carries the semantic risk (clamp removal changes existing overlap
behavior) and gets the golden regression focus.

## 7. Impact Surface

- `simumax/core/base_struct.py`: class attributes, lane dict, post/wait
  unification, write-site rework (core change area).
- `simumax/core/generate_tracing.py`: prefix classification deleted, event
  stream consumer, scope detection rewrite, fused multi-lane rendering.
- `simumax/core/simu_runner.py`: resource registry init, sink wiring.
- `simumax/core/config.py`: `engines` (system), `compute_engine_map` /
  `fused_ops` / `fused_mem_mode` (strategy) fields and validation.
- `simumax/core/transformer/pipeline_schedule.py`: builder interface
  (Phase 3).
- `docs/`: `strategy.md` / `system.md` and their `-zh` mirrors updated when
  the config fields land.

## 8. Acceptance Criteria

- Phase 0/1: `examples/simulator_trace_snapshot.py` and
  `examples/perf_deepseekv2_layer4_ep4_pp2.py` traces event-for-event
  equivalent to pre-change output (event ids may reorder).
- Phase 2: default-config golden regression green; `--no-merge-lanes`
  8-rank case timeline does not regress.
- Phase 3: in the synthetic fused case, the fused op shows correlated
  slices on the cube and comm lanes, and `end_t` matches the fusion
  policy's analytic value.

## 9. Decisions Log (resolved open questions)

1. **Vector engine cost source**: use a peak-scaled analytical estimate as
   the placeholder (system.json `engines.vector` carries only scalar peaks,
   no measured efficiency table). A measurement workflow can be added later
   without interface changes.
2. **Fused-op memory accounting**: default is the closed-form steady-state
   booking described in 4.7; the faithful per-chunk ramp mode is reserved
   behind the `fused_mem_mode` config switch.
3. **post/wait trace shape**: accepted the faithful double-event form
   (post event + wait event). Phase 2 traces intentionally differ from the
   old shape; golden equivalence covers Phase 0/1 only.
