<p align="center">
  <a href="design_simu_cost_model_tunability.md">English</a>|
  <a href="design_simu_cost_model_tunability-zh.md">中文版本</a>
</p>

# Design Proposal: Workload Ingestion & Per-Operator Cost-Model Tunability

- Status: **Draft v0.1** (design agreed in discussion, not yet implemented)
- Date: 2026-07-17
- Scope: the analytical cost path (`compute_op_accuracy_time`,
  `_comp_cost_info_impl`), module cost specs, model JSON composition,
  measurement feedback loop. Builds on (does not alter) the DES work of
  the two earlier design docs.

## 1. Background and Problems

Workload ingestion today is code-driven, in three layers:

1. **Structure**: model JSON carries hyper-parameters only; layer
   composition is hard-coded in the Python module library
   (`dense_module.py` / `moe_module.py`).
2. **Formulas**: every module class hand-derives its FLOPs and accessed
   bytes for fwd / bwd_grad_act / bwd_grad_w, and its comm sizes in
   `prefill`.
3. **Costing**: the fake-forward fills `_compute_info` per module, then
   `_comp_cost_info_impl` (`base_struct.py:846`) converts to time via
   `compute_op_accuracy_time(op_name, flops, shape_desc)`
   (`config.py:906`): `time = flops / (tflops * eff)`.

Efficiency lookup has only two dimensions:

- `op_name` — a coarse op type hard-coded at each call site (7 types:
  `matmul`, `fp8_matmul`, `group_matmul`, `fp8_group_matmul`, `sdp_fwd`,
  `sdp_bwd`, `default`), shared by every module of that kind.
- `shape_desc` — a measured-shape key
  (`b=, m=, k=, n=, layout=, accumulate=, out_dtype=`) that exists **only
  for LinearBase GEMMs**; sdp / elementwise ops pass an empty string and
  can only use the op-type default.

Limitations:

- **No per-operator tunability**: all Linear layers share `matmul`; a
  measured shape entry (`accurate_efficient_factor`) carries no operator
  identity, so same-shape LinearCol and LinearRow share one record. You
  cannot tune "only ParallelCE" or "only layer_3's MLP".
- **Ingestion requires Python**: a new architecture (e.g. MLA) means new
  module classes with three hand-derived FLOPs formulas each; formulas
  are entangled with topology code; there is no declarative composition
  entry point.

## 2. Goals / Non-Goals

Goals (numbered per the decisions in section 12):

1. Per-operator efficiency tuning at **two key granularities**: class
   name and module path.
2. `operator_efficiency` lives in system.json (machine property);
   strategy/API overrides for temporary what-if adjustments — with user
   docs refreshed.
3. **Declarative recipe**: model JSON may declare layer composition from
   registered templates; new-model ingestion becomes JSON (+ optional
   cost-spec registration) instead of new Python classes.
4. Non-GEMM ops (sdp, elementwise) gain a shape dimension for
   efficiency lookup.
5. Full backward compatibility: no keys configured ⇒ bit-identical
   results.

Non-goals:

- Graph/trace-based workload import (ONNX in); the static analytic
  philosophy stays.
- Changing any cost formula itself (flops/bytes derivations move into
  specs, unchanged in content).
- Real-machine measurement workflow rework (only its output regrouping).

## 3. cost_op_key and the Lookup Chain

Every module instance gets:

- **class_key**: defaults to the class name (`"LinearCol"`,
  `"ParallelCE"`, `"ExpertMLP"`), overridable per instance via a new
  `cost_op_key` attribute.
- **path_key**: dot-joined module attribute path from the model root,
  e.g. `layer_3.mlp`, `layer_3.attention.q_proj`. Computed during the
  fake-forward from the existing parent/child tree (the same tree that
  already renders `peak_path`); the model-root prefix is dropped for
  brevity.

Efficiency lookup order at `compute_op_accuracy_time` time (first hit
wins, miss falls through):

| Level | Key | Source |
|---|---|---|
| 1 | `(path_key, shape_desc)` | overrides chain |
| 2 | `path_key` | overrides chain |
| 3 | `(class_key, shape_desc)` | overrides chain |
| 4 | `class_key` | overrides chain |
| 5 | `(op_name, shape_desc)` | existing `accurate_efficient_factor` |
| 6 | `op_name` | existing `efficient_factor` |
| 7 | `default` op | existing fallback |

The "overrides chain" itself is: API overrides > strategy
`efficiency_overrides` > system `operator_efficiency`. Every lookup
records the winning level and source in `hit_efficiency`; misses are
recorded in `miss_efficiency` **grouped by class_key/path_key** so the
measurement loop knows which operator to benchmark.

## 4. Configuration

system.json (machine property, primary):

```json
"operator_efficiency": {
  "ParallelCE": 0.52,
  "LinearCol": {"default": 0.60, "shapes": {"b=1, m=4096, k=8192, n=10240, layout=TN, accumulate=False, out_dtype=bf16": 0.66}},
  "layer_3.mlp": {"default": 0.55}
}
```

strategy.json (temporary what-if, wins over system):

```json
"efficiency_overrides": {
  "layer_3.mlp": 0.48
}
```

API (wins over both, for search/tuning loops):

```python
perf.configure(..., efficiency_overrides={"ParallelCE": 0.50})
```

Key grammar: a bare key (class or path) maps to either a scalar
(default eff for that key) or a dict with `default` + `shapes`.
Validation: keys must resolve to a known class_key or a valid path
pattern; unknown keys raise at configure time (fail fast, not silent
no-op).

## 5. Shape Dimensions for Non-GEMM Ops

- `sdp_fwd` / `sdp_bwd`: attention modules (CoreAttention, MLA variants)
  build a shape_desc like
  `b=, s_q=, s_kv=, h_q=, h_kv=, d=, causal=` so flash-attention
  efficiency can be measured and keyed per shape (seq_len-sensitive).
- Elementwise / norm ops (currently `default` op): a light
  `b=, s=, h=` descriptor; class_key-level tuning already covers most of
  the need, shapes are optional refinement.
- `accurate_efficient_factor[(op_name, shape_desc)]` keeps working
  unchanged at level 5.

## 6. Cost-Spec Registry and the Declarative Recipe

**Registry** (`simumax/core/cost_specs.py`): separates *what an op
costs* from *where it sits*. A `CostSpec` entry binds:

- `flops(stage)` / `bytes(stage)` derivations for fwd / bwd_grad_act /
  bwd_grad_w (the existing formulas, relocated verbatim),
- the `op_name` binding (e.g. `matmul`),
- the shape_desc builder,
- the default class_key.

Existing module classes register their specs with zero behavior change;
new ops register new specs instead of subclassing just to tweak cost
math.

**Recipe** (model JSON, optional): layer composition declared from
registered templates:

```json
"recipe": {
  "stem": {"embedding": "Embedding", "head": "ParallelCE"},
  "blocks": [
    {"template": "DenseLLMBlock", "count": 3},
    {"template": "MoELLMBlock", "count": 58}
  ]
}
```

`ModelConfig.recipe` + a recipe-driven `LLMModel` build path; absent a
recipe, today's composition logic runs unchanged. Templates map to the
existing module classes — the recipe generalizes the current
`dense_layers` + `layer_num` pattern (which remains the implicit default
recipe). Custom attention/MLP variants beyond the registered templates
still require Python modules, but the common "N blocks + dense prefix"
family becomes JSON-only.

## 7. Measurement Feedback Loop

- `miss_efficiency` groups by class_key/path_key (§3) — the report
  directly names the operator to measure.
- `simu_tools/efficency_test` output regrouped by cost key, emitting
  paste-ready `operator_efficiency` fragments (per-shape entries for
  GEMM/sdp, key-level defaults otherwise).
- Docs will show the closed loop: run → inspect `miss_efficiency` →
  measure → paste into system.json.

## 8. Documentation Refresh (decision 2)

- `docs/system.md` (+zh): `operator_efficiency` schema, lookup chain,
  measurement loop.
- `docs/strategy.md` (+zh): `efficiency_overrides` field and precedence.
- `docs/model.md` (+zh): `recipe` section and template table.
- `docs/tutorial.md` (+zh if present): one worked example of tuning a
  single operator.
- `AGENTS.md`: registry / key / recipe summary at the end.

## 9. Phased Implementation

- **Phase 1 — keys & overrides**: `cost_op_key`/path computation, lookup
  chain in `compute_op_accuracy_time`, `operator_efficiency` (system) +
  `efficiency_overrides` (strategy + API), key-grouped miss/hit
  records, docs for this phase. Validation: no-key golden equivalence;
  synthetic per-key override cases.
- **Phase 2 — non-GEMM shapes**: sdp + elementwise shape_desc builders,
  efficiency_test regrouping, docs. Validation: synthetic sdp per-shape
  entries win over defaults.
- **Phase 3 — registry & recipe**: `cost_specs.py`, `ModelConfig.recipe`
  + recipe build path, template table for dense/MoE families, docs.
  Validation: recipe-built model == class-built model on golden cases
  (bit-identical traces/analysis); a JSON-only variant of an existing
  model produced as the demo.

Each phase independently mergeable; commits per phase.

## 10. Impact Surface

- `simumax/core/config.py`: lookup chain, `operator_efficiency` /
  `efficiency_overrides` / `recipe` fields + validation.
- `simumax/core/base_struct.py`: key/path computation on modules,
  key-aware lookup plumbing in `_comp_cost_info_impl`.
- `simumax/core/cost_specs.py` (new, Phase 3).
- `simumax/core/transformer/dense_module.py` / `moe_module.py` /
  `language_model.py`: spec registration, shape_desc builders, recipe
  build path.
- `simumax/core/perf_llm.py`: API overrides entry.
- `simu_tools/efficency_test/`: output regrouping (Phase 2).
- docs as listed in §8.

## 11. Acceptance Criteria

- Default (no keys/recipe) results bit-identical on golden cases.
- A path-key override changes exactly the targeted module's time in
  outputs; precedence API > strategy > system demonstrated.
- sdp per-shape efficiency entry wins over the op default in a
  synthetic case.
- A recipe-built existing model reproduces the class-built results
  exactly.

## 12. Decisions Log

1. **Key granularity**: both class-name and path level (the user wants
   the clearer attribution of paths alongside class-wide tuning).
2. **Placement**: `operator_efficiency` in system.json as the machine
   property; strategy/API overrides for temporary adjustment; usage
   docs must be refreshed with the change.
3. **Ingestion scope**: full declarative recipe (not just the registry).
4. **Non-GEMM shape dimension**: required — sdp and elementwise ops get
   shape_desc builders.
