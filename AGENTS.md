# AGENTS.md

Guidance for AI coding agents working on the SimuMax repository.

## Project Overview

SimuMax is a static analytical model and simulator for large-scale LLM
distributed training (Moore Threads / MT AI Team). It models compute,
communication, and memory of a training job **without launching real
training**, so users can estimate throughput (MFU), peak memory, pipeline
behavior, and traces before running real workloads.

Key characteristics:

- Pure Python package, no compiled components. Package name: `simumax`
  (`simumax/version.py`, currently `0.1.dev0`; public release train is v1.2,
  see `docs/release_v1.2.md`).
- Requires **Python >= 3.10** in practice: `simumax/core/perf_llm.py` uses
  `int | None` style annotations without `from __future__ import annotations`.
- Config-driven: every run is defined by a triple of JSON configs —
  `system` (machine capability, bandwidth, latency, operator efficiency),
  `strategy` (parallelism and runtime policy), and `model` (architecture).
- Feature coverage: dense and MoE models, TP / PP / EP / CP / SP / ZeRO-1 /
  recompute / MLA, Megatron-LM 0.14 selective recompute via
  `megatron_recompute` + `megatron_recompute_modules`, sync-VPP (Preview).
- Known modeling constraint (from README): all Linear models are forced to
  perform gradient accumulation fusion.

## Repository Layout

- `simumax/` — the only installable package.
  - `simumax/core/perf_llm.py` (~3700 lines) — main API. `PerfLLM` (subclass
    of `PerfBase`) is the central entry point: `configure()` →
    `run_estimate()` → `analysis()` / `analysis_mem()` / `analysis_cost()` →
    optional `simulate()`. Also hosts the strategy-search APIs
    (`search_max_micro_batch_size_fixed_gbs`,
    `search_best_parallel_strategy`, `search_best_selective_recompute`, …)
    with internal profile caching (`CachedChunkProfile` etc.).
  - `simumax/core/config.py` — `ModelConfig`, `StrategyConfig`,
    `SystemConfig` dataclasses (base class `Config`) with JSON loading,
    sanity checks, and strategy-name parsing (e.g. `tp1_pp2_dp4_mbs1`).
    Module level defines run-time env flags (see "Environment variables").
  - `simumax/core/base_struct.py` — modeling framework: `MetaModule` /
    `LeafModel` / `AtomModel` module tree, forward/backward queues, simulated
    comm ops (`all_gather`, `reduce_scatter`, `all2all`, `send`/`recv`,
    async/sync p2p variants), `SimuSystem` / `SimuContext` global simulation
    state, `SimuThread`.
  - `simumax/core/transformer/` — module library: `dense_module.py`
    (Embedding, LinearCol/LinearRow, LayerNorm, CoreAttention, MLA variants,
    MLP, Swiglu/Gelu, ParallelCE, FP8 quantizers), `moe_module.py`,
    `language_model.py` (`LLMModel`, `PeakPoint`), `pipeline_schedule.py`,
    `function.py`, `simu_ops.py`.
  - `simumax/core/model_struct.py` — info/result structs
    (`ModuleComputeInfo`, `ActivationInfo`, `ModuleMemoryInfo`,
    `ModuleCostInfo`, …).
  - `simumax/core/graph.py` — `SimuONNXGraphBuilder` and graphviz
    visualization of the modeled graph (enabled via `ENABLE_SIMU_GRAPH`).
  - `simumax/core/simu_runner.py`, `simu_memory.py`, `generate_tracing.py`,
    `trace_export.py`, `simu_artifacts.py` — simulator execution, memory
    timeline/snapshots, and Chrome-trace (`tracing_logs.json`) export behind
    `PerfLLM.simulate()`.
  - `simumax/core/simu_events.py` — structured DES event stream.
  - `simumax/core/fusion.py` — fusion policies for fused ops. The DES
    classifies ops via explicit `simu_kind` declarations (see
    `docs/design_simu_kind_resource_model.md`).
  - `simumax/core/tensor.py` — `FakeTensor` shape/dtype carrier.
  - `simumax/pp_simu/utils.py` — DualPipe-style overlap/duration analytic
    helpers and plotting.
  - `simumax/tuning/strategy_searcher.py` — strategy search helpers.
  - `simumax/testing/base_test_tool.py` — numeric comparators
    (`RelDiffComparator`, `ResultCheck`) for prediction-vs-golden checks.
  - `simumax/utils.py` — config registry: `get_simu_model_config`,
    `get_simu_strategy_config`, `get_simu_system_config`,
    `show_simu_models/strategy/system`, `create_default_strategy`.
- `configs/` — shipped JSON configs, resolved by name via the helpers above:
  - `configs/models/` — model architectures (llama2/3, deepseek v2/v3,
    mixtral, qwen3, kimi-1T, …).
  - `configs/strategy/` — parallelism strategies (tp/pp/ep/dp/mbs, recompute
    variants, sync-VPP example).
  - `configs/system/` — machine configs (`a100_pcie.json`,
    `b200_bf16_ceperm.json`) including per-shape `accurate_efficient_factor`
    matmul entries and fitted `networks` values.
- `examples/` — runnable `perf_*.py` examples, `simulator_trace_snapshot.py`,
  `search_strategy_llama3_8b.py`, `search/llm_search.py`, `run_all.sh`.
  See `examples/README.md` for trace/snapshot artifact fields.
- `app/` — Streamlit UI (`streamlit_app.py`, `install.sh`); run with
  `python -m streamlit run streamlit_app.py` from `app/`.
- `simu_tools/efficency_test/` — machine-measurement workflow that produces a
  SimuMax-ready `system.json` (GEMM/grouped-GEMM/FA/CE efficiency tests, NCCL
  fitting). Its `test_*.py` files are GPU measurement scripts, not unit tests.
- `simu_tools/megatron_scripts/` — Megatron-LM real-benchmark reference
  scripts (`run_llama3.sh`, `run_deepseekv2.sh`, patches) used for
  perf-vs-real validation.
- `tools/b200/` — retained public B200 workflow:
  `build_current_machine_system_config.py` (wraps the efficiency workflow),
  `run_megatron_perf_real_*.py`, `plot_release_charts.py`, `run_params/`.
- `tools/lint/` — `pylint.sh` + `pylintrc`.
- `docs/` — user documentation in English with `-zh` Chinese variants
  (`tutorial.md`, `system.md`, `model.md`, `strategy.md`, `FULL_RESULTS.md`,
  `release_v1.2.md`, `b200/`). Design proposals also live here, e.g.
  `design_simu_kind_resource_model.md` (+ `-zh` mirror) for the
  `simu_kind` / resource-lane / fused-op rework of the `simulate()` DES
  path (implemented), `design_simu_network_fabric.md` (+ `-zh` mirror)
  for NIC-level cross-node contention modeling (implemented), and
  `design_simu_cost_model_tunability.md` (+ `-zh` mirror) for planned
  per-operator cost-model tunability and declarative model recipes.

## Build / Install / Run

There is no `pyproject.toml`; packaging is `setup.py` (distutils) +
`requirements.txt`. `setup.py` declares only `packages=['simumax']` and no
`install_requires` — runtime deps come from `requirements.txt`, and editable
install is the expected mode.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -v -e .
```

Dependencies (from `requirements.txt`): numpy, pandas, sympy, tabulate,
matplotlib, graphviz, tqdm, streamlit, pytest.

Run an example (examples expect to run from the `examples/` directory and
resolve configs by name through `simumax/utils.py`, which locates `configs/`
relative to the installed package — this works because of the editable
install from the repo root):

```bash
cd examples
python perf_llama3_8b_tp1_pp2.py
```

Typical outputs written to the working directory: `compute_result.json`,
`mem_result.json`, `base_info.json`, `model_arch`, plus copies of the three
input configs. `simulate()` always writes `tracing_logs.json` (Chrome trace /
Perfetto format); memory artifacts (`simu_memory_result.json`,
`simu_memory_snapshot.json` schema `simumax_memory_snapshot_v1`,
`simu_memory_viz_snapshot.pickle`) are exported when `pp_size == 1` or
`pp_comm_async == false`.

## Testing and Validation

There is no unit-test suite in the repo (`pytest` is a dependency, but no
`test_` files exist outside the GPU measurement scripts in
`simu_tools/efficency_test/`). Typical local validation (from README):

```bash
python -m compileall -q simumax app tools examples simu_tools/efficency_test simu_tools/megatron_scripts/*.py
PYTHONPATH=. python examples/perf_llama3_8b_tp1_pp2.py
```

For numeric comparisons (e.g. validating a change against golden results),
use `simumax/testing/base_test_tool.py` (`ResultCheck` with relative
tolerance). Perf-vs-real accuracy is validated externally through
`simu_tools/megatron_scripts/` and the B200 materials in `docs/b200/`;
`docs/FULL_RESULTS.md` is the public benchmark summary.

Lint (only enforced on the `simumax/` package):

```bash
bash tools/lint/pylint.sh        # python3 -m pylint simumax/ --rcfile=tools/lint/pylintrc
```

The pylintrc ignores `docs/`, `tools/`, `examples/`; sets
`max-line-length=100`, 4-space indent; and disables many checkers
(`design`, `no-member`, `import-error`, `protected-access`, …).

## Code Style and Conventions

- Language: code, comments, and primary docs are English; several docs have
  `-zh.md` Chinese mirrors that should be kept in sync when the English
  version changes.
- Style: standard PEP 8-ish Python, 4-space indent, line length 100, dataclass
  configs, docstrings on public classes/methods. Match the surrounding file's
  conventions; several core files are large and dense — make minimal,
  targeted edits.
- Config-first workflow: prefer adding/copying a JSON config under
  `configs/models|strategy|system` over hard-coding new parameters; register
  nothing — files are discovered by walking those directories.
- Strategy configs are often mutated in example scripts after loading (e.g.
  `model_config.padded_vocab_size = True`,
  `model_config.make_vocab_size_divisible_by = 128` to align with Megatron
  benchmarks) — this is an accepted pattern.
- Run artifacts (`tmp/`, `tmp_check/`, `*_efficiency/`, profiler outputs,
  local Megatron checkouts) are gitignored; do not commit them.
- `simumax/utils.py` also looks up `develop/configs` for `version='dev'`;
  that directory is not shipped in the repo, so effectively only
  `version='release'` works out of the box.

## Environment Variables

Read in `simumax/core/config.py` (and `setup.py`):

- `SIMU_CHECK=1` — route run artifacts to `tmp_check/`.
- `SIMU_DEBUG=1` — debug output.
- `ENABLE_SIMU_GRAPH=1` — enable graph capture/visualization path.
- `SIMUMAX_TMP_PATH` — override the artifact directory entirely (default is a
  timestamped `tmp_YYYYMMDD_HHMMSS/`).
- `TAG_NAME` — overrides the package version at build time.

## Security and Operational Notes

- No secrets, credentials, or network services in the package; the repo
  contains no `.env` handling. `simu_tools/megatron_scripts/hostfile` is a
  user-edited list of benchmark host IPs — do not commit real internal IPs.
- The Streamlit app is a local UI only; do not expose it publicly without
  review.
- `simu_tools/efficency_test/` and `simu_tools/megatron_scripts/` execute
  real GPU/NCCL benchmarks and clone/run Megatron-LM locally (gitignored);
  they require actual multi-GPU clusters and are not runnable in a plain dev
  environment. Treat their shell scripts as potentially destructive to GPU
  state (`clear.sh`, `stop_all.sh` kill processes) — read before running.
- Benchmark scripts use mock data only; no datasets are downloaded by the
  package itself.
