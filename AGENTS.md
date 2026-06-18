# AGENTS.md

## Build & Verify

```bash
# Install
pip install -r requirements.txt
pip install -v -e .

# Lint (only checks simumax/, uses custom pylintrc)
bash tools/lint/pylint.sh

# Compile-check all Python sources
python -m compileall -q simumax app tools examples simu_tools/efficency_test simu_tools/megatron_scripts/*.py

# Smoke test (single example)
PYTHONPATH=. python examples/perf_llama3_8b_tp1_pp2.py

# Run all examples
cd examples && bash run_all.sh
```

Run order: `lint` then `compileall` then smoke test. There is no CI, no pre-commit, no pyproject.toml, and no automated test suite despite pytest being in requirements.txt.

## Architecture

- **`simumax/core/perf_llm.py`** (~3700 lines) — central modeling engine. `PerfLLM` is the main public class.
- **`simumax/core/config.py`** (~1200 lines) — `ModelConfig`, `StrategyConfig`, `SystemConfig` dataclasses. All configs load from JSON via `*.init_from_config_file()`.
- **`simumax/core/transformer/`** — per-op modeling: dense modules, MoE modules, pipeline schedules, simulated ops.
- **`simumax/tuning/strategy_searcher.py`** — automated search over micro-batch and parallel strategy settings.
- **`simumax/utils.py`** — config lookup helpers (`get_simu_model_config`, `get_simu_strategy_config`, `get_simu_system_config`). Includes a Windows symlink-text workaround for git checkouts.
- **`configs/`** — JSON configs split into `models/`, `system/`, `strategy/`. New configs go here.
- **`examples/`** — standalone runnable scripts. Each wires configs into `PerfLLM` and calls `run_estimate()` + `analysis()`.
- **`app/`** — Streamlit UI (`streamlit run streamlit_app.py`). Sets `SIMU_CHECK=1` at import time.
- **`simu_tools/`** — auxiliary: `efficency_test/` for operator measurement, `megatron_scripts/` for real benchmark runs.
- **`tools/b200/`** — B200-specific chart plotting and real-batch/pipeline run scripts.

## Key Conventions

- `setup.py` uses `distutils` and only packages `simumax`. The `app/`, `tools/`, `simu_tools/`, and `examples/` directories are not installed as packages.
- Max line length is **100 characters** (enforced by pylintrc).
- Pylint only lints `simumax/`. It ignores `docs/`, `tools/`, `examples/`, `benchmark/`, `ci/`.
- All Linear layers force gradient accumulation fusion (hardcoded behavior).
- Simulator artifacts (`tmp/`, `tmp_check/`, tracing JSONs, memory snapshots) are gitignored. The `SIMUMAX_TMP_PATH` env var overrides the output directory.

## Environment Variables

| Variable | Effect |
|---|---|
| `SIMU_CHECK=1` | Enables simulator verification mode; outputs to `tmp_check/` |
| `SIMU_DEBUG=1` | Enables debug logging |
| `ENABLE_SIMU_GRAPH=1` | Enables computation graph capture |
| `SIMUMAX_TMP_PATH` | Overrides default temp output directory |
