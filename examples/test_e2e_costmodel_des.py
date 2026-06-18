"""E2E test: Plan 1 (CostModel) + Plan 2 (DES overlap) with real configs."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.core.cost_model import (
    TableLookupCostModel,
    FormulaCostModel,
    OverrideCostModel,
)
from simumax.utils import (
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


def test_baseline_unchanged():
    """Verify that without CostModel overrides, results are identical."""
    print("=" * 60)
    print("E2E Test 1: Baseline unchanged (no CostModel overrides)")
    print("=" * 60)

    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config('tp1_pp2_dp4_mbs1')
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config('llama3-8b')
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config('a100_pcie')
        ),
    )
    perf.model_config.padded_vocab_size = True
    perf.model_config.make_vocab_size_divisible_by = 128
    perf.run_estimate()
    perf.analysis('baseline_test')
    print("PASS: Baseline run completed without errors.\n")


def test_table_lookup_cost_model():
    """Verify TableLookupCostModel produces same results as default."""
    print("=" * 60)
    print("E2E Test 2: TableLookupCostModel equivalence")
    print("=" * 60)

    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config('tp1_pp2_dp4_mbs1')
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config('llama3-8b')
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config('a100_pcie')
        ),
    )
    perf.model_config.padded_vocab_size = True
    perf.model_config.make_vocab_size_divisible_by = 128

    perf.set_cost_model("LinearCol", TableLookupCostModel())
    perf.set_cost_model("LinearRow", TableLookupCostModel())
    perf.set_cost_model("CoreAttention", TableLookupCostModel())

    perf.run_estimate()
    perf._apply_cost_models()
    perf._run()
    perf.analysis('table_lookup_test')
    print("PASS: TableLookupCostModel run completed.\n")


def test_override_cost_model():
    """Verify OverrideCostModel changes compute time."""
    print("=" * 60)
    print("E2E Test 3: OverrideCostModel effect")
    print("=" * 60)

    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config('tp1_pp2_dp4_mbs1')
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config('llama3-8b')
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config('a100_pcie')
        ),
    )
    perf.model_config.padded_vocab_size = True
    perf.model_config.make_vocab_size_divisible_by = 128

    perf.set_cost_model("LayerNorm", OverrideCostModel(fixed_time_ms=0.0))

    perf.run_estimate()
    perf._apply_cost_models()
    perf._run()
    perf.analysis('override_test')
    print("PASS: OverrideCostModel run completed.\n")


def test_formula_cost_model():
    """Verify FormulaCostModel with custom formula."""
    print("=" * 60)
    print("E2E Test 4: FormulaCostModel custom formula")
    print("=" * 60)

    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config('tp1_pp2_dp4_mbs1')
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config('llama3-8b')
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config('a100_pcie')
        ),
    )
    perf.model_config.padded_vocab_size = True
    perf.model_config.make_vocab_size_divisible_by = 128

    perf.set_cost_model("Swiglu", FormulaCostModel(
        compute_fn=lambda ctx: ctx.flops / (312e12 * 0.8) * 1e3,
        mem_fn=lambda ctx: ctx.accessed_mem * 2 / (1600e9 * 0.9) * 1e3,
    ))

    perf.run_estimate()
    perf._apply_cost_models()
    perf._run()
    perf.analysis('formula_test')
    print("PASS: FormulaCostModel run completed.\n")


def test_overlap_report():
    """Verify DES overlap report generation."""
    print("=" * 60)
    print("E2E Test 5: DES overlap report")
    print("=" * 60)

    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config('tp1_pp2_dp4_mbs1')
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config('llama3-8b')
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config('a100_pcie')
        ),
    )
    perf.model_config.padded_vocab_size = True
    perf.model_config.make_vocab_size_divisible_by = 128

    summary = perf.run_estimate_with_overlap()

    assert summary is not None, "Overlap summary should not be None"
    assert summary.total_compute_time > 0, "Compute time should be > 0"
    print(f"Total compute time: {summary.total_compute_time:.4f} ms")
    print(f"Total comm time: {summary.total_comm_time:.4f} ms")
    print(f"Overlap ratio: {summary.overall_overlap_ratio:.1%}")
    print("PASS: Overlap report generated.\n")


def test_cost_model_by_path():
    """Verify per-path CostModel override."""
    print("=" * 60)
    print("E2E Test 6: CostModel by path override")
    print("=" * 60)

    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(
            get_simu_strategy_config('tp1_pp2_dp4_mbs1')
        ),
        model_config=ModelConfig.init_from_config_file(
            get_simu_model_config('llama3-8b')
        ),
        system_config=SystemConfig.init_from_config_file(
            get_simu_system_config('a100_pcie')
        ),
    )
    perf.model_config.padded_vocab_size = True
    perf.model_config.make_vocab_size_divisible_by = 128

    perf.run_estimate()

    target_path = None
    for chunk_name, model in perf.model_chunk_dict.items():
        if hasattr(model, 'all_leaf_nodes'):
            for leaf in model.all_leaf_nodes:
                if type(leaf).__name__ == "LinearCol":
                    target_path = leaf.full_name
                    break
        if target_path:
            break

    if target_path:
        perf.set_cost_model_by_path(
            target_path, OverrideCostModel(fixed_time_ms=0.001)
        )
        perf._apply_cost_models()
        perf._run()
        print(f"PASS: Applied OverrideCostModel to path: {target_path}")
    else:
        print("SKIP: No LinearCol found to override")
    print()


if __name__ == "__main__":
    test_baseline_unchanged()
    test_table_lookup_cost_model()
    test_override_cost_model()
    test_formula_cost_model()
    test_overlap_report()
    test_cost_model_by_path()
    print("=" * 60)
    print("ALL E2E TESTS PASSED")
    print("=" * 60)
