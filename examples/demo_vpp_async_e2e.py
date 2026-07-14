"""E2E Demo: VPP interleaved schedule + async P2P + DES tracing.

Runs 3 scenarios on the same llama3-8b model:
  1. PP=2, VP=1 (non-VPP baseline, sync P2P)
  2. PP=2, VP=1 (async P2P)
  3. PP=2, VP=2 (VPP interleaved, sync P2P)

Each exports a Chrome Tracing JSON + overlap report to ./output/e2e_vpp_demo/.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


def build_perf(label, pp, vp, mbc, async_pp):
    """Build and configure a PerfLLM instance for the given settings."""
    model_cfg = ModelConfig.init_from_config_file(
        get_simu_model_config("llama3-8b")
    )
    model_cfg.padded_vocab_size = True
    model_cfg.make_vocab_size_divisible_by = 128

    sys_cfg = SystemConfig.init_from_config_file(
        get_simu_system_config("a100_pcie")
    )

    s = StrategyConfig.init_from_config_file(
        get_simu_strategy_config("tp1_pp2_dp4_mbs1")
    )
    s.pp_size = pp
    s.interleaving_size = vp
    s.pp_comm_async = async_pp
    s.micro_batch_num = mbc

    perf = PerfLLM()
    perf.configure(strategy_config=s, model_config=model_cfg, system_config=sys_cfg)
    perf.run_estimate()
    perf._apply_cost_models()
    perf._run()

    label_full = (
        f"PP{pp}_VP{vp}_mbc{mbc}_{'async' if async_pp else 'sync'}_{label}"
    )
    return perf, label_full


def main():
    scenarios = [
        ("baseline_sync", 2, 1, 4, False),
        ("baseline_async", 2, 1, 4, True),
        ("vpp_sync", 4, 2, 4, False),
    ]

    for label, pp, vp, mbc, async_pp in scenarios:
        print(f"\n{'=' * 70}")
        print(f"  {label}: PP={pp}, VP={vp}, mbc={mbc}, async={async_pp}")
        print(f"{'=' * 70}")

        perf, tag = build_perf(label, pp, vp, mbc, async_pp)
        out_dir = os.path.join("output", "e2e_vpp_demo", tag)
        perf.run_estimate_with_overlap(output_dir=out_dir)

        it_ms = perf._overlap_summary.iteration_time if perf._overlap_summary else 0
        overlap = (
            perf._overlap_summary.overall_overlap_ratio
            if perf._overlap_summary
            else 0
        )
        print(f"  iteration_time = {it_ms * 1e3:.0f} μs")
        print(f"  overlap_ratio  = {overlap:.1%}")
        print(f"  output → {out_dir}")

    print("\nAll E2E VPP demos complete. Open output/e2e_vpp_demo/ in chrome://tracing.")


if __name__ == "__main__":
    main()
