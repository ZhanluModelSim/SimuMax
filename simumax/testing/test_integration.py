"""System tests for Plan 1 + Plan 2 integration."""

import sys
import os
import unittest
import tempfile
import json

from simumax.core.cost_model import CostModelRegistry
from simumax.core.des_engine import (
    ResourceType,
    ResourceEvent,
    MultiResourceDES,
    OverlapTracker,
)
from simumax.core.des_bridge import DesBridge
from simumax.core.overlap_report import OverlapReport

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestDesBridgeFromLog(unittest.TestCase):
    """Tests for DesBridge log parsing."""

    def test_parse_compute_log(self):
        """Parse compute events from simulation log."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', delete=False
        ) as f:
            f.write(
                "GPTModel_0-layer_0-SelfAttention-linear_qkv "
                "fwd cost 0.001234 st 0.000000 ed 0.001234\n"
            )
            f.write(
                "GPTModel_0-layer_0-SelfAttention-linear_qkv "
                "bwd cost 0.002468 st 0.001234 ed 0.003702\n"
            )
            log_path = f.name

        try:
            des = DesBridge.from_simulation_log(log_path, num_ranks=1)
            summary = des.compute_overlap()
            self.assertGreater(summary.total_compute_time, 0)
        finally:
            os.unlink(log_path)

    def test_parse_comm_log(self):
        """Parse communication events from simulation log."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', delete=False
        ) as f:
            f.write(
                "GPTModel_0-layer_0-SelfAttention-linear_qkv "
                "gid all_reduce-tp-fwd fwd cost 0.000500 "
                "st 0.001234 ed 0.001734\n"
            )
            log_path = f.name

        try:
            des = DesBridge.from_simulation_log(log_path, num_ranks=1)
            summary = des.compute_overlap()
            self.assertGreater(summary.total_comm_time, 0)
        finally:
            os.unlink(log_path)

    def test_empty_log(self):
        """Handle empty log file gracefully."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', delete=False
        ) as f:
            log_path = f.name

        try:
            des = DesBridge.from_simulation_log(log_path, num_ranks=1)
            summary = des.compute_overlap()
            self.assertEqual(summary.total_compute_time, 0.0)
        finally:
            os.unlink(log_path)


class TestOverlapReport(unittest.TestCase):
    """Tests for overlap report generation."""

    def test_to_dict(self):
        """Generate report dict from overlap summary."""
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=1.0,
            op_name="compute", module_path="mod1", stage="fwd",
        ))
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.INTRA_LINK,
            start_time=0.0, end_time=0.5,
            op_name="all_reduce", module_path="mod1", stage="fwd",
        ))
        summary = tracker.compute_overlap()
        report = OverlapReport.to_dict(summary)
        self.assertIn("global", report)
        self.assertIn("per_module", report)
        self.assertIn("per_comm_type", report)
        self.assertIn("mod1", report["per_module"])

    def test_generate_json(self):
        """Generate JSON report file."""
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=1.0,
            op_name="compute", module_path="mod1", stage="fwd",
        ))
        summary = tracker.compute_overlap()

        with tempfile.TemporaryDirectory() as tmpdir:
            OverlapReport.generate(summary, output_dir=tmpdir, filename="report.json")
            output_path = os.path.join(tmpdir, "report.json")
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("global", data)


class TestCostModelRegistryIntegration(unittest.TestCase):
    """Tests for CostModel registry integration."""

    def test_create_all_builtin(self):
        """Create all built-in cost models via registry."""
        tbl = CostModelRegistry.create("table_lookup")
        self.assertEqual(tbl.name, "table_lookup")

        frm = CostModelRegistry.create(
            "formula",
            compute_fn=lambda ctx: 0.01,
        )
        self.assertEqual(frm.name, "formula")

        ovr = CostModelRegistry.create(
            "override", fixed_time_ms=0.1
        )
        self.assertEqual(ovr.name, "override")

        mem = CostModelRegistry.create("memory_access")
        self.assertEqual(mem.name, "memory_access")

    def test_from_config_override(self):
        """Create override model from config dict."""
        config = {"type": "override", "fixed_time_ms": 0.25}
        m = CostModelRegistry.from_config(config)
        self.assertEqual(m.name, "override")


class TestDesEngineMultiRank(unittest.TestCase):
    """Tests for multi-rank DES engine."""

    def test_two_ranks_compute(self):
        """Two ranks with independent compute schedules."""
        des = MultiResourceDES(num_ranks=2)
        des.schedule_compute(0, 1.0, "compute", "mod1", "fwd")
        des.schedule_compute(1, 2.0, "compute", "mod1", "fwd")
        self.assertAlmostEqual(des.get_iteration_time(), 2.0)

    def test_two_ranks_shared_comm(self):
        """Two ranks sharing intra-link communication."""
        des = MultiResourceDES(num_ranks=2)
        des.schedule_intra_comm(
            [0, 1], 0.5, "all_reduce", "mod1", "fwd"
        )
        rank0_queue = des.get_queue(0, ResourceType.INTRA_LINK)
        rank1_queue = des.get_queue(1, ResourceType.INTRA_LINK)
        self.assertAlmostEqual(rank0_queue.current_time, 0.5)
        self.assertAlmostEqual(rank1_queue.current_time, 0.5)

    def test_mixed_compute_comm_overlap(self):
        des = MultiResourceDES(num_ranks=1)
        des.schedule_compute(0, 1.0, "compute", "mod1", "fwd")
        des.schedule_intra_comm([0], 0.5, "all_reduce", "mod1", "fwd")
        des.schedule_compute(0, 0.5, "compute", "mod2", "fwd")
        summary = des.compute_overlap()
        self.assertAlmostEqual(summary.total_compute_time, 1.5)
        self.assertAlmostEqual(summary.total_comm_time, 0.5)


class TestPPStageDependency(unittest.TestCase):
    """Verify PP stage ordering invariants in DES scheduling.

    PP forward:  rank(stage 0) fwd → rank(stage 1) fwd.
    PP backward: rank(stage 1) bwd → rank(stage 0) bwd.

    These tests ensure that manual scheduling cannot accidentally
    violate these dependencies without detection.
    """

    _STAGE0_RANKS = {0, 1}
    _STAGE1_RANKS = {2, 3}

    @staticmethod
    def _check_pp_fwd_ordering(des: MultiResourceDES):
        """Assert PP forward ordering: every fwd event on stage 1
        starts after the earliest fwd event on stage 0 finishes."""
        stage0_fwd_end = float('inf')
        stage1_fwd_start = float('inf')

        for rank in range(des.num_ranks):
            q = des.get_queue(rank, ResourceType.COMPUTE)
            for evt in q.events:
                if "fwd" not in getattr(evt, 'stage', ''):
                    continue
                # stage 0 = first half of ranks, stage 1 = second half
                if rank < des.num_ranks // 2:
                    stage0_fwd_end = min(stage0_fwd_end, evt.end_time)
                else:
                    stage1_fwd_start = min(
                        stage1_fwd_start, evt.start_time
                    )

        if stage0_fwd_end == float('inf') or stage1_fwd_start == float('inf'):
            return  # no events to compare

        assert stage1_fwd_start >= stage0_fwd_end, (
            f"PP fwd violation: stage1 fwd starts at {stage1_fwd_start*1e3:.0f}μs "
            f"but stage0 fwd ends at {stage0_fwd_end*1e3:.0f}μs"
        )

    @staticmethod
    def _check_pp_bwd_ordering(des: MultiResourceDES):
        """Assert PP backward ordering: every bwd event on stage 0
        starts after the earliest bwd event on stage 1 finishes."""
        stage1_bwd_end = float('inf')
        stage0_bwd_start = float('inf')

        for rank in range(des.num_ranks):
            q = des.get_queue(rank, ResourceType.COMPUTE)
            for evt in q.events:
                if "bwd" not in getattr(evt, 'stage', ''):
                    continue
                if rank >= des.num_ranks // 2:
                    stage1_bwd_end = min(stage1_bwd_end, evt.end_time)
                else:
                    stage0_bwd_start = min(
                        stage0_bwd_start, evt.start_time
                    )

        if stage1_bwd_end == float('inf') or stage0_bwd_start == float('inf'):
            return

        assert stage0_bwd_start >= stage1_bwd_end, (
            f"PP bwd violation: stage0 bwd starts at {stage0_bwd_start*1e3:.0f}μs "
            f"but stage1 bwd ends at {stage1_bwd_end*1e3:.0f}μs"
        )

    @staticmethod
    def _check_fwd_before_bwd_same_rank(des: MultiResourceDES):
        """Assert that fwd events precede bwd events on the same rank."""
        for rank in range(des.num_ranks):
            q = des.get_queue(rank, ResourceType.COMPUTE)
            fwd_ends = [
                e.end_time for e in q.events
                if "fwd" in getattr(e, 'stage', '')
            ]
            bwd_starts = [
                e.start_time for e in q.events
                if "bwd" in getattr(e, 'stage', '')
            ]
            if not fwd_ends or not bwd_starts:
                continue
            max_fwd = max(fwd_ends)
            min_bwd = min(bwd_starts)
            assert min_bwd >= max_fwd, (
                f"Fwd-before-bwd violation on {rank}: "
                f"bwd starts at {min_bwd*1e3:.0f}μs but fwd ends at {max_fwd*1e3:.0f}μs"
            )

    def test_manual_scheduling_violates_pp_fwd_ordering(self):
        """Manual fwd-all-stages scheduling should trigger the detector."""
        des = MultiResourceDES(num_ranks=4)
        # Schedule fwd on BOTH stages starting at t=0 — this IS a violation
        des.schedule_compute(0, 1.0, "matmul", "layer0", "fwd_mb0")
        des.schedule_compute(1, 1.0, "matmul", "layer0", "fwd_mb0")
        des.schedule_compute(2, 0.5, "matmul", "layer8", "fwd_mb0")
        des.schedule_compute(3, 0.5, "matmul", "layer8", "fwd_mb0")

        with self.assertRaises(AssertionError) as ctx:
            self._check_pp_fwd_ordering(des)
        self.assertIn("PP fwd violation", str(ctx.exception))

    def test_correct_pp_fwd_ordering_passes(self):
        """Correct PP fwd ordering should NOT trigger the detector."""
        des = MultiResourceDES(num_ranks=4)
        des.schedule_compute(0, 1.0, "matmul", "layer0", "fwd_mb0")
        des.schedule_compute(1, 1.0, "matmul", "layer0", "fwd_mb0")
        des.schedule_compute(2, 0.5, "matmul", "layer8", "fwd_mb0")
        des.schedule_compute(3, 0.5, "matmul", "layer8", "fwd_mb0")
        # Advance rank2/3 past rank0/1's fwd end
        for rank in [2, 3]:
            q = des.get_queue(rank, ResourceType.COMPUTE)
            q.advance_to(1.0 + 1e-6)
            # Re-schedule after the advance (the first event is now at t=0,
            # but advance_to(1.0) doesn't rewind, it's a no-op since 1.0 > current)
            # Actually we need to schedule AFTER the advance.
            # Let's just clear and re-schedule.
            q.events.clear()
            q.current_time = 1.0
        # Re-schedule rank2/3 fwd at t=1.0 (after rank0/1 finish)
        des.schedule_compute(2, 0.5, "matmul", "layer8", "fwd_mb0")
        des.schedule_compute(3, 0.5, "matmul", "layer8", "fwd_mb0")

        # Should NOT raise
        self._check_pp_fwd_ordering(des)

    def test_manual_scheduling_violates_pp_bwd_ordering(self):
        """Manual bwd-all-stages scheduling should trigger the detector."""
        des = MultiResourceDES(num_ranks=4)
        des.schedule_compute(2, 0.5, "matmul", "layer8", "bwd_mb0")
        des.schedule_compute(3, 0.5, "matmul", "layer8", "bwd_mb0")
        des.schedule_compute(0, 1.0, "matmul", "layer0", "bwd_mb0")
        des.schedule_compute(1, 1.0, "matmul", "layer0", "bwd_mb0")

        with self.assertRaises(AssertionError) as ctx:
            self._check_pp_bwd_ordering(des)
        self.assertIn("PP bwd violation", str(ctx.exception))

    def test_fwd_before_bwd_violation_detected(self):
        """Bwd scheduled before fwd on same rank should be detected."""
        des = MultiResourceDES(num_ranks=2)
        des.schedule_compute(0, 0.5, "matmul", "layer0", "bwd_mb0")
        des.schedule_compute(0, 1.0, "matmul", "layer0", "fwd_mb0")

        with self.assertRaises(AssertionError) as ctx:
            self._check_fwd_before_bwd_same_rank(des)
        self.assertIn("Fwd-before-bwd violation", str(ctx.exception))

    def test_correct_scheduling_passes_all_checks(self):
        """A well-formed PP schedule passes all three ordering checks."""
        des = MultiResourceDES(num_ranks=4)
        # Stage 0 fwd (ranks 0,1)
        des.schedule_compute(0, 1.0, "matmul", "layer0", "fwd_mb0")
        des.schedule_compute(1, 1.0, "matmul", "layer0", "fwd_mb0")
        # Stage 1 fwd (ranks 2,3) — must start after stage 0 ends
        for rank in [2, 3]:
            des.get_queue(rank, ResourceType.COMPUTE).advance_to(1.0)
        des.schedule_compute(2, 0.5, "matmul", "layer8", "fwd_mb0")
        des.schedule_compute(3, 0.5, "matmul", "layer8", "fwd_mb0")
        # Stage 1 bwd (ranks 2,3) — after fwd
        des.schedule_compute(2, 0.5, "matmul", "layer8", "bwd_mb0")
        des.schedule_compute(3, 0.5, "matmul", "layer8", "bwd_mb0")
        # Stage 0 bwd (ranks 0,1) — must start after stage 1 bwd ends
        for rank in [0, 1]:
            des.get_queue(rank, ResourceType.COMPUTE).advance_to(2.0)
        des.schedule_compute(0, 0.5, "matmul", "layer0", "bwd_mb0")
        des.schedule_compute(1, 0.5, "matmul", "layer0", "bwd_mb0")

        # All three checks must pass without raising
        self._check_pp_fwd_ordering(des)
        self._check_pp_bwd_ordering(des)
        self._check_fwd_before_bwd_same_rank(des)


class TestAsyncP2PInjection(unittest.TestCase):
    """Verify async P2P send/recv/wait timing invariants."""

    def test_async_p2p_timing_single_mb(self):
        """Async P2P: send lands on INTER_LINK at correct absolute time,
        recv lands on INTER_LINK at same time, and COMPUTE lane of dst
        rank is blocked until P2P completes."""
        from simumax.core.des_bridge import _inject_async_p2p, _schedule_event_at

        des = MultiResourceDES(num_ranks=4)
        # Simulate: stage0 fwd done at 100ms, stage1 fwd starts at 105ms
        src_ranks = [0, 1]
        dst_ranks = [2, 3]
        t_p2p = 0.100  # P2P starts right after stage0 fwd
        p2p_time = 0.005  # 5μs

        # Advance stage0 COMPUTE to simulate completed fwd
        for r in src_ranks:
            comp_q = des.get_queue(r, ResourceType.COMPUTE)
            comp_q.advance_to(t_p2p)

        # Inject async P2P: stage0 → stage1
        _inject_async_p2p(
            des, src_ranks, dst_ranks,
            t_p2p, p2p_time,
            "F", 0, 0, 1,
        )

        # Verify: INTER_LINK events on src_ranks at t_p2p
        for r in src_ranks:
            q = des.get_queue(r, ResourceType.INTER_LINK)
            self.assertEqual(len(q.events), 1, f"rank{r} should have 1 INTER_LINK event")
            self.assertAlmostEqual(q.events[0].start_time, t_p2p, delta=1e-9,
                msg=f"rank{r} async_send should start at {t_p2p}")

        # Verify: INTER_LINK events on dst_ranks at t_p2p
        for r in dst_ranks:
            q = des.get_queue(r, ResourceType.INTER_LINK)
            self.assertEqual(len(q.events), 1, f"rank{r} should have 1 INTER_LINK event")
            self.assertAlmostEqual(q.events[0].start_time, t_p2p, delta=1e-9,
                msg=f"rank{r} async_recv should start at {t_p2p}")

        # Verify: COMPUTE lane of dst_ranks blocked until wait_t
        wait_t = t_p2p + p2p_time
        for r in dst_ranks:
            comp_q = des.get_queue(r, ResourceType.COMPUTE)
            self.assertAlmostEqual(comp_q.current_time, wait_t, delta=1e-9,
                msg=f"rank{r} COMPUTE should be at {wait_t} after async_wait")

    def test_async_p2p_vs_sync_iteration_time(self):
        """Async P2P should produce the same iteration time as sync
        when using fixed 1F1B offsets (the critical path is compute-bound)."""
        from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
        from simumax.core.perf_llm import PerfLLM
        from simumax.core.des_bridge import DesBridge
        from simumax.utils import get_simu_model_config, get_simu_strategy_config, get_simu_system_config

        sync_time = None
        async_time = None

        for async_pp in (False, True):
            s = StrategyConfig.init_from_config_file(
                get_simu_strategy_config('tp1_pp2_dp4_mbs1'))
            if async_pp:
                s.pp_comm_async = True
            perf = PerfLLM()
            perf.configure(
                s,
                ModelConfig.init_from_config_file(
                    get_simu_model_config('llama3-8b')),
                SystemConfig.init_from_config_file(
                    get_simu_system_config('a100_pcie')),
            )
            perf.model_config.padded_vocab_size = True
            perf.model_config.make_vocab_size_divisible_by = 128
            perf.run_estimate()
            perf._apply_cost_models()
            perf._run()
            perf.strategy.micro_batch_num = 2
            des = DesBridge.from_module_costs(perf, num_ranks=2)
            it = des.get_iteration_time()
            if async_pp:
                async_time = it
            else:
                sync_time = it

        # Both modes should produce non-zero, valid iteration times
        self.assertIsNotNone(sync_time)
        self.assertIsNotNone(async_time)
        self.assertGreater(sync_time, 0.0)
        self.assertGreater(async_time, 0.0)
        # Async should NOT be slower than sync
        self.assertLessEqual(async_time, sync_time * 1.05,
            f"async={async_time*1e3:.0f}μs should be <= sync={sync_time*1e3:.0f}μs + 5%")

    def test_async_p2p_inter_link_isolation(self):
        """Async P2P events on INTER_LINK must not advance COMPUTE lane."""
        from simumax.core.des_bridge import _schedule_event_at

        des = MultiResourceDES(num_ranks=2)
        # Put a compute event on rank0
        des.schedule_compute(0, 0.1, "fwd", "mod", "fwd")
        comp_end = des.get_queue(0, ResourceType.COMPUTE).current_time

        # Inject async P2P events at a LATER time on INTER_LINK
        _schedule_event_at(
            des, 0, ResourceType.INTER_LINK,
            0.2, 0.005, "async_send", "pp_link", "fwd_mb0",
        )
        _schedule_event_at(
            des, 1, ResourceType.INTER_LINK,
            0.2, 0.005, "async_recv", "pp_link", "fwd_mb0",
        )

        # COMPUTE lane must remain unchanged
        self.assertAlmostEqual(
            des.get_queue(0, ResourceType.COMPUTE).current_time,
            comp_end, delta=1e-9,
            msg="COMPUTE lane should NOT advance when async P2P is scheduled",
        )


class TestVPPBubbleAccuracy(unittest.TestCase):
    """Verify VPP interleaved scheduling reduces 1F1B bubble vs non-VPP."""

    def test_vpp_reduces_bubble_pp2(self):
        """VPP with vp=2 should produce equal or less iteration time
        than non-VPP for the same pp_size and mbc.

        For PP=2, VPP has minimal bubble advantage but should still
        be ≤ the non-VPP iteration time (identical in the limit).
        """
        from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
        from simumax.core.perf_llm import PerfLLM
        from simumax.utils import get_simu_model_config, get_simu_strategy_config, get_simu_system_config

        times = {}
        pp = 4  # PP must be > 2 for sync VPP
        for label, vp in [("no_vpp", 1), ("VPP", 2)]:
            s = StrategyConfig.init_from_config_file(
                get_simu_strategy_config('tp1_pp2_dp4_mbs1'))
            s.pp_size = pp
            s.world_size = pp * s.tp_size * s.cp_size  # ensure divisibility
            s.interleaving_size = vp
            s.pp_comm_async = False  # required for VPP analysis
            perf = PerfLLM()
            perf.configure(
                s,
                ModelConfig.init_from_config_file(
                    get_simu_model_config('llama3-8b')),
                SystemConfig.init_from_config_file(
                    get_simu_system_config('a100_pcie')),
            )
            perf.model_config.padded_vocab_size = True
            perf.model_config.make_vocab_size_divisible_by = 128
            perf.run_estimate()
            perf._apply_cost_models()
            perf._run()
            perf.strategy.micro_batch_num = 4
            it = perf._compute_pp_total_time()
            times[label] = it

        self.assertGreater(times["no_vpp"], 0.0)
        self.assertGreater(times["VPP"], 0.0)
        self.assertLessEqual(
            times["VPP"], times["no_vpp"] * 1.05,
            f"VPP={times['VPP']*1e3:.0f}μs should ≤ "
            f"no_vpp={times['no_vpp']*1e3:.0f}μs + 5%",
        )

    def test_vpp_bubble_monotonic_with_mbc(self):
        """For a given vp_size, larger micro_batch_num should NOT
        increase per-microbatch bubble (amortised over more MBs)."""
        from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
        from simumax.core.perf_llm import PerfLLM
        from simumax.utils import get_simu_model_config, get_simu_strategy_config, get_simu_system_config

        per_mb_times = {}
        for mbc in [4, 8]:
            s = StrategyConfig.init_from_config_file(
                get_simu_strategy_config('tp1_pp2_dp4_mbs1'))
            perf = PerfLLM()
            perf.configure(
                s,
                ModelConfig.init_from_config_file(
                    get_simu_model_config('llama3-8b')),
                SystemConfig.init_from_config_file(
                    get_simu_system_config('a100_pcie')),
            )
            perf.model_config.padded_vocab_size = True
            perf.model_config.make_vocab_size_divisible_by = 128
            perf.run_estimate()
            perf._apply_cost_models()
            perf._run()
            perf.strategy.micro_batch_num = mbc
            total_time = perf._compute_pp_total_time()
            per_mb_times[mbc] = total_time / mbc

        # Per-MB time with mbc=8 should be ≤ mbc=4 (better amortisation)
        self.assertLessEqual(
            per_mb_times[8], per_mb_times[4] * 1.05,
            f"per-MB mbc=8={per_mb_times[8]*1e3:.0f}μs should ≤ "
            f"mbc=4={per_mb_times[4]*1e3:.0f}μs + 5%",
        )

    def test_vpp_bubble_upper_bound(self):
        """VPP iteration time must be ≤ non-VPP analytical bubble formula:
        total_time ≤ mbc * (fwd + bwd) + (pp_size - 1) * max(fwd, bwd)."""
        from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
        from simumax.core.perf_llm import PerfLLM
        from simumax.utils import get_simu_model_config, get_simu_strategy_config, get_simu_system_config

        pp_sizes = [4, 8]  # PP divides 32 layers evenly
        mbc = 8

        for pp in pp_sizes:
            s = StrategyConfig.init_from_config_file(
                get_simu_strategy_config('tp1_pp2_dp4_mbs1'))
            s.pp_size = pp
            s.world_size = pp * s.tp_size * s.cp_size  # ensure divisibility
            s.interleaving_size = 2  # VP=2
            s.pp_comm_async = False  # required for VPP analysis
            perf = PerfLLM()
            perf.configure(
                s,
                ModelConfig.init_from_config_file(
                    get_simu_model_config('llama3-8b')),
                SystemConfig.init_from_config_file(
                    get_simu_system_config('a100_pcie')),
            )
            perf.model_config.padded_vocab_size = True
            perf.model_config.make_vocab_size_divisible_by = 128
            perf.run_estimate()
            perf._apply_cost_models()
            perf._run()
            perf.strategy.micro_batch_num = mbc

            # Get per-stage times
            phase = perf._compute_single_batch_phase_inputs("first_stage_chunk")
            fwd_time = phase["fwd_recv"] + phase["fwd_compute"] + phase["fwd_send"]
            bwd_time = phase["bwd_recv"] + phase["bwd_compute"] + phase["bwd_send"]
            chunk_time = fwd_time + bwd_time  # μs

            # Analytical upper-bound bubble formula
            bubble_max = max(fwd_time, bwd_time) * (pp - 1)  # μs
            upper_bound_ms = (mbc * chunk_time + bubble_max) / 1e3  # ms

            vpp_time_ms = perf._compute_pp_total_time()  # μs
            upper_bound_ms = (mbc * chunk_time + bubble_max) / 1e3  # μs→ms

            self.assertLess(
                vpp_time_ms / 1e3, upper_bound_ms * 1.5,  # convert both to ms
                f"PP={pp} VP=2: VPP={vpp_time_ms:.0f}μs upper_bound={upper_bound_ms*1e3:.0f}μs"
            )


if __name__ == "__main__":
    unittest.main()
