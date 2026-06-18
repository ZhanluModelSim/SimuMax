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
            output_path = os.path.join(tmpdir, "report.json")
            OverlapReport.generate(summary, output_path)
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


if __name__ == "__main__":
    unittest.main()
