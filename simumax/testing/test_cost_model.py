"""Unit tests for the pluggable CostModel interface (Plan 1)."""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simumax.core.cost_model import (
    CostContext,
    CostResult,
    CostModel,
    TableLookupCostModel,
    FormulaCostModel,
    OverrideCostModel,
    MemoryAccessCostModel,
    CostModelRegistry,
)


def _make_mock_system():
    system = MagicMock()
    system.compute_op_accuracy_time.return_value = {
        "op_name": "matmul",
        "tflops": 312,
        "efficient_factor": 0.75,
        "compute_only_time": 0.05,
    }
    system.compute_mem_access_time.return_value = {
        "gbps": 1600,
        "efficient_factor": 0.91,
        "latency_us": 40,
        "io_time": 0.01,
    }
    system.compute_end2end_time.return_value = 0.05
    return system


def _make_context(op_name="matmul", flops=1000000, accessed_mem=8000):
    return CostContext(
        op_name=op_name,
        stage="fwd",
        flops=flops,
        accessed_mem=accessed_mem,
        shape_desc="b=1, m=4096, k=4096, n=4096",
        element_size=2,
        strategy=MagicMock(),
        system=_make_mock_system(),
    )


class TestCostContext(unittest.TestCase):

    def test_creation(self):
        ctx = _make_context()
        self.assertEqual(ctx.op_name, "matmul")
        self.assertEqual(ctx.stage, "fwd")
        self.assertEqual(ctx.flops, 1000000)


class TestCostResult(unittest.TestCase):

    def test_creation(self):
        r = CostResult(compute_time=0.05, mem_time=0.01, end2end_time=0.05)
        self.assertEqual(r.compute_time, 0.05)
        self.assertEqual(r.mem_time, 0.01)
        self.assertEqual(r.details, {})


class TestTableLookupCostModel(unittest.TestCase):

    def test_name(self):
        m = TableLookupCostModel()
        self.assertEqual(m.name, "table_lookup")

    def test_compute_uses_system(self):
        m = TableLookupCostModel()
        ctx = _make_context()
        result = m.compute(ctx)
        self.assertIsInstance(result, CostResult)
        self.assertEqual(result.end2end_time, 0.05)
        ctx.system.compute_op_accuracy_time.assert_called_once()
        ctx.system.compute_mem_access_time.assert_called_once()
        ctx.system.compute_end2end_time.assert_called_once()

    def test_custom_op_name(self):
        m = TableLookupCostModel(op_name="fp8_matmul")
        ctx = _make_context(op_name="matmul")
        result = m.compute(ctx)
        call_args = ctx.system.compute_op_accuracy_time.call_args
        self.assertEqual(call_args[0][0], "fp8_matmul")

    def test_custom_bandwidth_op_name(self):
        m = TableLookupCostModel(bandwidth_op_name="ce")
        ctx = _make_context()
        result = m.compute(ctx)
        call_args = ctx.system.compute_mem_access_time.call_args
        self.assertEqual(call_args[0][0], "ce")


class TestFormulaCostModel(unittest.TestCase):

    def test_name(self):
        m = FormulaCostModel(compute_fn=lambda ctx: 0.1)
        self.assertEqual(m.name, "formula")

    def test_custom_name(self):
        m = FormulaCostModel(
            compute_fn=lambda ctx: 0.1, model_name="my_formula"
        )
        self.assertEqual(m.name, "my_formula")

    def test_compute_with_custom_fn(self):
        m = FormulaCostModel(
            compute_fn=lambda ctx: ctx.flops / 1e9,
            mem_fn=lambda ctx: ctx.accessed_mem / 1e6,
        )
        ctx = _make_context(flops=2000000, accessed_mem=500000)
        result = m.compute(ctx)
        self.assertAlmostEqual(result.compute_time, 0.002)
        self.assertAlmostEqual(result.mem_time, 0.5)
        self.assertAlmostEqual(result.end2end_time, 0.5)

    def test_default_combine_is_max(self):
        m = FormulaCostModel(
            compute_fn=lambda ctx: 0.03,
            mem_fn=lambda ctx: 0.05,
        )
        ctx = _make_context()
        result = m.compute(ctx)
        self.assertAlmostEqual(result.end2end_time, 0.05)

    def test_custom_combine(self):
        m = FormulaCostModel(
            compute_fn=lambda ctx: 0.03,
            mem_fn=lambda ctx: 0.05,
            combine_fn=lambda c, m: c + m,
        )
        ctx = _make_context()
        result = m.compute(ctx)
        self.assertAlmostEqual(result.end2end_time, 0.08)


class TestOverrideCostModel(unittest.TestCase):

    def test_name(self):
        m = OverrideCostModel(fixed_time_ms=0.1)
        self.assertEqual(m.name, "override")

    def test_returns_fixed_time(self):
        m = OverrideCostModel(fixed_time_ms=0.42)
        ctx = _make_context()
        result = m.compute(ctx)
        self.assertEqual(result.compute_time, 0.42)
        self.assertEqual(result.mem_time, 0.0)
        self.assertEqual(result.end2end_time, 0.42)
        self.assertEqual(result.details["source"], "fixed_override")


class TestMemoryAccessCostModel(unittest.TestCase):

    def test_name(self):
        m = MemoryAccessCostModel()
        self.assertEqual(m.name, "memory_access")

    def test_compute_uses_bandwidth(self):
        m = MemoryAccessCostModel(bandwidth_op_name="permute_fwd")
        ctx = _make_context()
        result = m.compute(ctx)
        self.assertEqual(result.compute_time, 0.0)
        self.assertEqual(result.end2end_time, result.mem_time)
        call_args = ctx.system.compute_mem_access_time.call_args
        self.assertEqual(call_args[0][0], "permute_fwd")


class TestCostModelRegistry(unittest.TestCase):

    def test_default_registrations(self):
        available = CostModelRegistry.available()
        self.assertIn("table_lookup", available)
        self.assertIn("formula", available)
        self.assertIn("override", available)
        self.assertIn("memory_access", available)

    def test_create_table_lookup(self):
        m = CostModelRegistry.create("table_lookup")
        self.assertIsInstance(m, TableLookupCostModel)

    def test_create_override(self):
        m = CostModelRegistry.create("override", fixed_time_ms=0.1)
        self.assertIsInstance(m, OverrideCostModel)

    def test_create_unknown_raises(self):
        with self.assertRaises(KeyError):
            CostModelRegistry.create("nonexistent")

    def test_from_config(self):
        config = {"type": "override", "fixed_time_ms": 0.5}
        m = CostModelRegistry.from_config(config)
        self.assertIsInstance(m, OverrideCostModel)

    def test_register_custom(self):
        class MyCostModel(CostModel):
            @property
            def name(self):
                return "my_custom"
            def compute(self, ctx):
                return CostResult(0, 0, 0)

        CostModelRegistry.register("my_custom", MyCostModel)
        m = CostModelRegistry.create("my_custom")
        self.assertIsInstance(m, MyCostModel)


if __name__ == "__main__":
    unittest.main()
