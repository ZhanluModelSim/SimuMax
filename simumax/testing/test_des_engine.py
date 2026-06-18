"""Unit tests for the multi-resource DES engine (Plan 2)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simumax.core.des_engine import (
    ResourceType,
    ResourceEvent,
    ResourceQueue,
    MultiResourceDES,
    OverlapTracker,
    ModuleOverlapStats,
    CommOverlapStats,
)


class TestResourceQueue(unittest.TestCase):

    def test_schedule_advances_time(self):
        q = ResourceQueue(resource=ResourceType.COMPUTE)
        e = q.schedule(0.5, "op1", "mod1", "fwd")
        self.assertAlmostEqual(e.start_time, 0.0)
        self.assertAlmostEqual(e.end_time, 0.5)
        self.assertAlmostEqual(q.current_time, 0.5)

    def test_multiple_schedules(self):
        q = ResourceQueue(resource=ResourceType.COMPUTE)
        q.schedule(0.3, "op1", "mod1", "fwd")
        e2 = q.schedule(0.2, "op2", "mod2", "fwd")
        self.assertAlmostEqual(e2.start_time, 0.3)
        self.assertAlmostEqual(e2.end_time, 0.5)
        self.assertAlmostEqual(q.current_time, 0.5)

    def test_advance_to(self):
        q = ResourceQueue(resource=ResourceType.COMPUTE)
        q.schedule(0.3, "op1", "mod1", "fwd")
        q.advance_to(1.0)
        self.assertAlmostEqual(q.current_time, 1.0)
        self.assertAlmostEqual(q.total_idle_time, 0.7)

    def test_advance_to_no_op(self):
        q = ResourceQueue(resource=ResourceType.COMPUTE)
        q.schedule(0.5, "op1", "mod1", "fwd")
        q.advance_to(0.3)
        self.assertAlmostEqual(q.current_time, 0.5)

    def test_utilization(self):
        q = ResourceQueue(resource=ResourceType.COMPUTE)
        q.schedule(0.5, "op1", "mod1", "fwd")
        q.advance_to(1.0)
        self.assertAlmostEqual(q.utilization, 0.5)

    def test_busy_time_tracking(self):
        q = ResourceQueue(resource=ResourceType.COMPUTE)
        q.schedule(0.3, "op1", "mod1", "fwd")
        q.schedule(0.2, "op2", "mod2", "fwd")
        self.assertAlmostEqual(q.total_busy_time, 0.5)


class TestResourceEvent(unittest.TestCase):

    def test_creation(self):
        e = ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0,
            end_time=0.5,
            op_name="matmul",
            module_path="layer_0.linear",
            stage="fwd",
        )
        self.assertEqual(e.resource, ResourceType.COMPUTE)
        self.assertEqual(e.op_name, "matmul")
        self.assertEqual(e.rank, 0)


class TestOverlapTracker(unittest.TestCase):

    def test_no_events(self):
        tracker = OverlapTracker()
        summary = tracker.compute_overlap()
        self.assertEqual(summary.total_compute_time, 0.0)
        self.assertEqual(summary.total_comm_time, 0.0)

    def test_compute_only(self):
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=1.0,
            op_name="compute", module_path="mod1", stage="fwd",
        ))
        summary = tracker.compute_overlap()
        self.assertAlmostEqual(summary.total_compute_time, 1.0)
        self.assertAlmostEqual(summary.total_comm_time, 0.0)

    def test_full_overlap(self):
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
        self.assertAlmostEqual(summary.total_compute_time, 1.0)
        self.assertAlmostEqual(summary.total_comm_time, 0.5)
        self.assertAlmostEqual(summary.total_overlapped_comm_time, 0.5)
        self.assertAlmostEqual(summary.overall_overlap_ratio, 1.0)

    def test_no_overlap(self):
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=1.0,
            op_name="compute", module_path="mod1", stage="fwd",
        ))
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.INTRA_LINK,
            start_time=1.0, end_time=1.5,
            op_name="all_reduce", module_path="mod1", stage="fwd",
        ))
        summary = tracker.compute_overlap()
        self.assertAlmostEqual(summary.total_overlapped_comm_time, 0.0)
        self.assertAlmostEqual(summary.overall_overlap_ratio, 0.0)

    def test_partial_overlap(self):
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=1.0,
            op_name="compute", module_path="mod1", stage="fwd",
        ))
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.INTRA_LINK,
            start_time=0.5, end_time=1.5,
            op_name="all_reduce", module_path="mod1", stage="fwd",
        ))
        summary = tracker.compute_overlap()
        self.assertAlmostEqual(summary.total_comm_time, 1.0)
        self.assertAlmostEqual(summary.total_overlapped_comm_time, 0.5)
        self.assertAlmostEqual(summary.overall_overlap_ratio, 0.5)

    def test_per_module_aggregation(self):
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=1.0,
            op_name="compute", module_path="layer0.linear", stage="fwd",
        ))
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.INTRA_LINK,
            start_time=0.0, end_time=0.5,
            op_name="all_reduce", module_path="layer0.linear", stage="fwd",
        ))
        summary = tracker.compute_overlap()
        self.assertIn("layer0.linear", summary.per_module)
        stats = summary.per_module["layer0.linear"]
        self.assertAlmostEqual(stats.fwd_compute_time, 1.0)
        self.assertAlmostEqual(stats.fwd_comm_time, 0.5)
        self.assertAlmostEqual(stats.fwd_overlapped_time, 0.5)

    def test_per_comm_type_aggregation(self):
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=2.0,
            op_name="compute", module_path="mod1", stage="fwd",
        ))
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.INTRA_LINK,
            start_time=0.0, end_time=0.5,
            op_name="all_reduce", module_path="mod1", stage="fwd",
        ))
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.INTRA_LINK,
            start_time=0.5, end_time=1.0,
            op_name="all_gather", module_path="mod2", stage="fwd",
        ))
        summary = tracker.compute_overlap()
        self.assertIn("all_reduce", summary.per_comm_type)
        self.assertIn("all_gather", summary.per_comm_type)

    def test_bwd_stage_aggregation(self):
        tracker = OverlapTracker()
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.COMPUTE,
            start_time=0.0, end_time=1.0,
            op_name="compute", module_path="mod1", stage="bwd",
        ))
        tracker.record_event(0, ResourceEvent(
            resource=ResourceType.INTRA_LINK,
            start_time=0.0, end_time=0.3,
            op_name="all_reduce", module_path="mod1", stage="bwd",
        ))
        summary = tracker.compute_overlap()
        stats = summary.per_module["mod1"]
        self.assertAlmostEqual(stats.bwd_compute_time, 1.0)
        self.assertAlmostEqual(stats.bwd_comm_time, 0.3)
        self.assertAlmostEqual(stats.bwd_overlapped_time, 0.3)


class TestModuleOverlapStats(unittest.TestCase):

    def test_finalize(self):
        s = ModuleOverlapStats(
            module_path="mod1",
            fwd_compute_time=1.0,
            fwd_comm_time=0.5,
            fwd_overlapped_time=0.3,
            bwd_compute_time=2.0,
            bwd_comm_time=0.8,
            bwd_overlapped_time=0.4,
        )
        s.finalize()
        self.assertAlmostEqual(s.fwd_exposed_time, 0.2)
        self.assertAlmostEqual(s.fwd_overlap_ratio, 0.6)
        self.assertAlmostEqual(s.bwd_exposed_time, 0.4)
        self.assertAlmostEqual(s.bwd_overlap_ratio, 0.5)


class TestCommOverlapStats(unittest.TestCase):

    def test_finalize(self):
        s = CommOverlapStats(
            comm_type="all_reduce",
            total_time=1.0,
            overlapped_time=0.7,
        )
        s.finalize()
        self.assertAlmostEqual(s.exposed_time, 0.3)
        self.assertAlmostEqual(s.overlap_ratio, 0.7)


class TestMultiResourceDES(unittest.TestCase):

    def test_creation(self):
        des = MultiResourceDES(num_ranks=2)
        self.assertEqual(des.num_ranks, 2)
        self.assertIn(ResourceType.COMPUTE, des.rank_resources[0])
        self.assertIn(ResourceType.INTRA_LINK, des.rank_resources[0])

    def test_schedule_compute(self):
        des = MultiResourceDES(num_ranks=1)
        evt = des.schedule_compute(
            0, 0.5, "matmul", "layer0.linear", "fwd"
        )
        self.assertAlmostEqual(evt.end_time, 0.5)

    def test_schedule_intra_comm(self):
        des = MultiResourceDES(num_ranks=2)
        des.schedule_intra_comm(
            [0, 1], 0.3, "all_reduce", "layer0.linear", "fwd"
        )
        intra0 = des.get_queue(0, ResourceType.INTRA_LINK)
        intra1 = des.get_queue(1, ResourceType.INTRA_LINK)
        self.assertAlmostEqual(intra0.current_time, 0.3)
        self.assertAlmostEqual(intra1.current_time, 0.3)

    def test_schedule_inter_comm(self):
        des = MultiResourceDES(num_ranks=2)
        des.schedule_inter_comm(
            [0, 1], 0.5, "p2p", "pp_send", "fwd"
        )
        inter0 = des.get_queue(0, ResourceType.INTER_LINK)
        self.assertAlmostEqual(inter0.current_time, 0.5)

    def test_compute_overlap(self):
        des = MultiResourceDES(num_ranks=1)
        des.schedule_compute(0, 1.0, "compute", "mod1", "fwd")
        des.schedule_intra_comm([0], 0.5, "all_reduce", "mod1", "fwd")
        summary = des.compute_overlap()
        self.assertAlmostEqual(summary.total_compute_time, 1.0)
        self.assertAlmostEqual(summary.total_comm_time, 0.5)

    def test_get_iteration_time(self):
        des = MultiResourceDES(num_ranks=2)
        des.schedule_compute(0, 1.0, "compute", "mod1", "fwd")
        des.schedule_compute(1, 2.0, "compute", "mod1", "fwd")
        self.assertAlmostEqual(des.get_iteration_time(), 2.0)

    def test_sync_rank_lanes(self):
        des = MultiResourceDES(num_ranks=1)
        des.schedule_compute(0, 1.0, "compute", "mod1", "fwd")
        des.schedule_intra_comm([0], 0.3, "all_reduce", "mod1", "fwd")
        des.sync_rank_lanes(0)
        comp_q = des.get_queue(0, ResourceType.COMPUTE)
        intra_q = des.get_queue(0, ResourceType.INTRA_LINK)
        self.assertAlmostEqual(comp_q.current_time, intra_q.current_time)

    def test_multi_resource_parallel(self):
        """Compute and comm on separate resources run in parallel."""
        des = MultiResourceDES(num_ranks=1)
        des.schedule_compute(0, 1.0, "compute", "mod1", "fwd")
        des.schedule_intra_comm([0], 0.5, "all_reduce", "mod1", "fwd")
        des.schedule_inter_comm([0], 0.3, "p2p", "pp_send", "fwd")
        comp_q = des.get_queue(0, ResourceType.COMPUTE)
        intra_q = des.get_queue(0, ResourceType.INTRA_LINK)
        inter_q = des.get_queue(0, ResourceType.INTER_LINK)
        self.assertAlmostEqual(comp_q.current_time, 1.0)
        self.assertAlmostEqual(intra_q.current_time, 0.5)
        self.assertAlmostEqual(inter_q.current_time, 0.3)


if __name__ == "__main__":
    unittest.main()
