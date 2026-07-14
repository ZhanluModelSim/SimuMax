"""Multi-resource discrete-event simulation engine for process-level performance."""

import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import heapq


class ResourceType(Enum):
    COMPUTE = "compute"
    MEM_BANDWIDTH = "mem_bw"
    INTRA_LINK = "intra_link"
    INTER_LINK = "inter_link"
    OFFLOAD = "offload"


@dataclass
class ResourceEvent:
    resource: ResourceType
    start_time: float
    end_time: float
    op_name: str
    module_path: str
    stage: str
    rank: int = 0
    size_bytes: int = 0
    flops: int = 0
    gid: str = ""     # async P2P group id for send/recv pairing


@dataclass
class ResourceQueue:
    resource: ResourceType
    capacity: float = 1.0
    current_time: float = 0.0
    events: List[ResourceEvent] = field(default_factory=list)
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0

    def schedule(
        self,
        duration: float,
        op_name: str,
        module_path: str,
        stage: str,
        rank: int = 0,
        size_bytes: int = 0,
        flops: int = 0,
    ) -> ResourceEvent:
        start = self.current_time
        end = start + duration
        event = ResourceEvent(
            resource=self.resource,
            start_time=start,
            end_time=end,
            op_name=op_name,
            module_path=module_path,
            stage=stage,
            rank=rank,
            size_bytes=size_bytes,
            flops=flops,
        )
        self.events.append(event)
        self.current_time = end
        self.total_busy_time += duration
        return event

    def advance_to(self, t: float):
        if t > self.current_time:
            self.total_idle_time += t - self.current_time
            self.current_time = t

    @property
    def utilization(self) -> float:
        total = self.total_busy_time + self.total_idle_time
        if total <= 0:
            return 0.0
        return self.total_busy_time / total


@dataclass
class OverlapRecord:
    rank: int
    overlapped_event: ResourceEvent
    overlapping_event: ResourceEvent
    overlap_duration: float
    overlap_ratio: float


@dataclass
class ModuleOverlapStats:
    module_path: str
    fwd_compute_time: float = 0.0
    fwd_comm_time: float = 0.0
    fwd_overlapped_time: float = 0.0
    fwd_exposed_time: float = 0.0
    fwd_overlap_ratio: float = 0.0
    bwd_compute_time: float = 0.0
    bwd_comm_time: float = 0.0
    bwd_overlapped_time: float = 0.0
    bwd_exposed_time: float = 0.0
    bwd_overlap_ratio: float = 0.0

    def finalize(self):
        if self.fwd_comm_time > 0:
            self.fwd_exposed_time = self.fwd_comm_time - self.fwd_overlapped_time
            self.fwd_overlap_ratio = self.fwd_overlapped_time / self.fwd_comm_time
        if self.bwd_comm_time > 0:
            self.bwd_exposed_time = self.bwd_comm_time - self.bwd_overlapped_time
            self.bwd_overlap_ratio = self.bwd_overlapped_time / self.bwd_comm_time


@dataclass
class ResourceOverlapStats:
    resource: ResourceType
    total_time: float = 0.0
    busy_time: float = 0.0
    overlapped_time: float = 0.0
    utilization: float = 0.0


@dataclass
class CommOverlapStats:
    comm_type: str
    total_time: float = 0.0
    overlapped_time: float = 0.0
    exposed_time: float = 0.0
    overlap_ratio: float = 0.0

    def finalize(self):
        self.exposed_time = self.total_time - self.overlapped_time
        if self.total_time > 0:
            self.overlap_ratio = self.overlapped_time / self.total_time


@dataclass
class OverlapSummary:
    per_module: Dict[str, ModuleOverlapStats] = field(default_factory=dict)
    per_resource: Dict[ResourceType, ResourceOverlapStats] = field(
        default_factory=dict
    )
    per_comm_type: Dict[str, CommOverlapStats] = field(default_factory=dict)
    total_compute_time: float = 0.0
    total_comm_time: float = 0.0
    total_overlapped_comm_time: float = 0.0
    total_exposed_comm_time: float = 0.0
    overall_overlap_ratio: float = 0.0
    compute_utilization: float = 0.0
    intra_link_utilization: float = 0.0
    inter_link_utilization: float = 0.0
    iteration_time: float = 0.0


class OverlapTracker:

    def __init__(self):
        self._events_by_rank: Dict[int, List[ResourceEvent]] = {}

    def record_event(self, rank: int, event: ResourceEvent):
        self._events_by_rank.setdefault(rank, []).append(event)

    def record_events(self, rank: int, events: List[ResourceEvent]):
        self._events_by_rank.setdefault(rank, []).extend(events)

    def compute_overlap(self) -> OverlapSummary:
        summary = OverlapSummary()
        all_compute_time = 0.0
        all_comm_time = 0.0
        all_overlapped = 0.0

        for rank, events in self._events_by_rank.items():
            compute_events = [
                e for e in events if e.resource == ResourceType.COMPUTE
            ]
            comm_events = [
                e for e in events
                if e.resource in (ResourceType.INTRA_LINK, ResourceType.INTER_LINK)
            ]

            rank_compute = sum(e.end_time - e.start_time for e in compute_events)
            rank_comm = sum(e.end_time - e.start_time for e in comm_events)
            all_compute_time += rank_compute
            all_comm_time += rank_comm

            overlaps = self._compute_pairwise_overlap(compute_events, comm_events)
            rank_overlapped = sum(o.overlap_duration for o in overlaps)
            all_overlapped += rank_overlapped

            self._aggregate_per_module(rank, events, overlaps, summary)
            self._aggregate_per_comm_type(events, overlaps, summary)

        summary.total_compute_time = all_compute_time
        summary.total_comm_time = all_comm_time
        summary.total_overlapped_comm_time = all_overlapped
        summary.total_exposed_comm_time = all_comm_time - all_overlapped
        if all_comm_time > 0:
            summary.overall_overlap_ratio = all_overlapped / all_comm_time

        for path, stats in summary.per_module.items():
            stats.finalize()
        for ct, stats in summary.per_comm_type.items():
            stats.finalize()

        self._compute_resource_utilization(summary)
        return summary

    @staticmethod
    def _compute_pairwise_overlap(
        events_a: List[ResourceEvent], events_b: List[ResourceEvent]
    ) -> List[OverlapRecord]:
        if not events_a or not events_b:
            return []

        endpoints = []
        for i, e in enumerate(events_a):
            endpoints.append((e.start_time, +1, 0, i, e))
            endpoints.append((e.end_time, -1, 0, i, e))
        for i, e in enumerate(events_b):
            endpoints.append((e.start_time, +1, 1, i, e))
            endpoints.append((e.end_time, -1, 1, i, e))
        endpoints.sort(key=lambda x: (x[0], -x[1]))

        active_a = 0
        active_b = 0
        overlap_start = None
        results = []
        last_active_a_event = None
        last_active_b_event = None

        for t, delta, group, _idx, event in endpoints:
            was_overlapping = active_a > 0 and active_b > 0
            if group == 0:
                active_a += delta
                if delta > 0:
                    last_active_a_event = event
            else:
                active_b += delta
                if delta > 0:
                    last_active_b_event = event
            is_overlapping = active_a > 0 and active_b > 0

            if is_overlapping and not was_overlapping:
                overlap_start = t
            elif was_overlapping and not is_overlapping and overlap_start is not None:
                overlap_dur = t - overlap_start
                if overlap_dur > 0:
                    comm_ev = last_active_b_event
                    comp_ev = last_active_a_event
                    comm_total = comm_ev.end_time - comm_ev.start_time
                    ratio = overlap_dur / comm_total if comm_total > 0 else 0.0
                    results.append(OverlapRecord(
                        rank=0,
                        overlapped_event=comm_ev,
                        overlapping_event=comp_ev,
                        overlap_duration=overlap_dur,
                        overlap_ratio=min(ratio, 1.0),
                    ))
                overlap_start = None

        return results

    @staticmethod
    def _aggregate_per_module(
        rank: int,
        events: List[ResourceEvent],
        overlaps: List[OverlapRecord],
        summary: OverlapSummary,
    ):
        for e in events:
            path = e.module_path
            if path not in summary.per_module:
                summary.per_module[path] = ModuleOverlapStats(module_path=path)
            stats = summary.per_module[path]
            dur = e.end_time - e.start_time
            if e.resource == ResourceType.COMPUTE:
                if e.stage in ("fwd", "recompute"):
                    stats.fwd_compute_time += dur
                else:
                    stats.bwd_compute_time += dur
            elif e.resource in (ResourceType.INTRA_LINK, ResourceType.INTER_LINK):
                if e.stage in ("fwd", "recompute"):
                    stats.fwd_comm_time += dur
                else:
                    stats.bwd_comm_time += dur

        for ov in overlaps:
            path = ov.overlapped_event.module_path
            if path in summary.per_module:
                stats = summary.per_module[path]
                if ov.overlapped_event.stage in ("fwd", "recompute"):
                    stats.fwd_overlapped_time += ov.overlap_duration
                else:
                    stats.bwd_overlapped_time += ov.overlap_duration

    @staticmethod
    def _aggregate_per_comm_type(
        events: List[ResourceEvent],
        overlaps: List[OverlapRecord],
        summary: OverlapSummary,
    ):
        for e in events:
            if e.resource in (ResourceType.INTRA_LINK, ResourceType.INTER_LINK):
                ct = e.op_name
                if ct not in summary.per_comm_type:
                    summary.per_comm_type[ct] = CommOverlapStats(comm_type=ct)
                dur = e.end_time - e.start_time
                summary.per_comm_type[ct].total_time += dur

        for ov in overlaps:
            ct = ov.overlapped_event.op_name
            if ct in summary.per_comm_type:
                summary.per_comm_type[ct].overlapped_time += ov.overlap_duration

    @staticmethod
    def _compute_resource_utilization(summary: OverlapSummary):
        for path, stats in summary.per_module.items():
            pass
        total_compute = sum(
            s.fwd_compute_time + s.bwd_compute_time
            for s in summary.per_module.values()
        )
        total_intra = 0.0
        total_inter = 0.0
        for ct, stats in summary.per_comm_type.items():
            total_intra += stats.total_time
        summary.compute_utilization = 1.0 if total_compute > 0 else 0.0
        summary.intra_link_utilization = 1.0 if total_intra > 0 else 0.0
        summary.inter_link_utilization = 1.0 if total_inter > 0 else 0.0


class MultiResourceDES:

    def __init__(
        self,
        num_ranks: int = 1,
        resource_types: List[ResourceType] = None,
    ):
        self.num_ranks = num_ranks
        if resource_types is None:
            resource_types = [
                ResourceType.COMPUTE,
                ResourceType.INTRA_LINK,
                ResourceType.INTER_LINK,
            ]
        self.rank_resources: Dict[int, Dict[ResourceType, ResourceQueue]] = {}
        for r in range(num_ranks):
            self.rank_resources[r] = {
                res: ResourceQueue(resource=res) for res in resource_types
            }
        self.overlap_tracker = OverlapTracker()
        self._barrier_state: Dict[str, Dict] = {}

    def get_queue(self, rank: int, resource: ResourceType) -> ResourceQueue:
        return self.rank_resources[rank][resource]

    def schedule_compute(
        self,
        rank: int,
        duration: float,
        op_name: str,
        module_path: str,
        stage: str,
        flops: int = 0,
    ) -> ResourceEvent:
        q = self.get_queue(rank, ResourceType.COMPUTE)
        event = q.schedule(
            duration, op_name, module_path, stage,
            rank=rank, flops=flops,
        )
        self.overlap_tracker.record_event(rank, event)
        return event

    def schedule_intra_comm(
        self,
        ranks: List[int],
        duration: float,
        op_name: str,
        module_path: str,
        stage: str,
        size_bytes: int = 0,
    ):
        for r in ranks:
            q = self.get_queue(r, ResourceType.INTRA_LINK)
            event = q.schedule(
                duration, op_name, module_path, stage,
                rank=r, size_bytes=size_bytes,
            )
            self.overlap_tracker.record_event(r, event)

    def schedule_inter_comm(
        self,
        ranks: List[int],
        duration: float,
        op_name: str,
        module_path: str,
        stage: str,
        size_bytes: int = 0,
    ):
        for r in ranks:
            q = self.get_queue(r, ResourceType.INTER_LINK)
            event = q.schedule(
                duration, op_name, module_path, stage,
                rank=r, size_bytes=size_bytes,
            )
            self.overlap_tracker.record_event(r, event)

    # ---- Async P2P support ----

    def schedule_async_send(
        self,
        rank: int,
        gid: str,
        duration: float,
        op_name: str = "async_send",
        module_path: str = "",
        stage: str = "",
        size_bytes: int = 0,
    ):
        """Schedule an async send on rank's INTER_LINK.  Does NOT block COMPUTE."""
        q = self.get_queue(rank, ResourceType.INTER_LINK)
        event = q.schedule(
            duration, op_name, module_path, stage,
            rank=rank, size_bytes=size_bytes,
        )
        self.overlap_tracker.record_event(rank, event)
        return event

    def schedule_async_recv(
        self,
        rank: int,
        gid: str,
        duration: float,
        op_name: str = "async_recv",
        module_path: str = "",
        stage: str = "",
        size_bytes: int = 0,
    ):
        """Schedule an async recv on rank's INTER_LINK.  Does NOT block COMPUTE."""
        q = self.get_queue(rank, ResourceType.INTER_LINK)
        event = q.schedule(
            duration, op_name, module_path, stage,
            rank=rank, size_bytes=size_bytes,
        )
        self.overlap_tracker.record_event(rank, event)
        return event

    def schedule_async_wait(
        self,
        rank: int,
        gid: str,
        target_ms: float,
    ):
        """Block rank's COMPUTE lane until *target_ms* (when async P2P completes)."""
        comp_q = self.get_queue(rank, ResourceType.COMPUTE)
        comp_q.advance_to(target_ms)

    # ---- sync / utility ----

    def sync_rank_lanes(self, rank: int):
        res = self.rank_resources[rank]
        max_t = max(q.current_time for q in res.values())
        for q in res.values():
            q.advance_to(max_t)

    def compute_overlap(self) -> OverlapSummary:
        return self.overlap_tracker.compute_overlap()

    def get_iteration_time(self) -> float:
        max_t = 0.0
        for r in range(self.num_ranks):
            for q in self.rank_resources[r].values():
                max_t = max(max_t, q.current_time)
        return max_t

    def _events_by_rank(self) -> Dict[int, List[ResourceEvent]]:
        events: Dict[int, List[ResourceEvent]] = {}
        for rank, resources in self.rank_resources.items():
            rank_events: List[ResourceEvent] = []
            for q in resources.values():
                rank_events.extend(q.events)
            rank_events.sort(key=lambda e: e.start_time)
            events[rank] = rank_events
        return events

    def export_chrome_tracing(
        self,
        output_dir: Optional[str] = None,
        filename: str = "des_tracing_logs.json",
    ) -> str:
        """Export all DES events as Chrome Tracing JSON.

        Args:
            output_dir: Directory to write the file into.  Defaults to
                ``./output/YYYYMMDD_HHMMSS/`` (timestamped subdirectory
                under the current working directory).
            filename: Name of the output JSON file.

        Lane layout per rank (tid):
          - ``compute`` — COMPUTE resource events (fwd + bwd on one lane)
          - ``comm`` — INTRA_LINK / INTER_LINK events

        Returns the absolute path to the generated file.
        """
        _LANE_MAP = {
            ResourceType.COMPUTE: "compute",
            ResourceType.INTRA_LINK: "comm",
            ResourceType.INTER_LINK: "comm",
        }

        _LANE_SORT = {
            "compute": 0,
            "comm": 1,
        }

        def _resolve_lane(event: ResourceEvent) -> str:
            return _LANE_MAP.get(event.resource, "compute")

        # Resolve output directory
        if output_dir is None:
            ts_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.getcwd(), "output", ts_dir)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        tracing_events: List[dict] = []
        event_id = 0

        events_by_rank = self._events_by_rank()
        sorted_ranks = sorted(events_by_rank.keys())

        # -- metadata events --
        for proc_idx, rank in enumerate(sorted_ranks):
            pid = f"rank{rank}"
            tracing_events.append({
                "name": "process_name", "ph": "M", "pid": pid,
                "args": {"name": pid},
            })
            tracing_events.append({
                "name": "process_sort_index", "ph": "M", "pid": pid,
                "args": {"sort_index": proc_idx},
            })
            tids = sorted(
                {_resolve_lane(e) for e in events_by_rank[rank]},
                key=lambda t: _LANE_SORT.get(t, 99),
            )
            for tid in tids:
                tracing_events.append({
                    "name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                    "args": {"name": tid},
                })
                tracing_events.append({
                    "name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid,
                    "args": {"sort_index": _LANE_SORT.get(tid, 99)},
                })

        # -- duration events (ph="X") --
        for rank in sorted_ranks:
            pid = f"rank{rank}"
            for evt in events_by_rank[rank]:
                ts_us = evt.start_time * 1e3
                dur_us = max(0.0, (evt.end_time - evt.start_time) * 1e3)
                tid = _resolve_lane(evt)
                cat = "compute" if evt.resource == ResourceType.COMPUTE else "comm"
                tracing_events.append({
                    "name": evt.op_name,
                    "cat": cat,
                    "ph": "X",
                    "ts": ts_us,
                    "dur": dur_us,
                    "pid": pid,
                    "tid": tid,
                    "id": event_id,
                    "args": {
                        "module_path": evt.module_path,
                        "stage": evt.stage,
                        "resource": evt.resource.value,
                        "size_bytes": evt.size_bytes,
                        "flops": evt.flops,
                    },
                })
                event_id += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tracing_events, f, indent=4, ensure_ascii=False)

        event_count = sum(len(v) for v in events_by_rank.values())
        print(f"DES: exported {event_count} events ({len(sorted_ranks)} ranks) "
              f"→ {os.path.abspath(output_path)}")
        return os.path.abspath(output_path)
