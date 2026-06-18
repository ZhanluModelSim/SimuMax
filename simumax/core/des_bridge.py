"""Bridge between existing SimuMax paths and the multi-resource DES engine."""

import re
from typing import Dict, List, Optional, Any

from simumax.core.des_engine import (
    ResourceType,
    ResourceEvent,
    MultiResourceDES,
    OverlapTracker,
    OverlapSummary,
    ModuleOverlapStats,
)


_LOG_PATTERN = re.compile(
    r"(?P<call_stk>\S+)\s+"
    r"(?:gid\s+(?P<gid>\S+)\s+)?"
    r"(?P<phase>fwd|bwd|recompute_fwd|recompute_bwd)\s+"
    r"cost\s+(?P<cost>[\d.eE+-]+)\s+"
    r"st\s+(?P<st>[\d.eE+-]+)\s+"
    r"ed\s+(?P<ed>[\d.eE+-]+)"
)

_COMM_PATTERN = re.compile(
    r"(?P<call_stk>\S+)\s+"
    r"gid\s+(?P<gid>\S+)\s+"
    r"(?P<phase>fwd|bwd)\s+"
    r"cost\s+(?P<cost>[\d.eE+-]+)\s+"
    r"st\s+(?P<st>[\d.eE+-]+)\s+"
    r"ed\s+(?P<ed>[\d.eE+-]+)"
)


def _classify_resource(gid: str, call_stk: str) -> ResourceType:
    if gid is None:
        return ResourceType.COMPUTE
    if "send_recv" in gid or "pp" in gid.lower() or "default_group" in gid:
        return ResourceType.INTER_LINK
    if any(kw in gid for kw in ("all_reduce", "all_gather", "reduce_scatter", "all2all")):
        return ResourceType.INTRA_LINK
    return ResourceType.INTRA_LINK


def _classify_stage(phase: str) -> str:
    if "fwd" in phase:
        return "fwd"
    if "bwd" in phase:
        return "bwd"
    return "fwd"


def _extract_op_name(gid: str, call_stk: str) -> str:
    if gid is None:
        parts = call_stk.split("-")
        return parts[-1] if parts else call_stk
    for prefix in ("all_reduce", "all_gather", "reduce_scatter", "all2all", "p2p"):
        if gid.startswith(prefix):
            return prefix
    if "send_recv" in gid:
        return "p2p"
    return gid


class DesBridge:

    @staticmethod
    def from_simulation_log(
        log_path: str,
        num_ranks: int = 1,
    ) -> MultiResourceDES:
        des = MultiResourceDES(num_ranks=num_ranks)
        rank_events: Dict[int, List[ResourceEvent]] = {}

        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = _COMM_PATTERN.match(line)
                if m:
                    call_stk = m.group("call_stk")
                    gid = m.group("gid")
                    phase = m.group("phase")
                    cost = float(m.group("cost"))
                    st = float(m.group("st"))
                    ed = float(m.group("ed"))
                else:
                    m2 = _LOG_PATTERN.match(line)
                    if m2:
                        call_stk = m2.group("call_stk")
                        gid = m2.group("gid")
                        phase = m2.group("phase")
                        cost = float(m2.group("cost"))
                        st = float(m2.group("st"))
                        ed = float(m2.group("ed"))
                    else:
                        continue

                resource = _classify_resource(gid, call_stk)
                stage = _classify_stage(phase)
                op_name = _extract_op_name(gid, call_stk)
                module_path = call_stk.replace("-", ".")

                rank = 0
                event = ResourceEvent(
                    resource=resource,
                    start_time=st,
                    end_time=ed,
                    op_name=op_name,
                    module_path=module_path,
                    stage=stage,
                    rank=rank,
                )
                rank_events.setdefault(rank, []).append(event)

        for rank, events in rank_events.items():
            des.overlap_tracker.record_events(rank, events)

        return des

    @staticmethod
    def from_module_costs(
        perf_model: Any,
        num_ranks: int = 1,
    ) -> MultiResourceDES:
        des = MultiResourceDES(num_ranks=num_ranks)

        for chunk_name, model in perf_model.model_chunk_dict.items():
            if not hasattr(model, 'all_leaf_nodes'):
                continue
            for rank in range(num_ranks):
                DesBridge._schedule_model_leaves(
                    des, rank, model, chunk_name,
                )

        return des

    @staticmethod
    def _schedule_model_leaves(
        des: MultiResourceDES,
        rank: int,
        model: Any,
        chunk_name: str,
    ):
        if not hasattr(model, 'all_leaf_nodes'):
            return
        for leaf in model.all_leaf_nodes:
            cost_info = leaf._cost_info
            module_path = getattr(leaf, 'full_name', str(leaf))

            fwd_compute = cost_info.fwd_compute_time
            if fwd_compute > 0:
                des.schedule_compute(
                    rank, fwd_compute / 1e3,
                    op_name="compute",
                    module_path=module_path,
                    stage="fwd",
                )

            fwd_net = cost_info.fwd_net_time
            if fwd_net > 0:
                des.schedule_intra_comm(
                    [rank], fwd_net / 1e3,
                    op_name="tp_comm_fwd",
                    module_path=module_path,
                    stage="fwd",
                )

            bwd_compute = (
                cost_info.bwd_grad_act_time + cost_info.bwd_grad_w_time
            )
            if bwd_compute > 0:
                des.schedule_compute(
                    rank, bwd_compute / 1e3,
                    op_name="compute",
                    module_path=module_path,
                    stage="bwd",
                )

            bwd_net = (
                cost_info.bwd_grad_act_net_time + cost_info.bwd_grad_w_net_time
            )
            if bwd_net > 0:
                des.schedule_intra_comm(
                    [rank], bwd_net / 1e3,
                    op_name="tp_comm_bwd",
                    module_path=module_path,
                    stage="bwd",
                )

            recomp_compute = cost_info.recompute_compute_time
            if recomp_compute > 0:
                des.schedule_compute(
                    rank, recomp_compute / 1e3,
                    op_name="recompute",
                    module_path=module_path,
                    stage="recompute",
                )

            recomp_net = cost_info.recompute_net_time
            if recomp_net > 0:
                des.schedule_intra_comm(
                    [rank], recomp_net / 1e3,
                    op_name="tp_comm_recompute",
                    module_path=module_path,
                    stage="recompute",
                )


def backfill_exposed_times(
    perf_model: Any,
    overlap_summary: OverlapSummary,
):
    for chunk_name, model in perf_model.model_chunk_dict.items():
        if not hasattr(model, 'all_leaf_nodes'):
            continue
        for leaf in model.all_leaf_nodes:
            path = getattr(leaf, 'full_name', None)
            if path and path in overlap_summary.per_module:
                stats = overlap_summary.per_module[path]
                leaf._cost_info.fwd_net_exposed_time = (
                    stats.fwd_exposed_time * 1e3
                )
                leaf._cost_info.bwd_net_exposed_time = (
                    stats.bwd_exposed_time * 1e3
                )
