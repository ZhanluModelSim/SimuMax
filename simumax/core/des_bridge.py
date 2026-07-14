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
        """Build a multi-resource DES timeline from PerfLLM module costs.

        Supports both single and multi-micro-batch (1F1B) scheduling.

        Pipeline chunks are assigned to the correct rank subset:
          - rank 0..(tp-1)           → first_stage_chunk
          - rank tp..(2*tp-1)        → middle_stage_chunk  (pp > 2)
          - rank (pp-1)*tp..num-1    → last_stage_chunk

        For PP: uses ``calculate_1f1b_bubble`` to derive per-rank per-micro-batch
        start offsets (warmup → 1F1B → cooldown).
        """
        des = MultiResourceDES(num_ranks=num_ranks)

        pp_size = getattr(perf_model.strategy, 'pp_size', 1)
        tp_size = getattr(perf_model.strategy, 'tp_size', 1)
        mbc = getattr(perf_model.strategy, 'micro_batch_num', 1)
        ranks_per_stage = max(1, num_ranks // max(1, pp_size))

        # Sort chunks by stage order
        stage_order = ["first_stage_chunk", "middle_stage_chunk", "last_stage_chunk"]
        ordered_chunks = sorted(
            perf_model.model_chunk_dict.items(),
            key=lambda kv: (
                stage_order.index(kv[0]) if kv[0] in stage_order else 99
            ),
        )

        # Compute per-stage fwd / bwd times for 1F1B offset calculation
        def _chunk_total_time(model, pass_dir):
            if not hasattr(model, 'all_leaf_nodes'):
                return 0.0
            total = 0
            leaves = list(model.all_leaf_nodes)
            if pass_dir == "bwd":
                leaves = list(reversed(leaves))
            for leaf in leaves:
                ci = leaf._cost_info
                if pass_dir == "fwd":
                    t = ci.fwd_compute_time
                    if ci.fwd_net_time > 0:
                        t += ci.fwd_net_time
                else:
                    t = ci.bwd_grad_act_time + ci.bwd_grad_w_time
                    net = ci.bwd_grad_act_net_time + ci.bwd_grad_w_net_time
                    if net > 0:
                        t += net
                total += t
            return total / 1e3  # μs → ms

        forward_times = [
            _chunk_total_time(model, "fwd")
            for _, model in ordered_chunks
        ]
        backward_times = [
            _chunk_total_time(model, "bwd")
            for _, model in ordered_chunks
        ]
        # For PP>2, replicate the middle chunk so we have pp_size entries
        if pp_size > 2 and len(forward_times) >= 3:
            # ordered_chunks = [first, middle, last]
            # We need   [first, middle, ..., middle, last]
            mid_fwd = forward_times[1]
            mid_bwd = backward_times[1]
            last_fwd = forward_times[-1]
            last_bwd = backward_times[-1]
            forward_times = [forward_times[0]] + [mid_fwd] * (pp_size - 2) + [last_fwd]
            backward_times = [backward_times[0]] + [mid_bwd] * (pp_size - 2) + [last_bwd]

        # Compute PP P2P time (same formula as _compute_single_batch_phase_inputs)
        p2p_time = 0.0
        if pp_size > 1:
            pp_comm_size = _get_pp_p2p_size(perf_model)
            p2p_time = perf_model.system.compute_net_op_time(
                "p2p", pp_comm_size, 2,
                net=getattr(perf_model.strategy, 'pp_net', 'default'),
            ) / 1e3  # μs → ms

        # ---- 1F1B offset map (rank, mb, kind) → start_ms ----
        # Add p2p_time to forward/backward times so the 1F1B gap between
        # stages includes real P2P latency (otherwise offset gaps are 0).
        offset: Dict[tuple, float] = {}
        if pp_size > 1 and mbc >= 1:
            _, raw_schedules = perf_model.calculate_1f1b_bubble(
                pp_size, mbc,
                [t + p2p_time for t in forward_times],
                [t + p2p_time for t in backward_times],
                return_schedules=True,
            )
            for rank, rank_sched in enumerate(raw_schedules):
                for entry in rank_sched:
                    if entry["kind"] in ("F", "B"):
                        key = (rank, entry["mb"], entry["kind"])
                        offset[key] = entry["start"]

            # Log offsets
            print(f"\n[DES] 1F1B offsets (pp={pp_size}, mbc={mbc}, p2p={p2p_time*1e3:.1f}μs):")
            print(f"      fwd_times={[f'{t*1e3:.0f}' for t in forward_times]} μs")
            print(f"      bwd_times={[f'{t*1e3:.0f}' for t in backward_times]} μs")
            for rank in range(pp_size):
                items = sorted(
                    [(mb, k, v) for (r, mb, k), v in offset.items() if r == rank],
                    key=lambda x: x[2],
                )
                line = " ".join(f"mb{mb}:{k}={v*1e3:.0f}" for mb, k, v in items)
                print(f"      rank{rank}: {line} μs")
            print()

        # ---- Dispatch by strategy ----
        vp_size = getattr(perf_model, '_vp_size', None)
        if vp_size is not None:
            vp_size = vp_size()
        else:
            vp_size = 1
        if vp_size > 1 and pp_size > 1:
            return _build_vpp_des(perf_model, des, num_ranks, vp_size, pp_size,
                                  ranks_per_stage, mbc, p2p_time)
        elif pp_size > 1 and mbc >= 1:
            ops: List[tuple] = []
            for (rank, mb_1idx, kind), start_ms in offset.items():
                ops.append((start_ms, rank, mb_1idx - 1, kind))
            ops.sort(key=lambda x: x[0])

            print("[DES] Scheduled 1F1B operations:")
            for start_ms, rank, mb, kind in ops[:12]:
                print(f"       t={start_ms*1e3:.0f} μs  rank{rank}  mb{mb}  {kind}")
            if len(ops) > 12:
                print(f"       ... ({len(ops)} ops total)")
            print()

            is_async = getattr(perf_model.strategy, 'pp_comm_async', False)
            stage_pass_end: Dict[tuple, float] = {}  # (stage_idx, mb, kind) → end_ms

            for start_ms, rank, mb, kind in ops:
                stage_idx = rank  # 1F1B "rank" = pipeline stage index
                # Map stage_idx to the correct chunk (pp>2 uses middle for intermediate stages)
                if pp_size > 2 and stage_idx == 0:
                    chunk_idx = 0
                elif pp_size > 2 and stage_idx == pp_size - 1:
                    chunk_idx = len(ordered_chunks) - 1
                else:
                    chunk_idx = min(stage_idx, max(0, len(ordered_chunks) - 2))
                if chunk_idx >= len(ordered_chunks):
                    continue
                _, model = ordered_chunks[chunk_idx]
                pass_dir = "fwd" if kind == "F" else "bwd"
                gpu_ranks = list(range(
                    stage_idx * ranks_per_stage,
                    min((stage_idx + 1) * ranks_per_stage, num_ranks),
                ))

                # ---- Async P2P: inject send/recv/wait before dst compute ----
                if is_async and p2p_time > 0:
                    is_fwd = kind == "F"
                    # Determine src/dst: fwd → N→N+1, bwd → N+1→N
                    if is_fwd and stage_idx > 0:
                        prev_stage = stage_idx - 1
                        src_key = (prev_stage, mb, kind)
                        if src_key in stage_pass_end:
                            src_end = stage_pass_end[src_key]
                            t_p2p = max(src_end, start_ms - p2p_time)
                            prev_gpu_ranks = list(range(
                                prev_stage * ranks_per_stage,
                                stage_idx * ranks_per_stage,
                            ))
                            _inject_async_p2p(
                                des, prev_gpu_ranks, gpu_ranks,
                                t_p2p, p2p_time,
                                kind, mb, prev_stage, stage_idx,
                            )
                    elif not is_fwd and stage_idx < pp_size - 1:
                        next_stage = stage_idx + 1
                        src_key = (next_stage, mb, kind)
                        if src_key in stage_pass_end:
                            src_end = stage_pass_end[src_key]
                            t_p2p = max(src_end, start_ms - p2p_time)
                            next_gpu_ranks = list(range(
                                (stage_idx + 1) * ranks_per_stage,
                                min((stage_idx + 2) * ranks_per_stage, num_ranks),
                            ))
                            _inject_async_p2p(
                                des, next_gpu_ranks, gpu_ranks,
                                t_p2p, p2p_time,
                                kind, mb, stage_idx, next_stage,
                            )

                # ---- Schedule compute ----
                for gpu_rank in gpu_ranks:
                    _advance_all_lanes_to(des, gpu_rank, start_ms)
                    DesBridge._schedule_leaves_pass(
                        des, gpu_rank, model, pass_dir, mb=mb,
                    )
                    end_t = max(
                        q.current_time
                        for q in des.rank_resources[gpu_rank].values()
                    )
                    # Record per-(stage_idx, mb, kind) end time
                    prev = stage_pass_end.get((stage_idx, mb, kind), 0.0)
                    stage_pass_end[(stage_idx, mb, kind)] = max(prev, end_t)

            # ---- Sync P2P: still injected as a second pass ----
            if not is_async and p2p_time > 0:
                injections: List[tuple] = []
                for mb in range(mbc):
                    for kind in ("F", "B"):
                        for stage_idx in range(pp_size - 1):
                            src_rank = stage_idx * ranks_per_stage
                            dst_rank = (stage_idx + 1) * ranks_per_stage
                            if kind == "B":
                                src_rank, dst_rank = dst_rank, src_rank
                            src_key = (stage_idx, mb, kind)
                            dst_key = (stage_idx + 1, mb, kind)
                            if src_key not in stage_pass_end or dst_key not in stage_pass_end:
                                continue
                            src_end = stage_pass_end[src_key]
                            stage_dst = stage_idx + 1
                            if kind == "B":
                                stage_dst = stage_idx
                            dst_start = offset.get(
                                (stage_dst, mb + 1, kind), src_end + p2p_time
                            )
                            t_p2p = max(src_end, dst_start - p2p_time)
                            injections.append((
                                t_p2p, src_rank, dst_rank, mb, kind, stage_idx,
                            ))

                injections.sort(key=lambda x: x[0])
                for t_p2p, src_rank, dst_rank, mb, kind, stage_idx in injections:
                    _schedule_inter_comm_at(
                        des, src_rank, dst_rank, t_p2p, p2p_time,
                        f"pp_{kind.lower()}_s{stage_idx}_to_s{stage_idx+1}_mb{mb}",
                        f"{kind.lower()}_mb{mb}",
                    )
                print(f"[DES] Injected {len(injections)} PP P2P events (sync).\n")
        else:
            # TP-only: schedule all MBs sequentially per rank
            for mb in range(mbc):
                mb_1idx = mb + 1
                # ---- Forward pass: all stages, forward order ----
                for stage_idx, (chunk_name, model) in enumerate(ordered_chunks):
                    if not hasattr(model, 'all_leaf_nodes'):
                        continue
                    rank_start = stage_idx * ranks_per_stage
                    rank_end = min(rank_start + ranks_per_stage, num_ranks)
                    for rank in range(rank_start, rank_end):
                        if pp_size > 1:
                            t_offset = offset.get(
                                (rank, mb_1idx, "F"),
                                max(q.current_time for q in des.rank_resources[rank].values()),
                            )
                            _advance_all_lanes_to(des, rank, t_offset)
                        DesBridge._schedule_leaves_pass(
                            des, rank, model, "fwd", mb=mb,
                        )

                    # PP P2P: after stage fwd, send to next stage
                    if pp_size > 1 and stage_idx < len(ordered_chunks) - 1 and p2p_time > 0:
                        next_start = stage_idx + 1
                        for r in range(rank_start, rank_end):
                            des.schedule_inter_comm(
                                [r, min(r + ranks_per_stage, num_ranks - 1)],
                                p2p_time,
                                op_name="p2p_send_fwd",
                                module_path=f"pp_fwd_{stage_idx}_to_{next_start}",
                                stage=f"fwd_mb{mb}",
                            )

                # ---- Backward pass: all stages, reverse order ----
                for stage_idx in range(len(ordered_chunks) - 1, -1, -1):
                    chunk_name, model = ordered_chunks[stage_idx]
                    if not hasattr(model, 'all_leaf_nodes'):
                        continue
                    rank_start = stage_idx * ranks_per_stage
                    rank_end = min(rank_start + ranks_per_stage, num_ranks)
                    for rank in range(rank_start, rank_end):
                        if pp_size > 1:
                            t_offset = offset.get(
                                (rank, mb_1idx, "B"),
                                max(q.current_time for q in des.rank_resources[rank].values()),
                            )
                            _advance_all_lanes_to(des, rank, t_offset)
                        DesBridge._schedule_leaves_pass(
                            des, rank, model, "bwd", mb=mb,
                        )

                    # PP P2P: after stage bwd, send to previous stage
                    if pp_size > 1 and stage_idx > 0 and p2p_time > 0:
                        prev_start = stage_idx - 1
                        for r in range(rank_start, rank_end):
                            des.schedule_inter_comm(
                                [r, max(0, r - ranks_per_stage)],
                                p2p_time,
                                op_name="p2p_send_bwd",
                                module_path=f"pp_bwd_{stage_idx}_to_{prev_start}",
                                stage=f"bwd_mb{mb}",
                            )

        return des

    @staticmethod
    def _schedule_leaves_pass(
        des: MultiResourceDES,
        rank: int,
        model: Any,
        pass_dir: str,  # "fwd" or "bwd"
        mb: int = 0,
        allow_overlap: bool = False,
    ):
        """Schedule one pass (fwd or bwd) through all leaf nodes.

        Enforces DAG data dependencies:
          For each leaf with TP comm:
            compute → comm (cross-lane: comm waits for compute)
            comm    → next compute (next compute waits for this comm)

        When *allow_overlap* is True, a leaf whose type is in
        ``_OVERLAP_LEAF_TYPES`` (e.g. Swiglu, LayerNorm) will NOT
        block on the preceding all-reduce, allowing its compute to run
        in parallel with communication.
        """
        _OVERLAP_LEAF_TYPES = {"Swiglu", "LayerNorm"}
        if not hasattr(model, 'all_leaf_nodes'):
            return

        comp_q = des.get_queue(rank, ResourceType.COMPUTE)
        comm_q = des.get_queue(rank, ResourceType.INTRA_LINK)

        leaves = list(model.all_leaf_nodes)
        if pass_dir == "bwd":
            leaves = list(reversed(leaves))

        pending_comm_end = 0.0  # end time of the most recent comm

        for leaf in leaves:
            cost_info = leaf._cost_info
            module_path = getattr(leaf, 'full_name', str(leaf))
            op_name = type(leaf).__name__

            if pass_dir == "fwd":
                comp_time = cost_info.fwd_compute_time
                net_time = cost_info.fwd_net_time
            else:
                comp_time = (
                    cost_info.bwd_grad_act_time + cost_info.bwd_grad_w_time
                )
                net_time = (
                    cost_info.bwd_grad_act_net_time
                    + cost_info.bwd_grad_w_net_time
                )

            # Apply pending comm barrier only if this leaf depends on it
            blocked = False
            if pending_comm_end > comp_q.current_time:
                if not (allow_overlap and op_name in _OVERLAP_LEAF_TYPES):
                    comp_q.advance_to(pending_comm_end)
                    blocked = True
                    print(
                        f"  [BLOCK] {op_name:20s}  comp_q {comp_q.current_time*1e3 - comp_time/1e3:.0f}"
                        f" → {comp_q.current_time*1e3:.0f} μs"
                        f"  (waited {pending_comm_end*1e3 - (comp_q.current_time*1e3 - comp_time/1e3):.0f} μs"
                        f" for comm at {pending_comm_end*1e3:.0f} μs)"
                    )
                else:
                    print(
                        f"  [SKIP] {op_name:20s}  comp_q={comp_q.current_time*1e3:.0f} μs"
                        f"  ignoring pending_comm_end={pending_comm_end*1e3:.0f} μs"
                        f"  (overlap leaf)"
                    )
            elif pending_comm_end > 0:
                print(
                    f"  [SYNC] {op_name:20s}  comp_q already at {comp_q.current_time*1e3:.0f} μs"
                    f"  ≥ pending_comm_end={pending_comm_end*1e3:.0f} μs"
                )

            # 1. Schedule compute
            if comp_time > 0:
                des.schedule_compute(
                    rank, comp_time / 1e3,
                    op_name=op_name,
                    module_path=module_path,
                    stage=f"{pass_dir}_mb{mb}",
                )

            # 2. If this op has TP communication
            if net_time > 0:
                # Comm starts after this leaf's compute finishes
                prev_comm = comm_q.current_time
                comm_q.advance_to(comp_q.current_time)
                des.schedule_intra_comm(
                    [rank], net_time / 1e3,
                    op_name=f"{op_name}_allreduce",
                    module_path=module_path,
                    stage=f"{pass_dir}_mb{mb}",
                )
                # Defer barrier: record comm end; next leaf decides whether to wait
                pending_comm_end = comm_q.current_time
                print(
                    f"  [COMM] {op_name:20s}  compute_end={comp_q.current_time*1e3:.0f} μs"
                    f"  comm: {prev_comm*1e3:.0f} → {comm_q.current_time*1e3:.0f} μs"
                    f"  pending_comm_end={pending_comm_end*1e3:.0f} μs"
                    f"  {'BLOCKED' if blocked else ''}"
                )
            else:
                if comp_time > 0:
                    print(
                        f"  [COMP] {op_name:20s}  comp_q → {comp_q.current_time*1e3:.0f} μs"
                        f"  {'BLOCKED' if blocked else ''}"
                    )
                pending_comm_end = 0.0


def _advance_all_lanes_to(des: MultiResourceDES, rank: int, t: float):
    """Advance every resource lane of *rank* to at least *t* (ms)."""
    for q in des.rank_resources[rank].values():
        q.advance_to(t)


def _build_vpp_des(
    perf_model: Any,
    des: MultiResourceDES,
    num_ranks: int,
    vp_size: int,
    pp_size: int,
    ranks_per_stage: int,
    mbc: int,
    p2p_time: float,
) -> MultiResourceDES:
    """Build DES timeline for VPP interleaved scheduling.

    Uses ``_compute_interleaved_sync_schedule`` to derive the schedule
    table, then maps each (rank, virtual_chunk, mb, kind) tuple to DES events.
    """
    # Build physical chunk lookup by stage index
    stage_order = ["first_stage_chunk", "middle_stage_chunk", "last_stage_chunk"]
    ordered_chunks = sorted(
        perf_model.model_chunk_dict.items(),
        key=lambda kv: stage_order.index(kv[0]) if kv[0] in stage_order else 99,
    )

    # Build virtual chunk lookup: (pp_stage_idx, virtual_idx) → model
    vpp_chunks: Dict[tuple, Any] = {}
    if hasattr(perf_model, 'vpp_chunk_dict') and perf_model.vpp_chunk_dict:
        for chunk_name, model in perf_model.vpp_chunk_dict.items():
            # chunk_name format: "first_stage_chunk_v0"
            for stage_key in stage_order:
                if chunk_name.startswith(stage_key):
                    stage_idx = stage_order.index(stage_key)
                    suffix = chunk_name[len(stage_key):]
                    if suffix.startswith("_v"):
                        v_idx = int(suffix[2:])
                        vpp_chunks[(stage_idx, v_idx)] = model
                        break

    if not vpp_chunks:
        print("[DES] VPP: no virtual chunks found, falling back to physical chunks.")
        return des

    print(f"[DES] VPP: vp={vp_size}, pp={pp_size}, mbc={mbc}, "
          f"virtual_chunks={len(vpp_chunks)}")

    # Get VPP schedule from PerfLLM
    try:
        raw_schedules = _get_vpp_schedule(perf_model, pp_size, vp_size, mbc,
                                          ordered_chunks, p2p_time)
    except Exception as exc:
        print(f"[DES] VPP schedule generation failed: {exc}")
        print("[DES] VPP: falling back to physical-chunk 1F1B.")
        return des

    # Map schedule to DES events, interleaving PP P2P
    is_async = getattr(perf_model.strategy, 'pp_comm_async', False)
    stage_pass_end: Dict[tuple, float] = {}  # (stage_idx, mb, kind) → end_ms

    for rank in range(pp_size):  # 1F1B "rank" = PP stage index
        gpu_ranks = list(range(
            rank * ranks_per_stage,
            min((rank + 1) * ranks_per_stage, num_ranks),
        ))
        for entry in raw_schedules[rank]:
            kind = entry["kind"]
            if kind not in ("F", "B"):
                continue
            mb = entry["mb"] - 1  # 1-indexed → 0-indexed
            start_ms = entry["start"]

            # ---- Async P2P injection before dst stage compute ----
            if is_async and p2p_time > 0:
                is_fwd = kind == "F"
                if is_fwd and rank > 0:
                    src_key = (rank - 1, mb, kind)
                    if src_key in stage_pass_end:
                        src_end = stage_pass_end[src_key]
                        t_p2p = max(src_end, start_ms - p2p_time)
                        prev_ranks = list(range(
                            (rank - 1) * ranks_per_stage,
                            rank * ranks_per_stage,
                        ))
                        _inject_async_p2p(
                            des, prev_ranks, gpu_ranks,
                            t_p2p, p2p_time,
                            kind, mb, rank - 1, rank,
                        )
                elif not is_fwd and rank < pp_size - 1:
                    src_key = (rank + 1, mb, kind)
                    if src_key in stage_pass_end:
                        src_end = stage_pass_end[src_key]
                        t_p2p = max(src_end, start_ms - p2p_time)
                        next_ranks = list(range(
                            (rank + 1) * ranks_per_stage,
                            min((rank + 2) * ranks_per_stage, num_ranks),
                        ))
                        _inject_async_p2p(
                            des, next_ranks, gpu_ranks,
                            t_p2p, p2p_time,
                            kind, mb, rank, rank + 1,
                        )

            # Maps stage_idx to the correct chunk
            pass_dir = "fwd" if kind == "F" else "bwd"
            if pp_size > 2 and rank == 0:
                chunk_idx = 0
            elif pp_size > 2 and rank == pp_size - 1:
                chunk_idx = len(ordered_chunks) - 1
            else:
                chunk_idx = min(rank, max(0, len(ordered_chunks) - 2))
            _, model = ordered_chunks[chunk_idx]

            for gpu_rank in gpu_ranks:
                _advance_all_lanes_to(des, gpu_rank, start_ms)
                DesBridge._schedule_leaves_pass(
                    des, gpu_rank, model, pass_dir, mb=mb,
                )
                end_t = max(
                    q.current_time for q in des.rank_resources[gpu_rank].values()
                )
                prev = stage_pass_end.get((rank, mb, kind), 0.0)
                stage_pass_end[(rank, mb, kind)] = max(prev, end_t)

    # ---- Sync P2P injection ----
    if not is_async and p2p_time > 0:
        injections: List[tuple] = []
        for mb in range(mbc):
            for kind in ("F", "B"):
                for stage_idx in range(pp_size - 1):
                    src_rank = stage_idx * ranks_per_stage
                    dst_rank = (stage_idx + 1) * ranks_per_stage
                    if kind == "B":
                        src_rank, dst_rank = dst_rank, src_rank
                    src_key = (stage_idx, mb, kind)
                    dst_key = (stage_idx + 1, mb, kind)
                    if src_key not in stage_pass_end or dst_key not in stage_pass_end:
                        continue
                    src_end = stage_pass_end[src_key]
                    t_p2p = src_end
                    injections.append((t_p2p, src_rank, dst_rank, mb, kind, stage_idx))
        injections.sort(key=lambda x: x[0])
        for t_p2p, src_rank, dst_rank, mb, kind, stage_idx in injections:
            _schedule_inter_comm_at(
                des, src_rank, dst_rank, t_p2p, p2p_time,
                f"pp_{kind.lower()}_s{stage_idx}_to_s{stage_idx+1}_mb{mb}",
                f"{kind.lower()}_mb{mb}",
            )
        print(f"[DES] VPP: Injected {len(injections)} PP P2P events (sync).\n")

    print(f"[DES] VPP: scheduled {len(raw_schedules)} stages.\n")
    return des


def _get_vpp_schedule(
    perf_model: Any,
    pp_size: int,
    vp_size: int,
    mbc: int,
    ordered_chunks: List[tuple],
    p2p_time: float,
) -> List[List[dict]]:
    """Generate VPP 1F1B schedule using the interleaved sync scheduler.

    Returns:
        List of per-rank schedule dicts with keys: kind, mb, start, duration, end,
        chunk_idx, virtual_idx.
    """
    # Compute per-(stage, virtual) fwd/bwd times
    def _vpp_chunk_time(stage_idx, v_idx, pass_dir):
        chunk_name = list(ordered_chunks)[stage_idx][0] if stage_idx < len(ordered_chunks) else ""
        vpp_name = f"{chunk_name}_v{v_idx}"
        model = perf_model.vpp_chunk_dict.get(vpp_name) if hasattr(perf_model, 'vpp_chunk_dict') else None
        if model is None and stage_idx < len(ordered_chunks):
            model = ordered_chunks[stage_idx][1]
        if model is None:
            return 0.0
        total = 0
        leaves = list(model.all_leaf_nodes) if hasattr(model, 'all_leaf_nodes') else []
        if pass_dir == "bwd":
            leaves = list(reversed(leaves))
        for leaf in leaves:
            ci = leaf._cost_info
            if pass_dir == "fwd":
                t = ci.fwd_compute_time
                if ci.fwd_net_time > 0:
                    t += ci.fwd_net_time
            else:
                t = ci.bwd_grad_act_time + ci.bwd_grad_w_time
                net = ci.bwd_grad_act_net_time + ci.bwd_grad_w_net_time
                if net > 0:
                    t += net
            total += t
        return total / 1e3  # μs → ms

    # Build forward/backward times per stage (sum over virtual chunks)
    fwd_times = []
    bwd_times = []
    for stage_idx in range(pp_size):
        f_total = sum(_vpp_chunk_time(stage_idx, v, "fwd") for v in range(vp_size))
        b_total = sum(_vpp_chunk_time(stage_idx, v, "bwd") for v in range(vp_size))
        fwd_times.append(f_total + p2p_time)
        bwd_times.append(b_total + p2p_time)

    # Use calculate_1f1b_bubble for the outer pipeline schedule
    _, raw_schedules = perf_model.calculate_1f1b_bubble(
        pp_size, mbc, fwd_times, bwd_times, return_schedules=True,
    )
    return raw_schedules


def _inject_async_p2p(
    des: MultiResourceDES,
    src_ranks: List[int],
    dst_ranks: List[int],
    t_p2p: float,
    p2p_time: float,
    kind: str,
    mb: int,
    src_stage: int,
    dst_stage: int,
):
    """Inject async P2P send→recv→wait between two stages, inline with compute.

    - async_send on src rank's INTER_LINK at t_p2p
    - async_recv on dst rank's INTER_LINK at t_p2p
    - COMPUTE lanes of dst ranks blocked until t_p2p + p2p_time
    """
    module_path = f"pp_{kind.lower()}_s{src_stage}_to_s{dst_stage}_mb{mb}"
    stage_label = f"{kind.lower()}_mb{mb}"
    wait_t = t_p2p + p2p_time

    for r in src_ranks:
        _schedule_event_at(
            des, r, ResourceType.INTER_LINK,
            t_p2p, p2p_time, "async_send",
            module_path, stage_label,
        )
    for r in dst_ranks:
        _schedule_event_at(
            des, r, ResourceType.INTER_LINK,
            t_p2p, p2p_time, "async_recv",
            module_path, stage_label,
        )
        # Block COMPUTE lane on dst until P2P completes
        comp_q = des.get_queue(r, ResourceType.COMPUTE)
        comp_q.advance_to(wait_t)

    print(f"  [ASYNCP2P] {kind} mb{mb} s{src_stage}→s{dst_stage}: "
          f"t={t_p2p*1e3:.0f}μs wait@{wait_t*1e3:.0f}μs "
          f"(src_ranks={src_ranks} dst_ranks={dst_ranks})")


def _schedule_event_at(
    des: MultiResourceDES,
    rank: int,
    resource: ResourceType,
    start_ms: float,
    duration_ms: float,
    op_name: str,
    module_path: str,
    stage: str,
):
    """Insert a ResourceEvent at a specific absolute time on a single rank's queue."""
    q = des.get_queue(rank, resource)
    event = ResourceEvent(
        resource=resource,
        start_time=start_ms,
        end_time=start_ms + duration_ms,
        op_name=op_name,
        module_path=module_path,
        stage=stage,
        rank=rank,
    )
    q.events.append(event)
    q.current_time = max(q.current_time, start_ms + duration_ms)
    des.overlap_tracker.record_event(rank, event)


def _schedule_inter_comm_at(
    des: MultiResourceDES,
    src_rank: int,
    dst_rank: int,
    start_ms: float,
    duration_ms: float,
    module_path: str,
    stage: str,
):
    """Schedule a P2P comm event on both ranks at a specific absolute time."""
    for r in (src_rank, dst_rank):
        q = des.get_queue(r, ResourceType.INTER_LINK)
        event = ResourceEvent(
            resource=ResourceType.INTER_LINK,
            start_time=start_ms,
            end_time=start_ms + duration_ms,
            op_name="p2p_send_recv",
            module_path=module_path,
            stage=stage,
            rank=r,
        )
        q.events.append(event)
        q.current_time = max(q.current_time, start_ms + duration_ms)
        des.overlap_tracker.record_event(r, event)


def _get_pp_p2p_size(perf_model: Any) -> int:
    """Estimate PP P2P communication size in bytes per micro-batch."""
    strategy = perf_model.strategy
    hidden_size = getattr(perf_model.model_config, 'hidden_size', 4096)
    seq_len = getattr(strategy, 'seq_len', 4096)
    micro_batch_size = getattr(strategy, 'micro_batch_size', 1)
    dtype = getattr(strategy, 'dtype', 'bf16')
    element_size = {'bf16': 2, 'fp16': 2, 'fp32': 4}.get(dtype, 2)
    return seq_len * micro_batch_size * hidden_size * element_size


def _sync_ranks_to_max(
    des: MultiResourceDES,
    target_ranks: range,
    source_ranks: range,
):
    """Advance target ranks' lanes to the max lane time of source ranks."""
    src_max = 0.0
    for r in source_ranks:
        for q in des.rank_resources[r].values():
            src_max = max(src_max, q.current_time)
    for r in target_ranks:
        for q in des.rank_resources[r].values():
            q.advance_to(src_max)


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
