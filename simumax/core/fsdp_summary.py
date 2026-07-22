"""FSDP communication summary extracted from DES trace events.

Post-processes ``tracing_logs.json`` to aggregate per-block and total
all-gather / reduce-scatter durations, exposed (stall) time, and overlap
statistics for layer-wise FSDP. Called by ``simu_runner.run_simulation``
after the trace file is written.
"""
import json
import os


def _is_fsdp_event(event, kind_substr):
    """Check if a trace event is an FSDP AG/RS event by its gid or base_name."""
    args = event.get("args", {})
    gid = str(args.get("gid", ""))
    base_name = str(args.get("base_name", ""))
    return kind_substr in gid or kind_substr in base_name


def summarize_fsdp_trace(trace_path, save_path=None):
    """Extract FSDP AG/RS summary from a Chrome trace file.

    Parameters
    ----------
    trace_path : str
        Path to ``tracing_logs.json``.
    save_path : str, optional
        Directory to write ``fsdp_summary.json``. If None, only returns the dict.

    Returns
    -------
    dict or None
        The summary dict, or None if no FSDP events were found.
    """
    with open(trace_path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    else:
        events = payload

    # Collect FSDP comm completion spans and wait stalls.
    # Comm spans: cat="comm", gid contains "fsdp_ag" or "fsdp_rs".
    #   These are the AG/RS activity durations on the dp_comm lane.
    # Wait stalls: cat="wait", non-zero dur.
    #   These are the exposed (non-overlapped) comm time on the comp lane.
    ag_comm = []       # (rank, block_tag, dur)
    bwd_ag_comm = []    # backward AG (FULL_SHARD)
    rs_comm = []        # reduce-scatter
    wait_stalls = []    # (rank, tag, dur) — exposed time

    for e in events:
        if e.get("ph") != "X":
            continue
        args = e.get("args", {})
        gid = str(args.get("gid", ""))
        base_name = str(args.get("base_name", ""))
        cat = e.get("cat", "")
        dur = e.get("dur", 0)
        rank = e.get("pid", "")

        # FSDP comm completion spans (on dp_comm lane)
        if cat == "comm" and "fsdp_" in gid:
            if "fsdp_bwd_ag" in gid or "fsdp_bwd_ag" in base_name:
                bwd_ag_comm.append((rank, base_name, dur))
            elif "fsdp_ag" in gid or "fsdp_ag" in base_name:
                ag_comm.append((rank, base_name, dur))
            elif "fsdp_rs" in gid or "fsdp_rs" in base_name:
                rs_comm.append((rank, base_name, dur))

        # Wait stalls (exposed comm time on comp lane)
        if cat == "wait" and dur > 0.01:
            call_stk = str(args.get("call_stack", ""))
            if "fsdp" in call_stk or "fsdp" in base_name:
                wait_stalls.append((rank, base_name, dur))

    # If no FSDP events, return None
    total_ag = sum(d for _, _, d in ag_comm)
    total_bwd_ag = sum(d for _, _, d in bwd_ag_comm)
    total_rs = sum(d for _, _, d in rs_comm)
    total_comm = total_ag + total_bwd_ag + total_rs
    total_exposed = sum(d for _, _, d in wait_stalls)
    overlap_time = max(0.0, total_comm - total_exposed)
    overlap_pct = (overlap_time / total_comm * 100.0) if total_comm > 0 else 0.0

    if total_comm == 0 and total_exposed == 0:
        return None

    # Per-rank breakdown
    ranks = sorted(set(
        [r for r, _, _ in ag_comm]
        + [r for r, _, _ in bwd_ag_comm]
        + [r for r, _, _ in rs_comm]
        + [r for r, _, _ in wait_stalls]
    ))

    per_rank = {}
    for rank in ranks:
        r_ag = sum(d for r, _, d in ag_comm if r == rank)
        r_bwd_ag = sum(d for r, _, d in bwd_ag_comm if r == rank)
        r_rs = sum(d for r, _, d in rs_comm if r == rank)
        r_exposed = sum(d for r, _, d in wait_stalls if r == rank)
        r_total = r_ag + r_bwd_ag + r_rs
        r_overlap = max(0.0, r_total - r_exposed)
        per_rank[rank] = {
            "fwd_ag_time": r_ag,
            "bwd_ag_time": r_bwd_ag,
            "rs_time": r_rs,
            "total_comm_time": r_total,
            "exposed_time": r_exposed,
            "overlap_time": r_overlap,
            "overlap_percentage": (r_overlap / r_total * 100.0) if r_total > 0 else 0.0,
        }

    summary = {
        "total": {
            "fwd_ag_time": total_ag,
            "bwd_ag_time": total_bwd_ag,
            "rs_time": total_rs,
            "total_comm_time": total_comm,
            "exposed_time": total_exposed,
            "overlap_time": overlap_time,
            "overlap_percentage": overlap_pct,
            "fwd_ag_count": len(ag_comm),
            "bwd_ag_count": len(bwd_ag_comm),
            "rs_count": len(rs_comm),
            "wait_stall_count": len(wait_stalls),
        },
        "per_rank": per_rank,
    }

    if save_path is not None:
        out_file = os.path.join(save_path, "fsdp_summary.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    return summary
