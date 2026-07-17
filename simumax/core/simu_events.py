"""Structured simulation event stream.

Contract module for the simulate() DES rework (see
docs/design_simu_kind_resource_model.md). Phase 0 replaces the private
text log with an in-memory stream of SimuEvent objects; Phase 1 fills in
the classification fields (``kind``/``lane``) so the trace exporter can
stop guessing from name prefixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_RANK_PREFIX = re.compile(r"^rank(\d+)-")


@dataclass
class SimuEvent:
    """One completed op/phase span on a simulated rank.

    Time fields are milliseconds, matching the legacy log format.
    ``kind``/``lane`` are Phase-1 extensions; leave them None in Phase 0.
    """

    rank: int
    name: str                  # last call-stack segment (display name)
    call_stack: List[str]      # call-stack segments, rank prefix removed
    operation: str             # 'fwd' | 'bwd' | 'recompute_fwd'
    cost: float                # ms, == ed - st
    st: float                  # ms
    ed: float                  # ms
    gid: Optional[str] = None  # comm group/op id, None for pure compute
    post: Optional[float] = None   # ms, async p2p post timestamp
    order: Optional[int] = None    # async p2p post order
    stream: str = "comp"       # lane clock the span ran on
    kind: Optional[str] = None     # Phase 1: compute|comm|wait|scope|fused
    lane: Optional[str] = None     # Phase 1: explicit display lane


class EventSink:
    """Collects SimuEvents during a simulation run."""

    def __init__(self) -> None:
        self.events: List[SimuEvent] = []
        # Spans whose call_stk lacks a 'rank<N>-' prefix are dropped here,
        # counted but otherwise ignored. This mirrors the legacy behavior
        # where such log lines were written to log.log but silently dropped
        # by the trace parser (e.g. utility modules in function.py whose
        # queues never received a rank-prefixed call_stk).
        self.dropped = 0

    def emit_span(self, call_stk, operation, st, ed, gid=None, post=None,
                  order=None, stream="comp", kind=None, lane=None):
        """Append one span. ``call_stk`` keeps the legacy 'rankN-...' form."""
        match = _RANK_PREFIX.match(call_stk)
        if not match:
            self.dropped += 1
            return
        segments = call_stk[match.end():].split("-")
        self.events.append(SimuEvent(
            rank=int(match.group(1)),
            name=segments[-1],
            call_stack=segments,
            operation=operation,
            cost=ed - st,
            st=st,
            ed=ed,
            gid=gid,
            post=post,
            order=order,
            stream=stream,
            kind=kind,
            lane=lane,
        ))


def event_to_record(event: SimuEvent) -> dict:
    """Adapt a SimuEvent to the dict shape the trace converter consumes.

    Values are rounded to 6 decimals to reproduce the legacy text
    round-trip exactly.
    """
    return {
        "rank": f"rank{event.rank}",
        "call_stack": list(event.call_stack),
        "gid": event.gid,
        "operation": event.operation,
        "cost": round(event.cost, 6),
        "st": round(event.st, 6),
        "ed": round(event.ed, 6),
        "post": round(event.post, 6) if event.post is not None else None,
        "order": event.order,
        "stream": event.stream,
        "kind": event.kind,
        "lane": event.lane,
    }


def format_event_line(event: SimuEvent) -> str:
    """Render one event in the legacy log line format (debug artifact)."""
    call_stk = f"rank{event.rank}-" + "-".join(event.call_stack)
    gid_part = f" gid {event.gid}" if event.gid is not None else ""
    tail = ""
    if event.post is not None:
        tail += f" post {event.post:.6f}"
    if event.order is not None:
        tail += f" order {event.order}"
    return (f"{call_stk}{gid_part} {event.operation} "
            f"cost {event.cost:.6f} st {event.st:.6f} ed {event.ed:.6f}{tail}")


def write_debug_log(events, log_path) -> None:
    """Write the legacy text log from the event stream (one-way, debug only)."""
    with open(log_path, "w") as f:
        for event in events:
            f.write(format_event_line(event) + "\n")
