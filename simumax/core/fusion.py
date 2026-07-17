"""Fusion policies for fused (multi-resource) operators.

Contract module for Phase 3 of docs/design_simu_kind_resource_model.md.
A fusion policy decides, given the per-lane busy costs of a fused op,
the op's total span and how long each occupied resource lane stays busy.
"""

from __future__ import annotations

from typing import Dict


class FusionPolicy:
    """Base class for fused-op duration composition."""

    name = "base"

    def span(self, costs: Dict[str, float]) -> float:
        """Total span of the fused op (ms)."""
        raise NotImplementedError

    def lane_durations(self, costs: Dict[str, float]) -> Dict[str, float]:
        """Busy duration per occupied resource lane (ms)."""
        raise NotImplementedError


class Serial(FusionPolicy):
    """Resources are occupied one after another (legacy serial behavior)."""

    name = "serial"

    def span(self, costs):
        return sum(costs.values())

    def lane_durations(self, costs):
        return dict(costs)


class MaxOverlap(FusionPolicy):
    """All lanes busy for the whole span; span = slowest resource."""

    name = "max_overlap"

    def span(self, costs):
        return max(costs.values()) if costs else 0.0

    def lane_durations(self, costs):
        span = self.span(costs)
        return {lane: span for lane in costs}


class ChunkedPipeline(FusionPolicy):
    """Chunked compute-comm pipeline (e.g. AG+GEMM split into N chunks).

    span = slowest lane + fastest lane / chunks (fill/drain bubble);
    each lane stays busy for its own cost. The formula is documented for
    the 2-lane case (one engine lane + one comm lane); for more lanes it
    uses the slowest/fastest pair as the critical path approximation.
    """

    name = "chunked_pipeline"

    def __init__(self, chunks: int = 1):
        assert int(chunks) >= 1, "chunks must be >= 1"
        self.chunks = int(chunks)

    def span(self, costs):
        if not costs:
            return 0.0
        return max(costs.values()) + min(costs.values()) / self.chunks

    def lane_durations(self, costs):
        return dict(costs)


FUSION_POLICIES = {
    Serial.name: Serial,
    MaxOverlap.name: MaxOverlap,
    ChunkedPipeline.name: ChunkedPipeline,
}


def build_fusion_policy(spec) -> FusionPolicy:
    """Build a policy from a name or a dict spec.

    Accepted forms: "serial" | "max_overlap" | "chunked_pipeline", or
    {"policy": <name>, "chunks": <n>} (chunks only for chunked_pipeline).
    Raises KeyError for unknown policy names and AssertionError for
    unexpected kwargs; config validation reports these to the user.
    """
    if isinstance(spec, str):
        name, kwargs = spec, {}
    else:
        name = spec.get("policy", ChunkedPipeline.name)
        kwargs = {k: v for k, v in spec.items() if k != "policy"}
    cls = FUSION_POLICIES[name]
    if cls is ChunkedPipeline:
        return cls(chunks=kwargs.get("chunks", 1))
    assert not kwargs, f"policy {name!r} takes no parameters, got {kwargs}"
    return cls()
