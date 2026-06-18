"""Pluggable CostModel interface for per-operator cost computation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable


@dataclass
class CostContext:
    op_name: str
    stage: str
    flops: int
    accessed_mem: int
    shape_desc: str
    element_size: int
    strategy: Any
    system: Any


@dataclass
class CostResult:
    compute_time: float
    mem_time: float
    end2end_time: float
    details: Dict[str, Any] = field(default_factory=dict)


class CostModel(ABC):

    @abstractmethod
    def compute(self, ctx: CostContext) -> CostResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class TableLookupCostModel(CostModel):

    def __init__(self, op_name: str = None, bandwidth_op_name: str = None):
        self._op_name = op_name
        self._bw_op_name = bandwidth_op_name

    @property
    def name(self) -> str:
        return "table_lookup"

    def compute(self, ctx: CostContext) -> CostResult:
        op_name = self._op_name or ctx.op_name
        bw_name = self._bw_op_name or ctx.op_name

        compute_detail = ctx.system.compute_op_accuracy_time(
            op_name, ctx.flops, ctx.shape_desc, reture_detail=True
        )
        mem_detail = ctx.system.compute_mem_access_time(
            bw_name, ctx.accessed_mem, reture_detail=True
        )
        e2e = ctx.system.compute_end2end_time(
            compute_detail['compute_only_time'],
            mem_detail['io_time'],
        )
        return CostResult(
            compute_time=compute_detail['compute_only_time'],
            mem_time=mem_detail['io_time'],
            end2end_time=e2e,
            details={"compute": compute_detail, "io": mem_detail},
        )


class FormulaCostModel(CostModel):

    def __init__(
        self,
        compute_fn: Callable[[CostContext], float],
        mem_fn: Callable[[CostContext], float] = None,
        combine_fn: Callable[[float, float], float] = None,
        model_name: str = "formula",
    ):
        self._compute_fn = compute_fn
        self._mem_fn = mem_fn or (lambda ctx: 0.0)
        self._combine_fn = combine_fn or (lambda c, m: max(c, m))
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, ctx: CostContext) -> CostResult:
        ct = self._compute_fn(ctx)
        mt = self._mem_fn(ctx)
        return CostResult(
            compute_time=ct,
            mem_time=mt,
            end2end_time=self._combine_fn(ct, mt),
            details={},
        )


class OverrideCostModel(CostModel):

    def __init__(self, fixed_time_ms: float, model_name: str = "override"):
        self._fixed_time = fixed_time_ms
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, ctx: CostContext) -> CostResult:
        return CostResult(
            compute_time=self._fixed_time,
            mem_time=0.0,
            end2end_time=self._fixed_time,
            details={"source": "fixed_override"},
        )


class MemoryAccessCostModel(CostModel):

    def __init__(self, bandwidth_op_name: str = "default"):
        self._bw_op_name = bandwidth_op_name

    @property
    def name(self) -> str:
        return "memory_access"

    def compute(self, ctx: CostContext) -> CostResult:
        mem_detail = ctx.system.compute_mem_access_time(
            self._bw_op_name, ctx.accessed_mem, reture_detail=True
        )
        mt = mem_detail['io_time']
        return CostResult(
            compute_time=0.0,
            mem_time=mt,
            end2end_time=mt,
            details={"io": mem_detail},
        )


class CostModelRegistry:
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, cost_model_cls: type):
        cls._registry[name] = cost_model_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> CostModel:
        if name not in cls._registry:
            raise KeyError(
                f"CostModel '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CostModel:
        config = dict(config)
        model_type = config.pop("type")
        return cls.create(model_type, **config)

    @classmethod
    def available(cls):
        return list(cls._registry.keys())


CostModelRegistry.register("table_lookup", TableLookupCostModel)
CostModelRegistry.register("formula", FormulaCostModel)
CostModelRegistry.register("override", OverrideCostModel)
CostModelRegistry.register("memory_access", MemoryAccessCostModel)
