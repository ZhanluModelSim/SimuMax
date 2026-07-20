"""Cost-spec registry and declarative block templates.

Implements the Phase 3 (v1) scope of the cost-model tunability design
(``docs/design_simu_cost_model_tunability.md``, section 6).

In v1 the registry is **metadata only**: it maps a template name to a
module family plus a description of the family's cost-op bindings. The
flops/bytes formulas still live in the module classes under
``simumax/core/transformer/``; relocating those formulas into the specs
(so new ops register specs instead of subclassing) is documented future
work and is intentionally not attempted here.

The templates back the optional declarative ``recipe`` section of the
model JSON (see ``ModelConfig.apply_recipe``), which expands a block
list into ``layer_num`` / ``dense_layers`` — the same composition that
hand-written model configs express today.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class BlockTemplate:
    """Metadata for one composable transformer block template.

    Attributes:
        name: template name as used in a model JSON ``recipe``.
        family: module family, ``"dense"`` or ``"moe"``. Dense blocks build
            the dense prefix of ``LLMModel`` (``use_dense=True``), MoE blocks
            the remaining layers (``use_dense=False``).
        op_bindings: the family's cost-op bindings, i.e. which cost operator
            each module role resolves against (metadata only in v1).
        description: human-readable summary of the template.
    """

    name: str
    family: str
    op_bindings: Dict[str, str]
    description: str = ""


BLOCK_TEMPLATES: Dict[str, BlockTemplate] = {
    "DenseLLMBlock": BlockTemplate(
        name="DenseLLMBlock",
        family="dense",
        op_bindings={
            "attention": "sdp_fwd/sdp_bwd",
            "linear": "matmul",
        },
        description=(
            "Standard dense transformer layer (self-attention + dense MLP), "
            "as built by LLMModel with use_dense=True."
        ),
    ),
    "MoELLMBlock": BlockTemplate(
        name="MoELLMBlock",
        family="moe",
        op_bindings={
            "attention": "sdp_fwd/sdp_bwd",
            "linear": "matmul",
            "experts": "group_matmul",
            "dispatch_combine": "all2all",
        },
        description=(
            "MoE transformer layer (self-attention + routed experts with "
            "all2all dispatch/combine), as built by LLMModel with "
            "use_dense=False."
        ),
    ),
}


def get_block_template(name: str) -> BlockTemplate:
    """Look up a block template by name.

    Raises:
        ValueError: if ``name`` is not registered in ``BLOCK_TEMPLATES``.
    """
    try:
        return BLOCK_TEMPLATES[name]
    except KeyError:
        available = ", ".join(sorted(BLOCK_TEMPLATES))
        raise ValueError(
            f"Unknown block template {name!r}. Available templates: {available}"
        ) from None
