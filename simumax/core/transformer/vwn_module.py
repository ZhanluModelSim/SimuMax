"""VWN (Variable Window Network) Roofline Cost Model.

Based on op_define.md MojoVWNInOp (L119-169) and MojoVWNOutOp (L941-964).

VWN is a gating / stream-mixing network that replaces the standard RMSNorm
in selected transformer layers. It consists of two operators:

  VWNInOp  — input side: RMSNorm + gating GEMM + tanh + stream mixing
  VWNOutOp — output side: B-gate mixing + residual add

Parameters (from op_define L126-131):
  C      = hidden size
  VWN_N  = residual streams count
  VWN_M  = block output streams count
  V      = C / VWN_M   (per-stream hidden dim)
  A      = VWN_N + VWN_M
  G      = VWN_N + 2 * VWN_M
"""

from simumax.core.base_struct import MetaModule, InputOutputInfo
from simumax.core.config import StrategyConfig, SystemConfig


class VWNInModule(MetaModule):
    """VWN Input Gating Operator.

    Corresponds to MojoVWNInOp in op_define.md (L119-169).

    Fuses RMSNorm + gating GEMM + tanh + stream mixing into a single
    roofline-modeled operator.

    Input:
      x[T, VWN_N, V]  —  V = C / VWN_M
    Outputs:
      block_input[T, VWN_M, V]  — feeds into main transformer block
      residual[T, VWN_N, V]     — residual bypass for VWNOut
      b_gate[T, VWN_N, VWN_M]   — gate values for VWNOut
    """

    def __init__(
        self,
        hidden_size: int,
        vwn_n: int,
        vwn_m: int,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name: str = 'VWNIn',
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.hidden_size = hidden_size   # C
        self.vwn_n = vwn_n               # residual streams
        self.vwn_m = vwn_m               # block output streams
        assert hidden_size % vwn_m == 0, (
            f"hidden_size({hidden_size}) must be divisible by vwn_m({vwn_m})"
        )
        self.V = hidden_size // vwn_m     # per-stream dim
        self.A = vwn_n + vwn_m           # A gate dimension
        self.G = vwn_n + 2 * vwn_m       # G gate dimension

    @property
    def _token_count(self) -> int:
        """Total token count T from input shape [T, VWN_N, V]."""
        input_tensor = self.input_info.tensors[0]
        if len(input_tensor.shape) == 3:
            return input_tensor.size(0)
        # 2D fallback: [T, VWN_N * V] → reshape implied
        return input_tensor.size(0)

    def create_output_info(self):
        # VWNIn produces three outputs; the primary output is block_input
        # with shape [T, VWN_M, V]
        T = self._token_count
        return InputOutputInfo(
            tensors=[
                __import__('simumax.core.tensor', fromlist=['TensorSize']).TensorSize(
                    shape=(T, self.vwn_m, self.V)
                )
            ]
        )

    def get_input_shapes_desc(self, stage: str) -> str:
        T = self._token_count
        return (
            f'T={int(T)}, C={int(self.hidden_size)}, '
            f'VWN_N={int(self.vwn_n)}, VWN_M={int(self.vwn_m)}, '
            f'V={int(self.V)}, A={int(self.A)}, G={int(self.G)}'
        )

    # ───  Method 1: FLOPS (source: op_define L154-156)  ───
    def _comp_leaf_flops_info(self):
        T = self._token_count

        # 1) gating GEMM: x[T, VWN_N, V] × w[V, G] → [T, VWN_N, G]
        #    (op_define L154)
        gating_gemm_flops = 2 * T * self.vwn_n * self.V * self.G

        # 2) stream mix (A-gate mix):
        #    y = einsum("tan,tnv->tav", a_gate^T, x_norm)
        #    a_gate: [T, VWN_N, A], x_norm: [T, VWN_N, V] → [T, A, V]
        #    (op_define L155)
        mix_tensor_flops = 2 * T * self.A * self.vwn_n * self.V

        # 3) vector FLOPs (RMSNorm + tanh)
        #    (op_define L156)
        vector_flops = 5 * T * self.vwn_n * self.V + 3 * T * self.vwn_n * self.G

        self._compute_info.fwd_flops = gating_gemm_flops + mix_tensor_flops + vector_flops
        # backward: ~2× forward for gradient computation,
        # plus weight gradient of the gating GEMM
        self._compute_info.bwd_grad_act_flops = 2 * (mix_tensor_flops + vector_flops)
        self._compute_info.bwd_grad_w_flops = gating_gemm_flops
        self._compute_info.recompute_flops = 0  # no separate recompute for norm-like ops

    # ───  Method 2: memory access  ───
    def _comp_leaf_mem_accessed_info(self):
        T = self._token_count
        element_size = self.dtype_to_element_size[self.strategy.dtype]

        # Reads (from op_define L120-153 input/weight descriptions):
        #   - x[T, VWN_N, V] bf16
        #   - norm_weight[V] bf16
        #   - gating_weight[V, G] bf16
        #   - gating_bias[G] bf16
        read_activation = T * self.vwn_n * self.V
        read_norm_weight = self.V
        read_gating_weight = self.V * self.G
        read_gating_bias = self.G
        total_read = read_activation + read_norm_weight + read_gating_weight + read_gating_bias

        # Writes:
        #   - block_input[T, VWN_M, V] bf16
        #   - residual[T, VWN_N, V] bf16
        #   - b_gate[T, VWN_N, VWN_M] bf16
        write_block = T * self.vwn_m * self.V
        write_residual = T * self.vwn_n * self.V
        write_b_gate = T * self.vwn_n * self.vwn_m
        total_write = write_block + write_residual + write_b_gate

        self._compute_info.fwd_accessed_mem = (total_read + total_write) * element_size
        # backward: re-read outputs + write input gradients
        self._compute_info.bwd_grad_act_accessed_mem = (
            total_read + total_write + total_read
        ) * element_size  # read + write_grad + re-read weights
        self._compute_info.bwd_grad_w_accessed_mem = read_gating_weight * element_size
        self._compute_info.recompute_accessed_mem = 0

    # ───  Method 3: activation memory  ───
    def _comp_leaf_act_info_impl(self):
        T = self._token_count
        element_size = self.dtype_to_element_size[self.strategy.dtype]

        # Live tensors during forward:
        #   - x_norm[T, VWN_N, V] (RMSNorm output)
        #   - gate_logits[T, VWN_N, G] (GEMM output)
        #   - gate[T, VWN_N, G] (tanh output)
        #   - y[T, A, V] (mix output)
        #   - block_input[T, VWN_M, V] + residual[T, VWN_N, V] + b_gate[T, VWN_N, VWN_M]
        #
        # Cache for backward:
        #   - x[T, VWN_N, V] (input, if needed for grad)
        #   - x_norm[T, VWN_N, V]
        #   - gate[T, VWN_N, G]
        cache_mem = (
            T * self.vwn_n * self.V          # x_norm
            + T * self.vwn_n * self.G         # gate (tanh output)
        ) * element_size

        # Forward peak (all intermediates alive):
        fwd_peak = (
            2 * T * self.vwn_n * self.V       # x + x_norm
            + T * self.vwn_n * self.G         # gate_logits
            + T * self.vwn_n * self.G         # gate
            + T * self.A * self.V             # y
            + T * (self.vwn_m + self.vwn_n) * self.V  # block_input + residual
            + T * self.vwn_n * self.vwn_m     # b_gate
        ) * element_size

        # Backward peak (reading cached x_norm, gate; recomputing forward; writing grads):
        bwd_peak = (
            2 * T * self.vwn_n * self.V       # re-read x + write dx
            + T * self.vwn_n * self.V         # x_norm still cached
            + T * self.vwn_n * self.G         # gate still cached
            + T * self.A * self.V             # dy
            + T * (self.vwn_m + self.vwn_n) * self.V  # d(block_input) + d(residual)
        ) * element_size

        self._act_info.activation_mem_cache = cache_mem
        self._act_info.fwd_peak_mem_no_cache = max(fwd_peak, cache_mem)
        self._act_info.bwd_peak_mem_no_cache = max(bwd_peak - cache_mem, 0)

    # ───  Method 4: model params  ───
    def _comp_leaf_model_info_impl(self):
        element_size = self.dtype_to_element_size[self.strategy.dtype]
        # norm_weight[V] + gating_weight[V, G] + gating_bias[G]
        weight_bytes = (self.V + self.V * self.G + self.G) * element_size
        grad_bytes = weight_bytes
        state_bytes = weight_bytes * 2  # optimizer state (Adam: m + v)
        self._model_info.dense_weight_bytes = weight_bytes
        self._model_info.dense_grad_bytes = grad_bytes
        self._model_info.dense_state_bytes = state_bytes
        self._model_info.moe_weight_bytes = 0
        self._model_info.moe_grad_bytes = 0
        self._model_info.moe_state_bytes = 0


class VWNOutModule(MetaModule):
    """VWN Output Mixing Operator.

    Corresponds to MojoVWNOutOp in op_define.md (L941-964).

    Fuses B-gate mixing + residual add into a single roofline-modeled operator.
    Has no trainable parameters.

    Inputs:
      block_out[T, VWN_M, V]   — output from main transformer block
      b_gate[T, VWN_N, VWN_M]  — gate values from VWNIn
      residual[T, VWN_N, V]    — residual from VWNIn
    Output:
      y[T, VWN_N, V]           — gated output
    """

    def __init__(
        self,
        hidden_size: int,
        vwn_n: int,
        vwn_m: int,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name: str = 'VWNOut',
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.hidden_size = hidden_size
        self.vwn_n = vwn_n
        self.vwn_m = vwn_m
        assert hidden_size % vwn_m == 0, (
            f"hidden_size({hidden_size}) must be divisible by vwn_m({vwn_m})"
        )
        self.V = hidden_size // vwn_m

    @property
    def _token_count(self) -> int:
        input_tensor = self.input_info.tensors[0]
        if len(input_tensor.shape) == 3:
            return input_tensor.size(0)
        return input_tensor.size(0)

    def create_output_info(self):
        T = self._token_count
        return InputOutputInfo(
            tensors=[
                __import__('simumax.core.tensor', fromlist=['TensorSize']).TensorSize(
                    shape=(T, self.vwn_n, self.V)
                )
            ]
        )

    def get_input_shapes_desc(self, stage: str) -> str:
        T = self._token_count
        return (
            f'T={int(T)}, C={int(self.hidden_size)}, '
            f'VWN_N={int(self.vwn_n)}, VWN_M={int(self.vwn_m)}, V={int(self.V)}'
        )

    # ───  Method 1: FLOPS (source: op_define L958-959)  ───
    def _comp_leaf_flops_info(self):
        T = self._token_count

        # 1) B-gate mix: b_gate[T, VWN_N, VWN_M] × block_out[T, VWN_M, V]
        #    → mixed[T, VWN_N, V]   (op_define L958)
        mix_tensor_flops = 2 * T * self.vwn_n * self.vwn_m * self.V

        # 2) residual add: mixed[T, VWN_N, V] + residual[T, VWN_N, V]
        #    (op_define L959)
        residual_add_flops = T * self.vwn_n * self.V

        self._compute_info.fwd_flops = mix_tensor_flops + residual_add_flops
        self._compute_info.bwd_grad_act_flops = 2 * (mix_tensor_flops + residual_add_flops)
        self._compute_info.bwd_grad_w_flops = 0   # no learnable params
        self._compute_info.recompute_flops = 0

    # ───  Method 2: memory access  ───
    def _comp_leaf_mem_accessed_info(self):
        T = self._token_count
        element_size = self.dtype_to_element_size[self.strategy.dtype]

        # Reads:
        #   - block_out[T, VWN_M, V] bf16
        #   - b_gate[T, VWN_N, VWN_M] bf16
        #   - residual[T, VWN_N, V] bf16
        read_block = T * self.vwn_m * self.V
        read_gate = T * self.vwn_n * self.vwn_m
        read_residual = T * self.vwn_n * self.V
        total_read = read_block + read_gate + read_residual

        # Write:
        #   - y[T, VWN_N, V] bf16
        write_output = T * self.vwn_n * self.V

        self._compute_info.fwd_accessed_mem = (total_read + write_output) * element_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            total_read + write_output + total_read
        ) * element_size  # re-read for backward
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = 0

    # ───  Method 3: activation memory  ───
    def _comp_leaf_act_info_impl(self):
        T = self._token_count
        element_size = self.dtype_to_element_size[self.strategy.dtype]

        # Cache for backward:
        #   - b_gate[T, VWN_N, VWN_M]
        #   - block_out[T, VWN_M, V]
        cache_mem = (
            T * self.vwn_n * self.vwn_m       # b_gate
            + T * self.vwn_m * self.V          # block_out
        ) * element_size

        # Forward peak:
        fwd_peak = (
            T * self.vwn_m * self.V             # block_out
            + T * self.vwn_n * self.vwn_m       # b_gate
            + T * self.vwn_n * self.V           # residual
            + 2 * T * self.vwn_n * self.V       # mixed (temp) + y (output)
        ) * element_size

        # Backward peak:
        bwd_peak = (
            cache_mem                           # cached b_gate + block_out
            + T * self.vwn_n * self.V           # dy + d(residual)
            + T * self.vwn_n * self.vwn_m       # d(b_gate)
        ) * element_size

        self._act_info.activation_mem_cache = cache_mem
        self._act_info.fwd_peak_mem_no_cache = max(fwd_peak, cache_mem)
        self._act_info.bwd_peak_mem_no_cache = max(bwd_peak - cache_mem, 0)

    # ───  Method 4: model params ───
    def _comp_leaf_model_info_impl(self):
        # VWNOut has no trainable parameters
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0
        self._model_info.moe_weight_bytes = 0
        self._model_info.moe_grad_bytes = 0
        self._model_info.moe_state_bytes = 0
