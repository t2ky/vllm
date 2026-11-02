# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


# --8<-- [start:short_conv]
@PluggableLayer.register("short_conv")
class ShortConv(MambaBase, PluggableLayer):
    # --8<-- [end:short_conv]

    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.conv_dim = dim
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = ColumnParallelLinear(
            input_size=self.L_cache,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv.weight.data = self.conv.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        vllm_config = get_current_vllm_config()
        self.num_spec = vllm_config.num_speculative_tokens
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        # Run the conv inside the CUDA graph (short_conv_graph). Speculative decoding and
        # mamba_cache_mode="all" (per-block states) keep the original op (a graph split point).
        # VLLM_SHORT_CONV_GRAPH=0 disables it.
        import os

        self.use_graph_conv = (
            os.environ.get("VLLM_SHORT_CONV_GRAPH", "1") == "1"
            and current_platform.is_cuda()
            and self.num_spec == 0
            and vllm_config.cache_config.mamba_cache_mode != "all"
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        # Reference torch causal conv1d; runs on all CPU platforms. AMX kernels
        # for causal conv can be plugged in here later.
        from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
            causal_conv1d_fn_cpu as causal_conv1d_torch,
        )
        from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
            causal_conv1d_update_cpu,
            causal_conv1d_update_torch,
        )
        from vllm.platforms import CpuArchEnum, current_platform

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata: AttentionMetadata | None = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw.get(self.prefix)
            assert attn_metadata is None or isinstance(
                attn_metadata, ShortConvAttentionMetadata
            )

        BCx, _ = self.in_proj(hidden_states)
        B, C, x = BCx.chunk(3, dim=-1)

        # (dim, kernel_size) — same reshape as forward_cuda
        conv_weights = self.conv.weight.view(
            self.conv.weight.size(0), self.conv.weight.size(2)
        )

        if attn_metadata is None:
            # Profile run — output value doesn't matter
            Bx = (B * x).contiguous()
            output_tensor, _ = self.out_proj(C * Bx)
            output[: hidden_states.shape[0]] = output_tensor
            return

        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )  # (num_blocks, dim, state_len)

        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        B_d, B_p = torch.split(
            B[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0
        )
        C_d, C_p = torch.split(
            C[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0
        )
        x_d, x_p = torch.split(
            x[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0
        )

        conv_output_list = []

        if has_prefill:
            assert attn_metadata.state_indices_tensor_p is not None
            Bx_p = (B_p * x_p).transpose(0, 1)  # (dim, num_prefill_tokens)
            out_p = causal_conv1d_torch(
                Bx_p,
                conv_weights,
                self.conv.bias,
                conv_state,
                attn_metadata.query_start_loc_p,
                attn_metadata.state_indices_tensor_p.flatten(),
                attn_metadata.has_initial_states_p,
                activation=None,
            ).transpose(0, 1)[:num_prefill_tokens]  # (num_prefill_tokens, dim)
            conv_output_list.append(C_p * out_p)

        if has_decode:
            assert attn_metadata.state_indices_tensor_d is not None
            state_indices_d = attn_metadata.state_indices_tensor_d.flatten()
            Bx_d = B_d * x_d  # (num_decodes, dim)
            if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
                conv_state_view = conv_state[state_indices_d].contiguous()
                out_d = causal_conv1d_update_torch(
                    Bx_d.unsqueeze(-1),
                    conv_state_view,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                ).squeeze(-1)
                conv_state[state_indices_d] = conv_state_view
            else:
                out_d = causal_conv1d_update_cpu(
                    Bx_d,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_d,
                )
            conv_output_list.insert(0, C_d * out_d)

        hidden_states_out = torch.vstack(conv_output_list)
        output[:num_actual_tokens], _ = self.out_proj(hidden_states_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        # Keep in_proj / out_proj (quantized GEMM + LoRA) outside the custom op so that they are captured
        # in the PIECEWISE CUDA graphs. The custom op (a graph split point; same name and signature as
        # before) only does the conv. Otherwise, in mixed prefill/decode steps the linear layers and LoRA
        # kernels are launched one by one outside the graph, which is CPU-bound on slow hosts.
        BCx, _ = self.in_proj(hidden_states)
        y = torch.empty(
            (BCx.shape[0], BCx.shape[1] // 3), dtype=BCx.dtype, device=BCx.device
        )
        if self.use_graph_conv:
            # conv inside the CUDA graph as well (not a split point); reads only persistent GPU metadata
            torch.ops.vllm.short_conv_graph(BCx, y, self.prefix)
        else:
            torch.ops.vllm.short_conv(BCx, y, self.prefix)
        out, _ = self.out_proj(y)
        output.copy_(out)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        # ShortConvAttentionMetadata contains metadata necessary for the
        # short_conv triton kernels to operate in continuous batching and in
        # chunked prefill modes; they are computed at top-level model forward
        # since they stay the same and reused for all mamba layers in the same
        # iteration.
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata: AttentionMetadata | None = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is not None:
            assert isinstance(attn_metadata, ShortConvAttentionMetadata)
            conv_state = (
                self.kv_cache[0]
                if is_conv_state_dim_first()
                else self.kv_cache[0].transpose(-1, -2)
            )
            state_indices_tensor_p = attn_metadata.state_indices_tensor_p
            state_indices_tensor_d = attn_metadata.state_indices_tensor_d
            has_initial_states_p = attn_metadata.has_initial_states_p
            query_start_loc_p = attn_metadata.query_start_loc_p
            num_accepted_tokens = attn_metadata.num_accepted_tokens
            query_start_loc_d = attn_metadata.query_start_loc_d

        # hidden_states is the in_proj output (BCx); output is the out_proj input (C * conv(B * x))
        BCx = hidden_states

        B, C, x = BCx.chunk(3, dim=-1)

        conv_weights = self.conv.weight.view(
            self.conv.weight.size(0), self.conv.weight.size(2)
        )

        if attn_metadata is None:
            # V1 profile run
            output.copy_(C * (B * x))
            return

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        # NOTE: V1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        B_d, B_p = torch.split(
            B[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        C_d, C_p = torch.split(
            C[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        x_d, x_p = torch.split(
            x[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        conv_output_list = []

        if has_prefill:
            Bx_p = (B_p * x_p).transpose(0, 1)
            Bx = causal_conv1d_fn(
                Bx_p,
                conv_weights,
                self.conv.bias,
                activation=None,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)[:num_prefill_tokens]

            y = C_p * Bx
            conv_output_list.append(y)

        if has_decode:
            Bx_d = (B_d * x_d).contiguous()
            if num_accepted_tokens is not None:
                # Speculative decode: the verify step feeds >1 query token per
                # decode request, so use the spec-aware conv update path
                # (mirrors mamba_mixer2). state_indices_tensor_d is
                # (num_decodes, 1 + num_spec_tokens) here.
                assert state_indices_tensor_d is not None
                Bx = causal_conv1d_update(
                    Bx_d,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_tensor_d,
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=query_start_loc_d,
                    max_query_len=state_indices_tensor_d.size(-1),
                )
            else:
                # Non-spec decode
                Bx = causal_conv1d_update(
                    Bx_d,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_tensor_d,
                )
            y = C_d * Bx
            conv_output_list.insert(0, y)

        # Merge prefill and decode outputs before passing to gated MLP
        hidden_states = torch.vstack(conv_output_list)

        output[:num_actual_tokens] = hidden_states
        # zero the padding rows (do not feed garbage into the router of the following MoE)
        output[num_actual_tokens:].zero_()

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.conv_dim,
            conv_kernel=self.L_cache,
            num_spec=self.num_spec,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV


def short_conv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if not current_platform.is_cpu():
        self.forward_cuda(hidden_states=hidden_states, output=output)
    else:
        self.forward_native(hidden_states=hidden_states, output=output)


direct_register_custom_op(
    op_name="short_conv",
    op_func=short_conv,
    mutates_args=["output"],
)


_GRAPH_CALLS = [0]


def short_conv_graph(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """hidden_states = in_proj output (B, C, x); output = C * conv(B * x).
    Not in splitting_ops, so it is captured in the piecewise CUDA graphs (this Python body does not run on
    replay). The kernels read only the kv-cache state and the persistent buffers that
    ShortConvAttentionMetadataBuilder writes every step."""
    from vllm.model_executor.layers.mamba.ops.short_conv_graph import (
        BUFFERS,
        short_conv_graph_forward,
    )

    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    buf = BUFFERS.get(layer_name)
    kv = self.kv_cache[0]
    _GRAPH_CALLS[0] += 1  # times the Python body ran (not incremented by graph replays; for checking)
    if kv.numel() == 0:
        # profile run (before the kv-cache exists): values are unused
        B, C, x = hidden_states.chunk(3, dim=-1)
        output.copy_(C * (B * x))
        return
    if buf is None:
        # The builder did not create the persistent buffers (e.g. VLLM_SHORT_CONV_GRAPH changed but a stale
        # torch.compile cache was reused). Fail loudly instead of producing wrong outputs.
        raise RuntimeError(
            f"short_conv_graph: no persistent metadata for {layer_name}. After changing VLLM_SHORT_CONV_GRAPH "
            "(or updating this code), clear the torch.compile cache (~/.cache/vllm/torch_compile_cache) or start "
            "with VLLM_DISABLE_COMPILE_CACHE=1"
        )
    conv_state = kv if is_conv_state_dim_first() else kv.transpose(-1, -2)
    conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
    short_conv_graph_forward(
        hidden_states,
        output,
        conv_weights,
        self.conv.bias,
        conv_state,
        buf,
        clear_armed=(layer_name == buf.last_layer),
    )


def short_conv_graph_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="short_conv_graph",
    op_func=short_conv_graph,
    mutates_args=["output"],
    fake_impl=short_conv_graph_fake,
)
