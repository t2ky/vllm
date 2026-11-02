# SPDX-License-Identifier: Apache-2.0
"""Marlin MoE -> Triton W4A16 grouped GEMM (moe_w4k) for Turing. Called from ``fused_marlin_moe`` when
VLLM_TRITON_MOE_W4=1.

- Without LoRA (``MarlinExperts._lora_context is None``, signalled through ``no_lora()``): fused fast path
  with SiLU * up in the w13 epilogue and top-k weights in w2.
- With LoRA: the same flow as ``_fused_marlin_moe``; only the two ``ops.moe_wna16_marlin_gemm`` calls are
  replaced by ``moe_w4k.moe_gemm`` (activation_func / moe_sum, i.e. the LoRA additions, run unchanged).
  Note: with --enable-lora this path is used even when a batch has no LoRA requests.

Not used (returns None, the caller runs Marlin) when M < VLLM_TRITON_MOE_W4_MIN_M (default 128), weights are
not uint4b8, there are biases / zero points / global scales / input quantization / an expert map / router
weights on the input, activations are not fp16, the group size is not 32, or the activation is not SiLU.

The Triton kernels are JIT-compiled on first use per shape and config; vLLM's profile run and CUDA graph
capture compile the common ones, but not necessarily all of them.
"""

import contextlib
import contextvars
import os

import torch

_NO_LORA = contextvars.ContextVar("turing_moe_w4_no_lora", default=False)
# Returned when the call is not handled (the caller runs Marlin). Not None: on the LoRA path the handled result is
# moe_sum's return value, which is None, and treating that as "not handled" ran the MoE twice.
NOT_HANDLED = object()
STATS = {"fused": 0, "lora_gemm": 0, "marlin": 0}  # call counts, for checking


@contextlib.contextmanager
def no_lora():
    tok = _NO_LORA.set(True)
    try:
        yield
    finally:
        _NO_LORA.reset(tok)


def _eligible(hidden_states, w1, w1_scale, quant_type_id, kw):
    from vllm.scalar_type import scalar_types

    if quant_type_id != scalar_types.uint4b8.id or hidden_states.dtype != torch.float16:
        return False
    for k in ("bias1", "bias2", "w1_zeros", "w2_zeros", "global_scale1", "global_scale2",
              "input_global_scale1", "input_global_scale2", "expert_map", "input_dtype"):
        if kw.get(k) is not None:
            return False
    if kw.get("apply_router_weight_on_input"):
        return False
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if kw.get("activation", MoEActivation.SILU) != MoEActivation.SILU:
        return False
    K = hidden_states.shape[1]
    if w1_scale.dtype != torch.float16 or w1_scale.shape[1] * 32 != K:  # group 32
        return False
    gne = kw.get("global_num_experts", -1)
    if gne not in (-1, w1.shape[0]):
        return False
    return hidden_states.shape[0] >= int(os.environ.get("VLLM_TRITON_MOE_W4_MIN_M", "128"))


def fused_marlin_moe(hidden_states, w1, w2, bias1, bias2, w1_scale, w2_scale, topk_weights, topk_ids,
                     quant_type_id, kw) -> torch.Tensor | None:
    """Same arguments as vllm's fused_marlin_moe (keyword arguments in ``kw``). NOT_HANDLED = not handled."""
    if not _eligible(hidden_states, w1, w1_scale, quant_type_id, dict(kw, bias1=bias1, bias2=bias2)):
        STATS["marlin"] += 1
        return NOT_HANDLED
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
        MoEActivation,
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.turing import moe_w4k

    M, K = hidden_states.shape
    E = w1.shape[0]
    topk = topk_ids.shape[1]
    I = w1_scale.shape[2] // 2  # noqa: E741
    cfg = moe_w4k.pick(M)
    BM = cfg["BM"]
    out = kw.get("output")
    if _NO_LORA.get():
        STATS["fused"] += 1
        return moe_w4k.fused_moe(hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids,
                                 BM=BM, cfg13=cfg["w13"], cfg2=cfg["w2"], out=out)
    # LoRA path: same flow as _fused_marlin_moe, only the GEMMs are replaced
    STATS["lora_gemm"] += 1
    BM = cfg.get("BM_g", BM)
    sid, eid, npad = moe_align_block_size(topk_ids, BM, E, None, ignore_invalid_experts=True)
    cache1 = torch.empty(M * topk, 2 * I, dtype=hidden_states.dtype, device=hidden_states.device)
    moe_w4k.moe_gemm(hidden_states, cache1, w1, w1_scale, sid, eid, npad, topk_weights, top_k=topk,
                     mul_topk_weights=False, size_m=M, size_n=2 * I, size_k=K, BM=BM, cfg=cfg["g13"])
    cache2 = torch.empty(M * topk, I, dtype=hidden_states.dtype, device=hidden_states.device)
    act = kw.get("activation_func")
    if act is None:
        acfg = kw.get("activation_config") or ApplyMoEActivationConfig()
        apply_moe_activation(MoEActivation.SILU, cache2, cache1, activation_config=acfg, topk_ids=topk_ids,
                             expert_map=None)
    else:
        act(MoEActivation.SILU, cache2, cache1, topk_ids=topk_ids, expert_map=None)
    cache3 = torch.empty(M * topk, K, dtype=hidden_states.dtype, device=hidden_states.device)
    moe_w4k.moe_gemm(cache2, cache3, w2, w2_scale, sid, eid, npad, topk_weights, top_k=1,
                     mul_topk_weights=True, size_m=M * topk, size_n=K, size_k=I, BM=BM, cfg=cfg["g2"])
    if out is None:
        out = torch.empty_like(hidden_states)
    moe_out = cache3.view(M, topk, K)
    moe_sum = kw.get("moe_sum")
    if moe_sum is None:
        return torch.sum(moe_out, dim=1, out=out)
    return moe_sum(moe_out, out, topk_ids, None)
