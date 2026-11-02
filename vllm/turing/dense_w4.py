# SPDX-License-Identifier: Apache-2.0
"""Dense Marlin GEMM -> CUDA W4A16 / W8A16 (cuda_w4) for Turing. Called from ``_marlin_gemm``
(the body of the vllm::marlin_gemm custom op) when VLLM_CUDA_DENSE_W4=1.

Inside the custom op it is invisible to torch.compile and captured in CUDA graphs as is. The decision uses
only shapes, dtypes and size_m (Python ints): no sync and no device-to-host copy. Weights are read in the
Marlin layout, so no repacking or extra workspace is needed.

Used when: uint4b8 or uint8b128 weights, fp16 activations, no bias / a_scales / global_scale / zero points,
group size 32, size_m >= VLLM_CUDA_DENSE_W4_MIN_M (default 1), and the config table does not mark the shape
and M as use_marlin. Otherwise returns None and the caller runs Marlin.
"""

import os

import torch

from vllm.turing import cuda_w4

STATS = {"cuda": 0, "marlin": 0, "no_config_shapes": set(), "shapes": set()}  # call counts, for checking
_MIN_M = int(os.environ.get("VLLM_CUDA_DENSE_W4_MIN_M", "1"))
_CACHE: dict = {}
_TYPE_IDS = None


def _type_ids():
    global _TYPE_IDS
    if _TYPE_IDS is None:
        from vllm.scalar_type import scalar_types

        _TYPE_IDS = (scalar_types.uint4b8.id, scalar_types.uint8b128.id)
    return _TYPE_IDS


def marlin_gemm(a, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, b_q_type_id,
                size_m, size_n, size_k) -> torch.Tensor | None:
    u4, u8 = _type_ids()
    bits = 4 if b_q_type_id == u4 else 8 if b_q_type_id == u8 else 0
    if (size_m >= _MIN_M and bits and a.dtype == torch.float16 and b_bias is None and a_scales is None
            and global_scale is None and (b_zeros is None or b_zeros.numel() == 0) and b_scales.dim() == 2
            and b_scales.shape[0] * 32 == size_k and a.shape[1] == size_k and b_scales.shape[1] == size_n
            and b_q_weight.dtype == torch.int32 and b_q_weight.shape[0] * 16 == size_k
            and b_q_weight.shape[1] == size_n * bits // 2):
        key = (size_m, size_k, size_n, bits)
        cfg = _CACHE.get(key, False)
        if cfg is False:
            cfg = _CACHE[key] = cuda_w4.pick(size_m, size_k, size_n, bits=bits)
        if cfg is not None:
            STATS["cuda"] += 1
            STATS["shapes"].add((size_k, size_n, bits))
            return cuda_w4.gemm(a, b_q_weight, b_scales, **cfg)
        STATS["no_config_shapes"].add((size_k, size_n, bits))
    STATS["marlin"] += 1
    return None
