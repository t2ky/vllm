# SPDX-License-Identifier: Apache-2.0
"""Kernels for Turing (sm_75) GPUs, enabled by environment variables (all off by default).

- ``VLLM_CUDA_DECODE_ATTN=1``: decode-only batches of TRITON_ATTN with an int8 per-token-head KV cache
  use a CUDA decode attention kernel (``cuda_attn``, source ``csrc/turing/decode_attn_sm75.cu``).
- ``VLLM_CUDA_DENSE_W4=1``: dense Marlin GEMMs (uint4b8 / uint8b128, group 32, fp16) use a CUDA W4A16 /
  W8A16 kernel that reads the Marlin layout as-is (``cuda_w4``, source ``csrc/turing/w4a16_sm75.cu``).
- ``VLLM_TRITON_MOE_W4=1``: Marlin MoE (uint4b8, group 32, fp16, SiLU) uses a Triton grouped GEMM that reads
  the Marlin MoE layout as-is (``moe_w4``).

See TURING.md at the repository root.
"""
