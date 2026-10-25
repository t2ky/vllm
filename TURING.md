# vLLM on Turing (sm_75)

This branch keeps vLLM **v0.31.0** usable and fast on Turing GPUs (RTX 20xx, T4, Quadro RTX) for
models and features that otherwise fail or fall back to slow paths. It is maintained for a small
self-hosted cluster of RTX 2080 / 2080 Ti cards, and stays as close to upstream as possible.

## Changes

1. **Attention tile fix for head_dim 512** (`vllm/v1/attention/ops/triton_unified_attention.py`).
   Gemma 4's global attention layers use head_dim 512. With the default prefill tile (32) the Triton
   attention kernel requests ~96 KB of shared memory and fails on Turing (64 KB):
   `OutOfResources: shared memory, Required: 98304, Hardware limit: 65536`.
   On CC < 8.0 with head_dim >= 512 the tiles are reduced to 16 with one stage (~48 KB).
   Same idea as [vllm-project/vllm#39018](https://github.com/vllm-project/vllm/pull/39018) by @lisp19.
2. **Tensor cores for Triton kernels** — not in this repo, but required for speed: Triton >= 3.3 lowers
   `tl.dot` to FMA on CC < 8.0 ([vllm-project/vllm#60821](https://github.com/vllm-project/vllm/issues/60821)),
   which makes `TRITON_ATTN` and the multi-LoRA shrink/expand kernels very slow on Turing. Use the Triton
   build from [t2ky/triton `turing-mma-v3.7.1`](https://github.com/t2ky/triton/tree/turing-mma-v3.7.1)
   (a port of triton-lang/triton#9970). `docker/Dockerfile.turing` combines both.

## Measured (Gemma 4 12B IT, GPTQ 4-bit, TP=2 on 2x RTX 2080 8GB, one LoRA rank 16, ~800-token prompts)

| | vLLM 0.31.0 + upstream Triton 3.7.1 (+ change 1) | this branch + Turing Triton |
|---|---:|---:|
| decode time per token (1 stream) | 50.9 ms | **18.6 ms** |
| median latency, 8 concurrent | 13.1 s | **4.7 s** |
| aggregate generation, 8 concurrent | 35 tok/s | **93 tok/s** |

Profiled before the Triton fix, 61% of GPU time in single-stream decode was LoRA shrink/expand
running on FMA. Greedy outputs matched a stock-vLLM RTX 3090 reference exactly for 17 of 20 prompts.

## Notes for Turing

- Needs a driver that runs the CUDA 13.0 image (R580 or newer). The `-cu129` images need R575+.
- Use `--dtype half` (no bf16 on Turing) and `--attention-backend TRITON_ATTN`.
  FlashInfer on sm_75 ([vllm-project/vllm#55380](https://github.com/vllm-project/vllm/pull/55380)) supports
  head_dim <= 256 only, so it cannot serve models that mix 256 and 512.
- FLEX_ATTENTION hits the same 96 KB shared-memory limit with head_dim 512.
- The NVIDIA device plugin does not consider PCIe topology; on dual-socket hosts a TP=2 pod can get two
  GPUs on different sockets. Check `nvidia-smi topo -m` inside the pod.
