# vLLM on Turing (sm_75)

This branch keeps vLLM **v0.31.0** usable and fast on Turing GPUs (RTX 20xx, T4, Quadro RTX) for
models and features that otherwise fail or fall back to slow paths. It is maintained for a small
self-hosted cluster of RTX 2080 / 2080 Ti cards, and stays as close to upstream as possible.
Everything new is either a plain fix or behind an environment variable; `docker/Dockerfile.turing`
builds an image with all of it (see [Image](#image)).

## Summary of changes

| # | Change | Switch (default) | Files |
|---|---|---|---|
| 1 | head_dim 512 attention fits in 64 KB shared memory | always | `vllm/v1/attention/ops/triton_unified_attention.py` |
| 2 | Triton with tensor cores on sm_75 (separate repo) | wheel in the image | [t2ky/triton](https://github.com/t2ky/triton) |
| 3 | LoRA kernel metadata built on the CPU (no GPU sync per step) | always | `vllm/lora/ops/triton_ops/lora_kernel_metadata.py`, `vllm/lora/punica_wrapper/punica_gpu.py` |
| 4 | LFM2 ShortConv layers inside CUDA graphs | `VLLM_SHORT_CONV_GRAPH` (`1`, on) | `vllm/model_executor/layers/mamba/short_conv.py`, `.../mamba/ops/short_conv_graph.py`, `vllm/v1/attention/backends/short_conv_attn.py` |
| 5 | 16-byte-aligned per-token-head KV slots (backport of vllm#62113) | always (int8/fp8 per-token-head KV) | `vllm/v1/attention/backends/triton_attn.py` |
| 6 | CUDA decode attention for int8 KV | `VLLM_CUDA_DECODE_ATTN` (`0`, off) | `csrc/turing/decode_attn_sm75.cu`, `vllm/turing/cuda_attn.py`, `triton_attn.py` |
| 7 | CUDA W4A16 / W8A16 GEMM for dense Marlin layers | `VLLM_CUDA_DENSE_W4` (`0`, off) | `csrc/turing/w4a16_sm75.cu`, `vllm/turing/cuda_w4.py`, `vllm/turing/dense_w4.py`, `marlin_utils.py` |
| 8 | Triton W4A16 grouped GEMM for Marlin MoE | `VLLM_TRITON_MOE_W4` (`0`, off) | `vllm/turing/moe_w4k.py`, `vllm/turing/moe_w4.py`, `marlin_moe.py` |
| - | Opt-in software pipelining in the Triton fork | `TRITON_TURING_PIPELINE` (`0`, off) | Triton fork |

The numbers below were measured on RTX 2080 8 GB cards (driver 580, CUDA 13.0 image) unless noted.
"Fixed load" means a fixed set of 96 chat requests (about 740 prompt tokens, 128 output tokens,
16 concurrent, closed loop), timed until all finish.

## 1. Attention tile fix for head_dim 512

Gemma 4's global attention layers use head_dim 512. With the default prefill tile (32) the Triton
attention kernel requests ~96 KB of shared memory and fails on Turing (64 KB):
`OutOfResources: shared memory, Required: 98304, Hardware limit: 65536`.
On CC < 8.0 with head_dim >= 512 the tiles are reduced to 16 with one stage (~48 KB).
Same idea as [vllm-project/vllm#39018](https://github.com/vllm-project/vllm/pull/39018) by @lisp19.

## 2. Tensor cores for Triton kernels

Not in this repo, but required for speed: Triton >= 3.3 lowers `tl.dot` to FMA on CC < 8.0
([vllm-project/vllm#60821](https://github.com/vllm-project/vllm/issues/60821)), which makes `TRITON_ATTN`
and the multi-LoRA shrink/expand kernels very slow on Turing. Use the Triton build from
[t2ky/triton](https://github.com/t2ky/triton): branch `turing-mma-v3.7.1` (a port of triton-lang/triton#9970)
or `turing-pipeline-v3.7.1` (the same plus the opt-in `TRITON_TURING_PIPELINE=1` register-staged software
pipelining; without the variable it behaves like `turing-mma-v3.7.1`). See `TURING.md` there.

Gemma 4 12B IT, GPTQ 4-bit, TP=2 on 2x RTX 2080, one LoRA rank 16, ~800-token prompts:

| | vLLM 0.31.0 + upstream Triton 3.7.1 (+ change 1) | + Turing Triton |
|---|---:|---:|
| decode time per token (1 stream) | 50.9 ms | **18.6 ms** |
| median latency, 8 concurrent | 13.1 s | **4.7 s** |
| aggregate generation, 8 concurrent | 35 tok/s | **93 tok/s** |

Profiled before the Triton fix, 61% of GPU time in single-stream decode was LoRA shrink/expand
running on FMA. Greedy outputs matched a stock-vLLM RTX 3090 reference exactly for 17 of 20 prompts.

`TRITON_TURING_PIPELINE=1` speeds up plain fp16 GEMMs by 12-15% at large M, but end-to-end serving did not
improve in our tests (attention loops are excluded from it), so it is not enabled in the image.

## 3. LoRA kernel metadata on the CPU

`PunicaWrapperGPU.update_metadata` used to sort and count the token-to-LoRA mapping on the GPU and read
the results back (`gpu_sync_allowed`), a host sync every step that breaks the overlap of async
scheduling. On a host with a slow CPU this was about half of the EngineCore CPU time. The same tensors are
now computed with numpy from `LoRAMapping` and copied with `non_blocking=True`. No switch; the outputs are
identical.

## 4. LFM2 ShortConv inside CUDA graphs (`VLLM_SHORT_CONV_GRAPH`, default on)

`vllm::short_conv` is a split point of the piecewise CUDA graphs, so in mixed prefill/decode steps its
in_proj / out_proj (quantized GEMM + LoRA) and the conv kernels ran eagerly: about 9 launches x 18 layers
per step for LFM2.5-8B-A1B. On hosts where a launch costs 40-60 us of CPU this dominated.

- in_proj / out_proj moved out of the custom op, so they are captured in the graphs.
- New op `vllm::short_conv_graph` (not a split point): two Triton kernels that read only persistent GPU
  buffers (per-request token ranges, state slots, an "armed" latch), written by
  `ShortConvAttentionMetadataBuilder.build()` with one kernel per step. No host-side decisions.
- Speculative decoding and `mamba_cache_mode="all"` keep the original op. `VLLM_SHORT_CONV_GRAPH=0`
  restores it everywhere.

Checked on an RTX 3090 (LFM2.5-8B-A1B GPTQ, expert LoRA, prefix caching): prompt NLL and greedy agreement
within the run-to-run variation of stock vLLM; the op's Python body never runs during generation.

**Pitfall: clear the torch.compile cache** (`~/.cache/vllm/torch_compile_cache`, or start with
`VLLM_DISABLE_COMPILE_CACHE=1`) after switching `VLLM_SHORT_CONV_GRAPH` and when moving from an image without
this change. The variable is not part of the compile cache key, so a stale graph calls the other op (we got
NLL 17 instead of 3 once). A mismatch now raises a RuntimeError instead of producing garbage.

**LFM2 MoE serving settings that helped:** `--max-num-batched-tokens 1024` together with CUDA graphs up to
1024 tokens, e.g.
`--compilation-config '{"max_cudagraph_capture_size": 1024, "cudagraph_capture_sizes": [1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,640,768,896,1024]}'`,
so that mixed steps also replay graphs.

## 5. 16-byte-aligned per-token-head KV slots (vllm#62113)

Backport of the `triton_attn.py` part of
[vllm-project/vllm#62113](https://github.com/vllm-project/vllm/pull/62113) by @noon-at-cgn: with
`--kv-cache-dtype int8_per_token_head` (or fp8) each K/V side (data + fp32 scale) is padded to 16 bytes so
the Triton kernels can vectorize the loads. Required by change 6.

## 6. CUDA decode attention (`VLLM_CUDA_DECODE_ATTN=1`, default off)

For `TRITON_ATTN` with `--kv-cache-dtype int8_per_token_head`, decode-only batches
(`max_query_len == 1`) go to a CUDA kernel (`csrc/turing/decode_attn_sm75.cu`): mma.m16n8k8 with 8 query
heads as n, split-KV with the split count from shapes and the SM count only, the last CTA combines, one
kernel, CUDA-graph safe. Layers it cannot handle (ALiBi, sinks, KV sharing, other head sizes, non-fp16)
stay on Triton. Mixed and prefill batches always use Triton.

- RTX 2080: ~1.15x the Triton kernel at 8-32 sequences with 512-2048 context (about 310 GB/s, 70% of the
  memory bandwidth), 0.8-1.1x for few or short sequences. Decode step -3% on Gemma 4 12B TP=2; no
  measurable end-to-end change at the fixed load.
- Accuracy: 2-3.5e-4 relative to an fp32 reference; teacher-forced decode NLL matches Triton to 4 digits.
- Tuning: `CUDA_ATTN_NSPLIT_SWA` / `CUDA_ATTN_NSPLIT_FULL` (max splits, 16), `CUDA_ATTN_TARGET` (CTAs per SM,
  4), `CUDA_ATTN_MIN_CHUNK_SWA` / `_FULL` (min tokens per split, 64 / 128).
- The workspace for `max(max_num_seqs, max_cudagraph_capture_size)` sequences is allocated in the profile
  run, so the KV-cache sizing accounts for it.

## 7. CUDA W4A16 / W8A16 GEMM for dense layers (`VLLM_CUDA_DENSE_W4=1`, default off)

`_marlin_gemm` (the body of the `vllm::marlin_gemm` custom op) branches to a CUDA kernel
(`csrc/turing/w4a16_sm75.cu`) that reads the dense Marlin layout as it is: the int4 tile goes straight into
registers, which already is the mma B fragment layout, so B never touches shared memory. Also handles
8-bit weights (uint8b128, e.g. an 8-bit lm_head). Small M uses split-K in one kernel (deterministic).

- Used for uint4b8 / uint8b128, group 32, fp16, no bias / zero points / activation or global scales,
  `M >= VLLM_CUDA_DENSE_W4_MIN_M` (default 1). Otherwise Marlin.
- Configs: `vllm/turing/configs/w4a16_sm75.json`, measured on an RTX 2080 for the Gemma 4 12B TP=2 shapes
  and the 8-bit lm_head. Other shapes use the nearest shape in the table; re-measure for other models.
  Format `{"K,N": {"M": {"BM","BN","BK","WM","WN","MINB","split"}}}`, 8-bit shapes as `"K,N,8"`, an entry
  with `"use_marlin": true` falls back to Marlin. `CUDA_W4_CONFIG` points to another table.
- RTX 2080: 1.12-1.23x faster than Marlin for M=16-2048 (6 Gemma-shaped GEMMs), 0.89-0.98x of cuBLAS on
  dequantized weights. Error vs fp32 2.7e-4 (Marlin on sm_75: 1.2-3.2e-3). End to end (Gemma 4 12B, TP=2,
  LoRA, fixed load): total -8%, prefill -12%, decode -2%.
- Pitfall: the split-K tickets are one buffer per device, so only one stream may issue these GEMMs at a
  time (true for vLLM's normal execution).

## 8. Triton grouped GEMM for Marlin MoE (`VLLM_TRITON_MOE_W4=1`, default off)

`fused_marlin_moe` branches to a Triton W4A16 grouped GEMM (`vllm/turing/moe_w4k.py`) that reads the
Marlin MoE layout as it is (relabelled k/n so the loaded words are Triton's mma B registers). Without LoRA
it uses a fused path (SiLU * up in the w13 epilogue, top-k weights in w2); with LoRA only the two GEMMs are
replaced and the LoRA hooks run unchanged. Note that with `--enable-lora` the LoRA path is used even for
batches without LoRA requests.

- Used for uint4b8, group 32, fp16, SiLU, no bias / zero points / global or input scales / expert_map,
  `M >= VLLM_TRITON_MOE_W4_MIN_M` (default 128 tokens). Otherwise Marlin MoE.
- Configs: `vllm/turing/configs/moe_w4_sm75.json` (RTX 2080, LFM2.5-8B-A1B shapes); `MOE_W4_CONFIG` overrides.
- RTX 2080: fused path 1.34-1.52x and LoRA path 1.23-1.39x faster than Marlin MoE for M=64-2048.
  NLL matches Marlin to 3-4 digits.
- Pitfall: the kernels are JIT-compiled per shape/config on first use. The profile run and graph capture
  compile most of them, but the first requests after startup can still be slow; warm the server up before
  measuring or taking traffic (keeping the Triton cache directory across restarts also helps).

## Image

`docker/Dockerfile.turing`: official `vllm/vllm-openai:v0.31.0` + the Triton wheel + the changed vLLM files
copied over the installed package + the CUDA extensions of `csrc/turing`, compiled during the build with
`TORCH_CUDA_ARCH_LIST=7.5+PTX` (`python -m vllm.turing._ext build`), so nothing is compiled at runtime.
The extensions land in `vllm/turing/_ext/` (override with `VLLM_TURING_EXT_DIR`). The PTX lets them run on
newer GPUs too (JIT by the driver), which is handy for testing; the configs are tuned for sm_75.

```bash
# 1. Triton wheel (inside the same base image, so the Python ABI matches; ~20 min on 16 cores)
git clone -b turing-pipeline-v3.7.1 https://github.com/t2ky/triton && cd triton
docker run --rm -v $PWD:/src -v $PWD/dist:/dist -e MAX_JOBS=12 -e TRITON_WHEEL_VERSION_SUFFIX=+turing \
  --entrypoint bash vllm/vllm-openai:v0.31.0 -c '
    apt-get update -qq && apt-get install -y -qq build-essential git zlib1g-dev
    pip install -r /src/python/requirements.txt
    cd /src && git config --global --add safe.directory /src
    pip wheel --no-build-isolation --no-deps -w /dist .'
# 2. image, from the root of this repository
cp ../triton/dist/triton-3.7.1+*.turing-cp312-*.whl docker/
docker build -f docker/Dockerfile.turing -t vllm-turing:v0.31.0 .
```

Example (Gemma 4 12B GPTQ, TP=2, LoRA, everything on):

```bash
VLLM_CUDA_DENSE_W4=1 VLLM_CUDA_DECODE_ATTN=1 vllm serve <model> --dtype half -tp 2 \
  --attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head --enable-lora ...
```

Example (LFM2.5-8B-A1B GPTQ):

```bash
VLLM_TRITON_MOE_W4=1 VLLM_CUDA_DENSE_W4=1 vllm serve <model> --dtype half \
  --max-num-batched-tokens 1024 \
  --compilation-config '{"max_cudagraph_capture_size": 1024, "cudagraph_capture_sizes": [1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,640,768,896,1024]}' ...
```

## Notes for Turing

- Needs a driver that runs the CUDA 13.0 image (R580 or newer). The `-cu129` images need R575+.
- Use `--dtype half` (no bf16 on Turing) and `--attention-backend TRITON_ATTN`.
  FlashInfer on sm_75 ([vllm-project/vllm#55380](https://github.com/vllm-project/vllm/pull/55380)) supports
  head_dim <= 256 only, so it cannot serve models that mix 256 and 512.
- FLEX_ATTENTION hits the same 96 KB shared-memory limit with head_dim 512.
- The NVIDIA device plugin does not consider PCIe topology; on dual-socket hosts a TP=2 pod can get two
  GPUs on different sockets, or a GPU in an x8 slot (20-50% slower all-reduce). Check `nvidia-smi topo -m`
  inside the pod.
- Without NVLink/P2P, TP=2 all-reduce goes through host memory. A custom host-memory all-reduce was tried
  and was no faster than NCCL (branch `turing-ce-allreduce`, `benchmarks/turing/ce_allreduce/README.md`).
