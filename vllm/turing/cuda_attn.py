# SPDX-License-Identifier: Apache-2.0
"""Decode attention for Turing (sm_75) in CUDA C++ (csrc/turing/decode_attn_sm75.cu).

- ``decode(q, out, kv_cache, head_size, block_table, seq_lens, query_start_loc, window, scale, softcap)``:
  batches where every request has query length 1. The KV cache is int8_per_token_head (with the 16-byte
  aligned slots of vllm-project/vllm#62113). No host sync or device-to-host copy, so it can be captured in
  CUDA graphs. The grid is (len(seq_lens), num_kv_heads, NSPLIT) and depends only on shapes.
- One workspace (split partial sums and counters) per shape; it is re-allocated when too small, and old
  workspaces are kept alive because captured CUDA graphs may point to them.
- Split-KV: each CTA derives its range from seq_len on the GPU, and the last CTA to finish combines the
  splits (single kernel; the counter resets itself).

Tuning environment variables:
  CUDA_ATTN_NSPLIT_SWA   max splits for sliding-window layers (default 16)
  CUDA_ATTN_NSPLIT_FULL  max splits for full-attention layers (default 16)
  CUDA_ATTN_TARGET       target CTA count = SMs x this (default 4);
                         splits = min(max, ceil(target / (num_seqs x kv_heads)))
  CUDA_ATTN_MIN_CHUNK_SWA / _FULL  min tokens per split (default 64 / 128); CUDA_ATTN_MIN_CHUNK sets both
"""

import os

import torch

from vllm.turing._ext import load_ext


def ext():
    return load_ext("cuda_attn_ext")


def nsplit_max(window):
    if window and window > 0:
        return int(os.environ.get("CUDA_ATTN_NSPLIT_SWA", "16"))
    return int(os.environ.get("CUDA_ATTN_NSPLIT_FULL", "16"))


def min_chunk(window=0):
    """Minimum tokens per split. Sliding-window layers: 64, global layers: 128 (global layers with GQA 8 and
    head_dim 512 have 16 KB of partial sums per split, so finer splits cost more in the combine step)."""
    if os.environ.get("CUDA_ATTN_MIN_CHUNK"):
        return int(os.environ["CUDA_ATTN_MIN_CHUNK"])
    if window and window > 0:
        return int(os.environ.get("CUDA_ATTN_MIN_CHUNK_SWA", "64"))
    return int(os.environ.get("CUDA_ATTN_MIN_CHUNK_FULL", "128"))


_SMS = {}


def nsplit_for(device, nseq, nkv, window):
    """Grid z (number of splits), from shapes (num seqs, kv heads) and the SM count only."""
    sms = _SMS.get(str(device))
    if sms is None:
        sms = _SMS[str(device)] = torch.cuda.get_device_properties(device).multi_processor_count
    target = float(os.environ.get("CUDA_ATTN_TARGET", "4")) * sms
    return max(1, min(nsplit_max(window), -(-int(target) // max(1, nseq * nkv))))


_WS = {}
_KEEP = []  # replaced workspaces (captured CUDA graphs may still point to them, so never freed)


def workspace(device, nseq, hq, nkv, hd, nsplit_cap, reserve=0):
    key = (str(device), hq, nkv, hd, nsplit_cap)
    ws = _WS.get(key)
    if ws is None or ws[0] < nseq:
        n = max(nseq, reserve, ws[0] * 2 if ws else 0)
        if ws is not None:
            _KEEP.append(ws)
        ws = (n,
              torch.empty(n * hq * nsplit_cap * hd, dtype=torch.float32, device=device),
              torch.empty(n * hq * nsplit_cap * 2, dtype=torch.float32, device=device),
              torch.zeros(n * nkv, dtype=torch.int32, device=device))
        _WS[key] = ws
    return ws


def decode(q, out, kv_cache, head_size, block_table, seq_lens, query_start_loc, window=0, scale=1.0, softcap=0.0,
           nsplit=None, min_chunk_=None, reserve=0):
    """q / out: [T, Hq, HD] fp16. kv_cache: vLLM's int8 cache, logical (num_blocks, nkv, block_size, 2*phs).
    window: sliding window length (vLLM's sliding_window; 0 for full attention)."""
    nseq = seq_lens.shape[0]
    g = q.shape[1] // kv_cache.shape[1]
    nkv = kv_cache.shape[1] * (-(-g // 8))  # GQA > 8 is split into groups of <= 8 (more CTAs on grid y)
    cap = max(nsplit_max(window), nsplit or 0)
    nsplit = nsplit or nsplit_for(q.device, nseq, nkv, window)
    mc = min_chunk_ or min_chunk(window)
    _, wo, wml, cnt = workspace(q.device, nseq, q.shape[1], nkv, q.shape[2], cap, reserve)  # nkv = grid y
    ext().decode(q, out, kv_cache, int(head_size), block_table, seq_lens, query_start_loc, wo, wml, cnt,
                 int(window or 0), float(scale), float(softcap or 0.0), int(nsplit), int(mc))
    return out
