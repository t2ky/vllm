# SPDX-License-Identifier: Apache-2.0
"""W4A16 / W8A16 GEMM for Turing (sm_75) in CUDA C++ (csrc/turing/w4a16_sm75.cu).

- ``gemm(a, w, s, **cfg)``: C[M, N] = A[M, K] @ W. ``w`` / ``s`` stay in vLLM's dense Marlin layout (output of
  gptq_marlin_repack / marlin_permute_scales): uint4b8 (w [K/16, 2N]) or uint8b128 (w [K/16, 4N], e.g. an
  8-bit lm_head), group 32, no zero points or bias, fp16. The bit width follows from the shape of ``w``.
  The int4 tiles are loaded straight into registers as mma B fragments (B does not go through shared memory).
- ``split > 1``: split-K in one kernel. Each split writes its partial sums to its own workspace plane; the
  last CTA of a tile (atomic ticket) adds the planes in split order and writes fp16 (deterministic).
- ``pick(M, K, N)``: the config for the largest table M <= M from the config table ({"K,N": {"M": cfg}},
  8-bit shapes use "K,N,8"). Shapes missing from the table use the nearest shape (log distance) whose tiles
  divide K and N. Entries marked use_marlin return None (Marlin was faster there).

Environment variables: CUDA_W4_CONFIG (config table, default configs/w4a16_sm75.json, measured on an
RTX 2080), CUDA_W4_SPLIT_WS_MAX (max split-K workspace in bytes, default 32 MiB).
"""

import json
import math
import os

from vllm.turing._ext import load_ext

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPLIT_WS_MAX = int(os.environ.get("CUDA_W4_SPLIT_WS_MAX", 32 << 20))


def ext():
    return load_ext("cuda_w4_ext")


def gemm(a, w, s, *, BM, BN, BK, WM, WN, MINB=1, split=1, group_m=8, out=None):
    """a: [M, K] fp16, w: Marlin layout [K/16, 2N] int32, s: Marlin-permuted scales [K/32, N] fp16 -> [M, N] fp16"""
    if a.stride(1) != 1 or a.stride(0) % 8:
        a = a.contiguous()
    # do not split when the split-K workspace (fp32, splits x M x N) would get too large
    # (happens when a small-M config from the table is used for a larger M)
    while split > 1 and split * a.shape[0] * s.shape[1] * 4 > _SPLIT_WS_MAX:
        split //= 2
    return ext().gemm(a, w, s, out, BM, BN, BK, WM, WN, MINB, split, group_m)


def configs(bits=4):
    return [dict(zip(("BM", "BN", "BK", "WM", "WN", "MINB"), c)) for c in ext().configs(bits)]


def table_key(K, N, bits=4):
    """Config table key: "K,N" for 4-bit, "K,N,8" for 8-bit."""
    return f"{K},{N}" if bits == 4 else f"{K},{N},{bits}"


_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        path = os.environ.get("CUDA_W4_CONFIG", os.path.join(_HERE, "configs", "w4a16_sm75.json"))
        raw = json.load(open(path)) if os.path.exists(path) else {}
        _TABLE = {}
        for k, v in raw.items():
            t = tuple(map(int, k.split(",")))
            _TABLE[t if len(t) == 3 else t + (4,)] = {int(m): c for m, c in v.items()}
    return _TABLE


def _fits(c, K, N):
    return N % c["BN"] == 0 and K % c["BK"] == 0


def pick(M: int, K: int, N: int, ignore_marlin: bool = False, bits: int = 4):
    """Config for the largest table M <= M (missing shapes: nearest (K, N) in log distance that divides).
    Returns None for entries marked use_marlin (unless ignore_marlin=True)."""
    tab = _table()
    t = tab.get((K, N, bits))
    if t is None:
        ok = [kn for kn in tab if kn[2] == bits
              and all(_fits(c, K, N) for c in tab[kn].values() if ignore_marlin or not c.get("use_marlin"))]
        if not ok:
            return None
        kn = min(ok, key=lambda kn: abs(math.log(kn[0] / K)) + abs(math.log(kn[1] / N)))
        t = tab[kn]
    ms = sorted(t)
    best = ms[0]
    for m in ms:
        if m <= M:
            best = m
    c = t[best]
    if (c.get("use_marlin") and not ignore_marlin) or not _fits(c, K, N):
        return None
    return {k: v for k, v in c.items() if k != "use_marlin"}
