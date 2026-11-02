# SPDX-License-Identifier: Apache-2.0
"""W4A16 grouped GEMM for MoE in Triton, reading vLLM's Marlin MoE weight layout as-is (written for the
LFM2 MoE models; any uint4b8 / group-32 / fp16 / SiLU Marlin MoE fits).

- ``moe_gemm(...)``: drop-in for ``ops.moe_wna16_marlin_gemm`` (uint4b8, group quantization, no bias / zero
  points, fp16). Input rows are sorted_token_ids // top_k, output rows are (token, k); mul_topk_weights has
  the same meaning. Used on the LoRA path (activation_func / moe_sum hooks keep working).
- ``fused_moe(...)``: fast path without LoRA: SiLU(gate) * up in the w13 epilogue, top-k weights in w2.
- ``pick(M)``: per-M configs (BM and BN / BK / warps / stages of each GEMM).

Reading the weights: the bits of a position inside Marlin's 16x64 tile are a permutation of (k, n) bits
(i0=k8 i1=n8 i2=k1 | i3=n16 i4=n32 | i5=k2 i6=k4 | i7..9=n1 n2 n4). Relabelling k and n with "logical"
indices (kappa, j) makes the int32 words loaded in Marlin order exactly the B operand registers of Triton's
mma (kWidth=4), so B never goes through shared memory. The k relabelling stays within 16, and A (the
activations) is permuted once up front to match (``permute_k``).
"""
import os

import torch
import triton
import triton.language as tl

_DEQ_ASM = tl.constexpr("""
{
.reg .b32 t, m1, m2, c, s, f, b, r0, r1, r2, r3;
mov.b32 m1, 0x000F000F;
mov.b32 m2, 0x00F000F0;
mov.b32 c, 0x64006400;
mov.b32 s, 0x64086408;
mov.b32 f, 0x2C002C00;
mov.b32 b, 0xD480D480;
lop3.b32 r0, $8, m1, c, 0xea;
lop3.b32 r1, $8, m2, c, 0xea;
shr.u32 t, $8, 8;
lop3.b32 r2, t, m1, c, 0xea;
lop3.b32 r3, t, m2, c, 0xea;
sub.f16x2 r0, r0, s;
fma.rn.f16x2 r1, r1, f, b;
sub.f16x2 r2, r2, s;
fma.rn.f16x2 r3, r3, f, b;
mov.b32 {$0, $1}, r0;
mov.b32 {$2, $3}, r1;
mov.b32 {$4, $5}, r2;
mov.b32 {$6, $7}, r3;
}
""")


@triton.jit
def _deq_nib(p):
    """int32 [...] → fp16 [..., 2, 2, 2]（uint4b8 → q - 8）。添字 (j0, j1, j2) がニブル j0 + 2 j1 + 4 j2"""
    o0, o4, o1, o5, o2, o6, o3, o7 = tl.inline_asm_elementwise(
        _DEQ_ASM, "=h,=h,=h,=h,=h,=h,=h,=h,r", [p],
        dtype=(tl.float16, tl.float16, tl.float16, tl.float16, tl.float16, tl.float16, tl.float16, tl.float16),
        is_pure=True, pack=1)
    lo = tl.join(tl.join(o0, o1), tl.join(o2, o3))
    hi = tl.join(tl.join(o4, o5), tl.join(o6, o7))
    return tl.join(lo, hi)


@triton.jit
def _kperm(c):
    """κ の並びの位置 c → 元の k（16 の中だけ）"""
    lo = c & 15
    return (c & ~15) + (lo & 1) + ((lo >> 1) & 1) * 8 + ((lo >> 2) & 1) * 2 + ((lo >> 3) & 1) * 4


@triton.jit
def _perm_a(a, BM: tl.constexpr, BK: tl.constexpr):
    """元の並びで読んだ A [BM, BK] を κ の並びに（レジスタの中の並べ替え。読みは 128 ビットのまま）"""
    a = tl.reshape(a, (BM, BK // 16, 2, 2, 2, 2))  # k8 k4 k2 k1
    a = tl.permute(a, (0, 1, 3, 4, 2, 5))          # κ8=k4 κ4=k2 κ2=k8 κ1=k1
    return tl.reshape(a, (BM, BK))


@triton.jit
def _load_a(a_rows, k0, offs_k, valid, A_SORTED: tl.constexpr, A_PERM: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    if A_PERM == 3:
        if A_SORTED:
            v = tl.load(a_rows + (k0 // 2 + offs_k)[None, :])
        else:
            v = tl.load(a_rows + (k0 // 2 + offs_k)[None, :], mask=valid[:, None], other=0)
        lo = (v & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        hi = (v >> 16).to(tl.int16).to(tl.float16, bitcast=True)
        a = tl.reshape(tl.join(lo, hi), (BM, BK))
    else:
        if A_SORTED:
            a = tl.load(a_rows + (k0 + offs_k)[None, :])
        else:
            a = tl.load(a_rows + (k0 + offs_k)[None, :], mask=valid[:, None], other=0.0)
        if A_PERM == 2:
            a = _perm_a(a, BM, BK)
    return a


@triton.jit
def _cpos(f):
    """元の k（特徴の番号）→ κ の並びの位置"""
    lo = f & 15
    return (f & ~15) | (lo & 1) | (((lo >> 3) & 1) << 1) | (((lo >> 1) & 1) << 2) | (((lo >> 2) & 1) << 3)


@triton.jit
def _jbits(j, NW: tl.constexpr):
    NTW: tl.constexpr = NW // 2
    lane = j & 7
    n32 = (j >> 3) & 1
    ntw = (j >> 4) & (NTW - 1)
    rest = j // (16 * NTW)
    return lane, n32, ntw, rest & 1, (rest >> 1) & 1, rest >> 2


@triton.jit
def _n_of_j(j, NW: tl.constexpr):
    """論理の列 j → タイルの中の本当の列 n"""
    NTW: tl.constexpr = NW // 2
    lane, n32, ntw, n8, n16, nth = _jbits(j, NW)
    return lane + n32 * 32 + n8 * 8 + n16 * 16 + (nth * NTW + ntw) * 64


@triton.jit
def _scale_pos(j, NW: tl.constexpr):
    """論理の列 j → Marlin の並びの scale の位置（64 列の中で 8(n%8) + (n//8)%8）"""
    NTW: tl.constexpr = NW // 2
    lane, n32, ntw, n8, n16, nth = _jbits(j, NW)
    return (nth * NTW + ntw) * 64 + lane * 8 + n8 + n16 * 2 + n32 * 4


@triton.jit
def _ld_raw(w_ptr, s_ptr, k0, n0, stride_wr, stride_sr,
            BK: tl.constexpr, BN: tl.constexpr, NW: tl.constexpr, GROUP: tl.constexpr):
    """Marlin の並びの [k0, k0+BK) × [n0, n0+BN) の int32 と scale を読むだけ（戻すのは _deq_b）"""
    NTW: tl.constexpr = NW // 2
    NTH: tl.constexpr = BN // 64 // NTW
    kt = (k0 // 16 + tl.arange(0, BK // 16)) * stride_wr
    th = tl.arange(0, NTH) * (NTW * 128)
    tw = tl.arange(0, NTW) * 128
    n32 = tl.arange(0, 2) * 2
    a8 = tl.arange(0, 8) * 16
    k42 = tl.arange(0, 4) * 4
    n16 = tl.arange(0, 2)
    off = (kt[:, None, None, None, None, None, None] + th[None, :, None, None, None, None, None]
           + tw[None, None, :, None, None, None, None] + n32[None, None, None, :, None, None, None]
           + a8[None, None, None, None, :, None, None] + k42[None, None, None, None, None, :, None]
           + n16[None, None, None, None, None, None, :])
    p = tl.load(w_ptr + (n0 // 64) * 128 + off)  # [KT, NTH, NTW, N32, N421, K42, N16]
    # scale は 2 つに分けて読む（スレッドの数より要素が少ない load は使い手のレイアウトで読み直される）
    j = tl.arange(0, BN // 2)
    g = k0 // GROUP + tl.arange(0, BK // GROUP)
    sp = s_ptr + g[:, None] * stride_sr + n0
    s_lo = tl.load(sp + _scale_pos(j, NW)[None, :])
    s_hi = tl.load(sp + _scale_pos(j + BN // 2, NW)[None, :])
    return p, s_lo, s_hi


@triton.jit
def _deq_b(p, s_lo, s_hi, BK: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr):
    """_ld_raw の結果 → 論理の (κ, j) の順の fp16 [BK, BN]（scale 込み）"""
    q = _deq_nib(p)                           # ... + [k8, n8, k1]
    q = tl.permute(q, (0, 5, 7, 9, 1, 6, 8, 2, 3, 4))  # κ = [KT, K42, k8, k1]、j = [NTH, N16, n8, NTW, N32, N421]
    q = tl.reshape(q, (BK // GROUP, GROUP, BN))
    s = tl.reshape(tl.permute(tl.join(s_lo, s_hi), (0, 2, 1)), (BK // GROUP, BN))
    return tl.reshape(q * s[:, None, :], (BK, BN))


@triton.jit
def _load_b(w_ptr, s_ptr, k0, n0, stride_wr, stride_sr,
            BK: tl.constexpr, BN: tl.constexpr, NW: tl.constexpr, GROUP: tl.constexpr):
    """Marlin の並びの [k0, k0+BK) × [n0, n0+BN) を論理の (κ, j) の順の fp16 [BK, BN] に（scale 込み）"""
    p, s_lo, s_hi = _ld_raw(w_ptr, s_ptr, k0, n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
    return _deq_b(p, s_lo, s_hi, BK, BN, GROUP)


@triton.jit
def _moe_kernel(a_ptr, w_ptr, s_ptr, c_ptr, sorted_ptr, expert_ptr, npad_ptr, tw_ptr,
                num_valid, K, HALF,
                stride_am, stride_we, stride_wr, stride_se, stride_sr, stride_cm,
                A_DIV: tl.constexpr, A_SORTED: tl.constexpr, A_PERM: tl.constexpr,
                GATED: tl.constexpr, MUL_TW: tl.constexpr, C_SORTED: tl.constexpr, C_KPERM: tl.constexpr,
                GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NW: tl.constexpr,
                PREFETCH: tl.constexpr = 0):
    """塊 pid_m（専門家 e の BM 行）× 列 [n0, n0+BN) を計算する。
    A の行: A_SORTED=0 なら sorted_ids // A_DIV、1 なら塊の中の位置（[npad 以上, K] の並べ替え済みの入力）
    A の列: A_PERM=0 なら κ の並び（前もって並べ替え済み）、1 なら元の並び（並べ替えた位置を読む。読みが 16 ビットずつになる）、
            2 なら元の並び（連続に読んでからレジスタの中で並べ替える）
    GATED=1: 列 n0 の gate と HALF + n0 の up を同時に計算し SiLU(gate)×up を書く（出力は HALF 列）
    C の行: C_SORTED=0 なら sorted_ids（(トークン, k) の通し番号）、1 なら塊の中の位置。C_KPERM=1 なら列を κ の並びで書く"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m * BM >= tl.load(npad_ptr):
        return
    e = tl.load(expert_ptr + pid_m)
    if e < 0:
        return
    offs_r = pid_m * BM + tl.arange(0, BM)
    ids = tl.load(sorted_ptr + offs_r)
    valid = ids < num_valid
    offs_k = tl.arange(0, BK)
    if A_PERM == 1:
        offs_k = _kperm(offs_k)
    if A_PERM == 3:
        offs_k = _kperm(tl.arange(0, BK // 2) * 2) // 2  # κ の対（κ1 = k1 なので k の対）の int32 の番号
    if A_PERM == 3:
        a_base = a_ptr.to(tl.pointer_type(tl.int32), bitcast=True)
        a_st = stride_am // 2
    else:
        a_base = a_ptr
        a_st = stride_am
    if A_SORTED:
        a_rows = a_base + offs_r[:, None].to(tl.int64) * a_st
    else:
        a_rows = a_base + (ids // A_DIV)[:, None].to(tl.int64) * a_st
    w_e = w_ptr + e.to(tl.int64) * stride_we
    s_e = s_ptr + e.to(tl.int64) * stride_se
    n0 = pid_n * BN
    if GATED:
        acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
        if PREFETCH:
            pg, sg_lo, sg_hi = _ld_raw(w_e, s_e, 0, n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
            pu, su_lo, su_hi = _ld_raw(w_e, s_e, 0, HALF + n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
            for k0 in range(0, K, BK):
                a = _load_a(a_rows, k0, offs_k, valid, A_SORTED, A_PERM, BM, BK)
                kn = tl.minimum(k0 + BK, K - BK)
                pg2, sg2_lo, sg2_hi = _ld_raw(w_e, s_e, kn, n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
                pu2, su2_lo, su2_hi = _ld_raw(w_e, s_e, kn, HALF + n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
                wg = _deq_b(pg, sg_lo, sg_hi, BK, BN, GROUP)
                wu = _deq_b(pu, su_lo, su_hi, BK, BN, GROUP)
                w = tl.reshape(tl.permute(tl.join(wg, wu), (0, 2, 1)), (BK, 2 * BN))
                acc = tl.dot(a, w, acc)
                pg, sg_lo, sg_hi = pg2, sg2_lo, sg2_hi
                pu, su_lo, su_hi = pu2, su2_lo, su2_hi
        else:
            for k0 in range(0, K, BK):
                a = _load_a(a_rows, k0, offs_k, valid, A_SORTED, A_PERM, BM, BK)
                wg = _load_b(w_e, s_e, k0, n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
                wu = _load_b(w_e, s_e, k0, HALF + n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
                w = tl.reshape(tl.permute(tl.join(wg, wu), (0, 2, 1)), (BK, 2 * BN))  # gate/up を j の最上位に
                acc = tl.dot(a, w, acc)
        acc_g, acc_u = tl.split(tl.permute(tl.reshape(acc, (BM, 2, BN)), (0, 2, 1)))
        g = acc_g.to(tl.float16).to(tl.float32)  # vLLM の silu_and_mul と同じく fp16 に丸めてから
        u = acc_u.to(tl.float16).to(tl.float32)
        y = g / (1.0 + tl.exp(-g)) * u
    else:
        y = tl.zeros((BM, BN), dtype=tl.float32)
        if PREFETCH:
            # 重みの int32 と scale を 1 周先に読んでおく（レジスタの二重バッファ。共有メモリは使わない）
            p, s_lo, s_hi = _ld_raw(w_e, s_e, 0, n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
            for k0 in range(0, K, BK):
                a = _load_a(a_rows, k0, offs_k, valid, A_SORTED, A_PERM, BM, BK)
                kn = tl.minimum(k0 + BK, K - BK)  # 最後の周は同じ所を読み直す（範囲の外を読まない）
                pn, sn_lo, sn_hi = _ld_raw(w_e, s_e, kn, n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
                w = _deq_b(p, s_lo, s_hi, BK, BN, GROUP)
                y = tl.dot(a, w, y)
                p, s_lo, s_hi = pn, sn_lo, sn_hi
        else:
            for k0 in range(0, K, BK):
                a = _load_a(a_rows, k0, offs_k, valid, A_SORTED, A_PERM, BM, BK)
                w = _load_b(w_e, s_e, k0, n0, stride_wr, stride_sr, BK, BN, NW, GROUP)
                y = tl.dot(a, w, y)
    if MUL_TW:
        tw = tl.load(tw_ptr + ids, mask=valid, other=0.0)
        y = y * tw[:, None]
    # 連続する 2 列（j の最下位 = n1）をまとめて書けるように [BM, BN/2, 2]
    y = tl.reshape(y.to(tl.float16), (BM, BN // 2, 2))
    col = n0 + _n_of_j(tl.arange(0, BN // 2) * 2, NW)
    if C_KPERM:
        col = _cpos(col)
    cols = col[None, :, None] + tl.arange(0, 2)[None, None, :]
    if C_SORTED:
        tl.store(c_ptr + offs_r[:, None, None].to(tl.int64) * stride_cm + cols, y)
    else:
        tl.store(c_ptr + ids[:, None, None].to(tl.int64) * stride_cm + cols, y, mask=valid[:, None, None])


_LAST = {}
_CACHE = {}
_FAST = os.environ.get("MOE_W4_FAST_LAUNCH", "1") == "1"


def last_shared():
    return _LAST["k"].metadata.shared


def _launch(a, w, s, c, sorted_ids, expert_ids, npad, tw, num_valid, K, n_out, half, *, a_div, a_sorted, a_perm,
            gated, mul_tw, c_sorted, c_kperm, BM, BN, BK, num_warps, num_stages, group=32, prefetch=0):
    assert num_warps >= 2 and BN >= 32 * num_warps and BN % 64 == 0 and n_out % BN == 0 and K % BK == 0
    assert BK % group == 0 and BK >= 32
    grid = (triton.cdiv(sorted_ids.numel(), BM), n_out // BN, 1)
    ints = (num_valid, K, half, a.stride(0), w.stride(0), w.stride(1), s.stride(0), s.stride(1), c.stride(0))
    const = (a_div, a_sorted, a_perm, gated, mul_tw, c_sorted, c_kperm, group, BM, BN, BK, num_warps, prefetch)
    ptrs = (a, w, s, c, sorted_ids, expert_ids, npad, tw)
    # Triton の JIT の起動（引数の特殊化の判定など）は 1 回 20µs 前後かかるので、コンパイル済みのカーネルを
    # 特殊化の鍵（int の 16 の倍数・1、ポインタの 16 バイト境界）ごとに覚えて直接起動する
    key = (const, num_stages, tuple((v % 16 == 0, v == 1) for v in ints), tuple(t.data_ptr() % 16 == 0 for t in ptrs),
           a.dtype, tw.dtype)
    k = _CACHE.get(key) if _FAST else None
    if k is None:
        k = _moe_kernel[grid[:2]](
            *ptrs, *ints,
            A_DIV=a_div, A_SORTED=a_sorted, A_PERM=a_perm, GATED=gated, MUL_TW=mul_tw, C_SORTED=c_sorted, C_KPERM=c_kperm,
            GROUP=group, BM=BM, BN=BN, BK=BK, NW=num_warps, PREFETCH=prefetch, num_warps=num_warps, num_stages=num_stages)
        _CACHE[key] = k
    else:
        k[grid](*ptrs, *ints, *const)
    _LAST["k"] = k
    return c


def moe_gemm(a, c, w, s, sorted_ids, expert_ids, npad, topk_weights, *, top_k, mul_topk_weights, size_m, size_n,
             size_k, BM, cfg, group=32):
    """ops.moe_wna16_marlin_gemm（uint4b8、group 量子化、bias/zero/a_scales なし）の差し替え。
    a: [size_m, size_k] fp16（元の並び）、c: [size_m * top_k, size_n] に (トークン, k) の行で書く。
    sorted_ids/expert_ids/npad は moe_align_block_size(topk_ids, BM, E) の結果（BM が同じであること）"""
    c = c.view(-1, size_n)
    a_perm = cfg.get("a_perm", 0)
    if a_perm == 0:
        a = permute_k(a)  # A の列を κ の並びに（読むときに並べ替えるより 1〜3 割速い。3090 で実測）
    kw = {k: v for k, v in cfg.items() if k != "a_perm"}
    return _launch(a, w, s, c, sorted_ids, expert_ids, npad, topk_weights.reshape(-1), size_m * top_k, size_k, size_n, 0,
                   a_div=top_k, a_sorted=0, a_perm=a_perm, gated=0, mul_tw=int(mul_topk_weights), c_sorted=0, c_kperm=0,
                   BM=BM, group=group, **kw)


KPERM16 = [(c & 1) + ((c >> 1) & 1) * 8 + ((c >> 2) & 1) * 2 + ((c >> 3) & 1) * 4 for c in range(16)]


def permute_k(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """[R, K] の列を κ の並びに（16 ごとに同じ並べ替え: 位置 c の 4 ビット (c1 c2 c4 c8) = 元の (k1 k8 k2 k4)）。
    添字の表を使わない strided なコピー 1 回"""
    R, K = x.shape
    v = x.view(R, K // 16, 2, 2, 2, 2).permute(0, 1, 3, 4, 2, 5)  # (k8 k4 k2 k1) → (k4 k2 k8 k1)
    if out is None:
        return v.reshape(R, K)
    out.view(R, K // 16, 2, 2, 2, 2).copy_(v)
    return out


def fused_moe(x, w13, s13, w2, s2, topk_weights, topk_ids, *, BM, cfg13, cfg2, x_perm=0, out=None, group=32,
              cache=None):
    """fused_marlin_moe（SiLU、LoRA なし）と同じ計算。w13 の epilogue で SiLU×mul、w2 で top-k の重みを掛ける。
    中間は「塊の中の位置」の行・κ の並びの列で持つ（w2 は A を連続に読める）。
    x_perm=0: 前もって permute_k で並べ替えてから読む（既定、速い）/ 1: 読むときに並べ替える"""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    M, K = x.shape
    E = w13.shape[0]
    I = s13.shape[2] // 2
    H = s2.shape[2]
    topk = topk_ids.shape[1]
    sorted_ids, expert_ids, npad = moe_align_block_size(topk_ids, BM, E)
    rows = sorted_ids.numel()
    h = torch.empty(rows, I, dtype=x.dtype, device=x.device)
    xa = x if x_perm else permute_k(x)
    tw = topk_weights.reshape(-1)
    _launch(xa, w13, s13, h, sorted_ids, expert_ids, npad, tw, M * topk, K, I, I,
            a_div=topk, a_sorted=0, a_perm=x_perm, gated=1, mul_tw=0, c_sorted=1, c_kperm=1, BM=BM, group=group, **cfg13)
    y = torch.empty(M * topk, H, dtype=x.dtype, device=x.device)
    _launch(h, w2, s2, y, sorted_ids, expert_ids, npad, tw, M * topk, I, H, 0,
            a_div=1, a_sorted=1, a_perm=0, gated=0, mul_tw=1, c_sorted=0, c_kperm=0, BM=BM, group=group, **cfg2)
    if out is None:
        out = torch.empty(M, H, dtype=x.dtype, device=x.device)
    return torch.sum(y.view(M, topk, H), dim=1, out=out)


# M (tokens) -> {"BM", "w13", "w2", "BM_g", "g13", "g2"} (w13 / w2: fast path, g13 / g2: moe_gemm on the LoRA
# path). Default table configs/moe_w4_sm75.json was measured on an RTX 2080; MOE_W4_CONFIG overrides it.
_TABLE = None


def pick(M: int):
    global _TABLE
    if _TABLE is None:
        import json
        path = os.environ.get("MOE_W4_CONFIG", os.path.join(os.path.dirname(__file__), "configs", "moe_w4_sm75.json"))
        _TABLE = {int(k): v for k, v in json.load(open(path)).items()}
    keys = sorted(_TABLE)
    best = keys[0]
    for k in keys:
        if k <= M:
            best = k
    return _TABLE[best]
