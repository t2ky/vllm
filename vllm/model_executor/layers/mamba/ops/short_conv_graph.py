# SPDX-License-Identifier: Apache-2.0
"""Triton kernels that run the LFM2 ShortConv (y = C * causal_conv(B * x), L_cache=3) inside CUDA graphs.

No host-side decisions are used (prefill/decode split, token or request counts). The kernels only read
persistent GPU buffers (ShortConvGraphBuffers, one set per kv-cache group, same addresses every step):
  qsl[MAXR + 1]   token range of each request (entries past the real count are filled with the total token
                  count -> zero-length requests)
  slot[MAXR]      state slot (first index of kv_cache[0])
  init[MAXR]      whether a previous state exists (num_computed_tokens > 0)
  armed[1]        1 when this step's metadata was written; reset to 0 after the last layer of the group.
                  A safety latch so dummy runs that replay piecewise graphs without building metadata do not
                  overwrite real requests' states with stale metadata.
The grid depends only on the padded token count (the graph size) and MAXR, so it does not change per step.

Two kernels, run in order on the same stream:
  1. _conv_out_kernel: outputs for all tokens. Only reads the old state. Padding rows and rows that belong
     to no request are written as 0.
  2. _conv_state_kernel: writes the new state per request (B*x of the last 2 tokens). For length-1 requests
     the second entry of the old state shifts left (read then written by the same program). It runs after
     kernel 1 has finished, so reads and writes do not race.
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _find_req(t, qsl_ptr, MAXR: tl.constexpr, LOG2R: tl.constexpr):
    """For tokens t ([BT]), the largest r (0..MAXR) with qsl[r] <= t. Binary search."""
    lo = tl.zeros_like(t)
    hi = tl.zeros_like(t) + MAXR
    for _ in tl.static_range(LOG2R + 1):
        active = lo < hi
        mid = (lo + hi + 1) // 2
        v = tl.load(qsl_ptr + mid, mask=active, other=0)
        le = v <= t
        lo = tl.where(active & le, mid, lo)
        hi = tl.where(active & ~le, mid - 1, hi)
    return lo


@triton.jit
def _conv_out_kernel(bcx_ptr, y_ptr, w_ptr, bias_ptr, st_ptr, qsl_ptr, slot_ptr, init_ptr,
                     T, D, stride_bt, stride_yt, stride_wd, stride_sb, stride_sd, stride_ss,
                     HAS_BIAS: tl.constexpr, MAXR: tl.constexpr, LOG2R: tl.constexpr,
                     BT: tl.constexpr, BD: tl.constexpr):
    pt = tl.program_id(0)
    pd = tl.program_id(1)
    t = pt * BT + tl.arange(0, BT)
    d = pd * BD + tl.arange(0, BD)
    md = d < D
    in_t = t < T
    r = _find_req(t, qsl_ptr, MAXR, LOG2R)
    start = tl.load(qsl_ptr + r, mask=in_t, other=0)
    end = tl.load(qsl_ptr + r + 1, mask=in_t & (r < MAXR), other=0)
    valid = in_t & (r < MAXR) & (t >= start) & (t < end)
    pos = t - start
    slot = tl.load(slot_ptr + r, mask=valid, other=0).to(tl.int64)
    init = tl.load(init_ptr + r, mask=valid, other=0) != 0
    m2 = valid[:, None] & md[None, :]
    row = t.to(tl.int64)[:, None] * stride_bt

    # u = B * x (rounded to fp16/bf16 like vLLM)
    b0 = tl.load(bcx_ptr + row + d[None, :], mask=m2, other=0.0)
    x0 = tl.load(bcx_ptr + row + 2 * D + d[None, :], mask=m2, other=0.0)
    u0 = (b0 * x0).to(tl.float32)
    m1 = m2 & (pos >= 1)[:, None]
    row1 = row - stride_bt
    u1 = (tl.load(bcx_ptr + row1 + d[None, :], mask=m1, other=0.0)
          * tl.load(bcx_ptr + row1 + 2 * D + d[None, :], mask=m1, other=0.0)).to(tl.float32)
    mm2 = m2 & (pos >= 2)[:, None]
    row2 = row - 2 * stride_bt
    u2 = (tl.load(bcx_ptr + row2 + d[None, :], mask=mm2, other=0.0)
          * tl.load(bcx_ptr + row2 + 2 * D + d[None, :], mask=mm2, other=0.0)).to(tl.float32)
    # previous state: state[slot, d, 0] is t-2, state[slot, d, 1] is t-1 (0 without a previous state)
    sbase = st_ptr + slot[:, None] * stride_sb + d[None, :].to(tl.int64) * stride_sd
    ms = m2 & init[:, None]
    s0 = tl.load(sbase, mask=ms & (pos == 0)[:, None], other=0.0).to(tl.float32)
    s1 = tl.load(sbase + stride_ss, mask=ms & (pos <= 1)[:, None], other=0.0).to(tl.float32)
    p1 = tl.where((pos >= 1)[:, None], u1, s1)                                         # t-1
    p2 = tl.where((pos >= 2)[:, None], u2, tl.where((pos == 1)[:, None], s1, s0))      # t-2
    w0 = tl.load(w_ptr + d * stride_wd + 0, mask=md, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + d * stride_wd + 1, mask=md, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + d * stride_wd + 2, mask=md, other=0.0).to(tl.float32)
    acc = p2 * w0[None, :] + p1 * w1[None, :] + u0 * w2[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + d, mask=md, other=0.0).to(tl.float32)[None, :]
    c = tl.load(bcx_ptr + row + D + d[None, :], mask=m2, other=0.0)
    out = acc.to(c.dtype) * c
    out = tl.where(m2, out, 0.0)
    tl.store(y_ptr + t.to(tl.int64)[:, None] * stride_yt + d[None, :], out, mask=in_t[:, None] & md[None, :])


@triton.jit
def _conv_state_kernel(bcx_ptr, st_ptr, qsl_ptr, slot_ptr, init_ptr, armed_ptr,
                       D, stride_bt, stride_sb, stride_sd, stride_ss, BD: tl.constexpr):
    r = tl.program_id(0)
    pd = tl.program_id(1)
    if tl.load(armed_ptr) == 0:
        return
    start = tl.load(qsl_ptr + r)
    end = tl.load(qsl_ptr + r + 1)
    n = end - start
    if n <= 0:
        return
    d = pd * BD + tl.arange(0, BD)
    md = d < D
    slot = tl.load(slot_ptr + r).to(tl.int64)
    init = tl.load(init_ptr + r) != 0
    sbase = st_ptr + slot * stride_sb + d.to(tl.int64) * stride_sd
    last = (end - 1).to(tl.int64) * stride_bt
    u_last = tl.load(bcx_ptr + last + d, mask=md, other=0.0) * tl.load(bcx_ptr + last + 2 * D + d, mask=md, other=0.0)
    if n >= 2:
        prev = (end - 2).to(tl.int64) * stride_bt
        u_prev = tl.load(bcx_ptr + prev + d, mask=md, other=0.0) * tl.load(bcx_ptr + prev + 2 * D + d, mask=md, other=0.0)
        new0 = u_prev.to(st_ptr.dtype.element_ty)
    else:
        old1 = tl.load(sbase + stride_ss, mask=md & init, other=0.0)
        new0 = old1.to(st_ptr.dtype.element_ty)
    tl.store(sbase, new0, mask=md)
    tl.store(sbase + stride_ss, u_last.to(st_ptr.dtype.element_ty), mask=md)


@triton.jit
def _pack_kernel(qsl_src, seq_lens_src, sd_ptr, sp_ptr, qsl_dst, slot_dst, init_dst, armed_ptr,
                 num_reqs, num_d, stride_sd_row, MAXR: tl.constexpr, BR: tl.constexpr):
    """Called once per step from the builder: write this step's metadata into the persistent buffers (GPU only)."""
    r = tl.arange(0, BR)
    real = r < num_reqs
    total = tl.load(qsl_src + num_reqs)
    q0 = tl.load(qsl_src + r, mask=real, other=0)
    q1 = tl.load(qsl_src + r + 1, mask=real, other=0)
    q0 = tl.where(real, q0, total)
    in_buf = r <= MAXR
    tl.store(qsl_dst + r, q0, mask=in_buf)
    sl = tl.load(seq_lens_src + r, mask=real, other=0)
    init = (sl - (q1 - q0)) > 0
    isd = r < num_d
    s_d = tl.load(sd_ptr + r * stride_sd_row, mask=real & isd, other=0)
    s_p = tl.load(sp_ptr + (r - num_d), mask=real & ~isd, other=0)
    slot = tl.where(isd, s_d, s_p)
    mr = r < MAXR
    tl.store(slot_dst + r, tl.where(real, slot, 0), mask=mr)
    tl.store(init_dst + r, tl.where(real & init, 1, 0), mask=mr)
    tl.store(armed_ptr + r, 1, mask=r == 0)


class ShortConvGraphBuffers:
    """Persistent buffers per kv-cache group (per ShortConv metadata builder)."""

    def __init__(self, max_reqs: int, device):
        self.maxr = max_reqs
        self.qsl = torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
        self.slot = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.init = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.armed = torch.zeros(1, dtype=torch.int32, device=device)
        self.last_layer: str | None = None  # last layer of this group (resets armed)

    def write(self, query_start_loc, seq_lens, state_d, state_p, num_reqs: int, num_decodes: int):
        """query_start_loc[num_reqs+1], seq_lens[num_reqs], state_d[num_decodes, k] (column 0), state_p[num_prefills]"""
        assert num_reqs <= self.maxr, (num_reqs, self.maxr)
        if state_d is None or state_d.numel() == 0:
            state_d = self.slot[:1].view(1, 1)
        if state_d.dim() == 1:
            state_d = state_d.view(-1, 1)
        if state_p is None or state_p.numel() == 0:
            state_p = self.slot[:1]
        BR = triton.next_power_of_2(self.maxr + 1)
        _pack_kernel[(1,)](query_start_loc, seq_lens, state_d, state_p, self.qsl, self.slot, self.init, self.armed,
                           num_reqs, num_decodes, state_d.stride(0), MAXR=self.maxr, BR=BR, num_warps=4)


# layer_name -> ShortConvGraphBuffers (created by the builder, read by the op)
BUFFERS: dict[str, ShortConvGraphBuffers] = {}


def short_conv_graph_forward(bcx: torch.Tensor, y: torch.Tensor, conv_weights: torch.Tensor,
                             bias: torch.Tensor | None, conv_state: torch.Tensor, buf: ShortConvGraphBuffers,
                             clear_armed: bool, BT: int = 16, BD: int = 128):
    """bcx [T, 3D] (B, C, x) -> y [T, D] = C * causal_conv(B * x). conv_state: a (blocks, D, 2) view."""
    T = bcx.shape[0]
    D = y.shape[1]
    assert conv_state.shape[-1] == 2 and conv_weights.shape[-1] == 3
    maxr = buf.maxr
    log2r = max(1, (maxr).bit_length())
    if T > 0:
        _conv_out_kernel[(triton.cdiv(T, BT), triton.cdiv(D, BD))](
            bcx, y, conv_weights, bias if bias is not None else conv_weights, conv_state, buf.qsl, buf.slot, buf.init,
            T, D, bcx.stride(0), y.stride(0), conv_weights.stride(0),
            conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
            HAS_BIAS=bias is not None, MAXR=maxr, LOG2R=log2r, BT=BT, BD=BD, num_warps=4)
    _conv_state_kernel[(maxr, triton.cdiv(D, BD))](
        bcx, conv_state, buf.qsl, buf.slot, buf.init, buf.armed,
        D, bcx.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), BD=BD, num_warps=4)
    if clear_armed:
        buf.armed.zero_()
