// Turing（sm_75）向けの decode（query の長さ 1）の paged attention。KV は int8（トークン × ヘッドごとの fp32 scale）。
//
// KV キャッシュの並び（vLLM の TRITON_ATTN、PR #62113 の int8_per_token_head）:
//   論理 (num_blocks, num_kv_heads, block_size, 2*phs) の int8。1 つの (block, head, slot) の中身は
//   [K(hs) | K の scale fp32 (4B) | pad | V(hs) | V の scale fp32 | pad]、phs = round_up(hs + 4, 16)。
//   ストライドは任意（HND / NHD どちらでも）。16 バイト単位であることだけ要る。
//
// 1 CTA = (seq, kv_head, split)。ワープ w は head_dim の [64w, 64w+64) を受け持つ（NW = HD/64 ワープ）。
// - QK: S^T[tok][head] = K[tok][d] · Q^T[d][head] を mma.m16n8k8（A=K: 16 トークン、B=Q^T: n=8 ヘッド）で、
//   各ワープが自分の 64 次元ぶんの部分和を出し、共有メモリで足し合わせる（1 タイル 32 トークンに __syncthreads 1 回）
// - PV: O^T[d][head] += V^T[d][tok] · P^T[tok][head] を mma.m16n8k8（A=V^T: 16 次元、B=P^T: n=8 ヘッド）
// - int8 → fp16 は prmt で 0x64 を重ねて 1152 を引く（値は整数のまま正確）。K の scale は S に、V の scale は P に掛ける
//   （vLLM の Triton と同じ置き方）。内積の d の並びは Q 側も同じに並べ替えてあるので、K/V は 16B / 8B の塊で読める
// - split-KV: グリッドは (num_seqs, num_kv_heads, NSPLIT) で固定（CUDA グラフに入る）。各 CTA は GPU 上の seq_len から
//   自分の範囲を決める（chunk = max(min_chunk, ceil(n/NSPLIT)) を 32 に切り上げ、使わない split は即終了）。
//   分けたときは部分（m, l, 正規化前の O）を作業領域に書き、(seq, kv_head) ごとのカウンタで最後に着いた CTA が
//   合成して書き、カウンタを 0 に戻す（カーネル 1 本、ホストとの往復なし）
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace cattn {

constexpr int TT = 32;      // 1 タイルのトークン数
constexpr int SROW = 36;    // 共有メモリの S の 1 行（トークン方向、32 + 4 で bank をずらす）
constexpr float LOG2E = 1.4426950408889634f;

struct Params {
  const half* q; int64_t q_s0, q_s1;          // [tokens, Hq, HD]（要素）
  half* out; int64_t o_s0, o_s1;
  const int8_t* kv;                           // KV キャッシュの先頭（バイト）
  int64_t s_blk, s_slot, s_head;              // バイト
  int k_off, ks_off, v_off, vs_off;           // (block, head, slot) の中のオフセット（バイト）
  const int* block_table; int64_t bt_s0; int block_size;
  const int* seq_lens; const int* qsl;
  int nkv, G, Hq, gsplit;                     // G = 1 CTA が受け持つ q ヘッド数（≤ 8）。GQA > 8 は gsplit 個の CTA に分ける
  int window;                                 // 0 なら全体
  float scale, softcap;
  int min_chunk;
  float* ws_o; float* ws_ml; int* counters;   // 分けたときの作業領域
};

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

// 符号つき int8 ×4（バイト 0..3）→ half2 {b0,b1}, {b2,b3}
__device__ __forceinline__ void i8x4_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  w ^= 0x80808080u;
  uint32_t a = prmt(w, 0x64646464u, 0x4140u);
  uint32_t b = prmt(w, 0x64646464u, 0x4342u);
  const uint32_t m = 0x64806480u;  // 1152
  half2 ha = __hsub2(*reinterpret_cast<half2*>(&a), *reinterpret_cast<const half2*>(&m));
  half2 hb = __hsub2(*reinterpret_cast<half2*>(&b), *reinterpret_cast<const half2*>(&m));
  lo = *reinterpret_cast<uint32_t*>(&ha);
  hi = *reinterpret_cast<uint32_t*>(&hb);
}

__device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ldg8(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ float ldgf(const void* p) {
  float r;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

#ifndef CA_MINB_512
#define CA_MINB_512 1
#endif
#ifndef CA_MINB_256
#define CA_MINB_256 1
#endif
template <int HD, bool SOFTCAP>
__global__ void __launch_bounds__(HD / 2, HD == 512 ? CA_MINB_512 : (HD == 256 ? CA_MINB_256 : 1))
decode_kernel(const Params p) {
  constexpr int NW = HD / 64;
  __shared__ float sS[2][NW][8][SROW];
  __shared__ float sW[8][32], sLs[8][32];  // 合成の重み・l（ヘッド × split）
  __shared__ int sLast;

  const int seq = blockIdx.x, vy = blockIdx.y, kvh = vy / p.gsplit, split = blockIdx.z, nsplit = gridDim.z;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane >> 2, c = lane & 3;
  const int G = p.G;

  const int q0 = p.qsl[seq];
  if (p.qsl[seq + 1] - q0 != 1) return;  // 詰め物の行（query の長さ 0）
  const int L = p.seq_lens[seq];
  const int lo = p.window > 0 ? max(0, L - p.window) : 0;
  const int n = L - lo;
  int nsp = 1, chunk = 0;
  if (n > 0) {
    chunk = max((n + nsplit - 1) / nsplit, p.min_chunk);
    chunk = (chunk + TT - 1) / TT * TT;
    nsp = (n + chunk - 1) / chunk;
  }
  if (split >= nsp) return;
  const int hq0 = vy * G;  // = kvh * (G * gsplit) + (vy % gsplit) * G
  if (n <= 0) {  // 見るものが無い: 0 を書く
    for (int i = tid; i < G * HD / 8; i += blockDim.x) {
      int h = i / (HD / 8), d8 = i % (HD / 8);
      *reinterpret_cast<uint4*>(p.out + q0 * p.o_s0 + (hq0 + h) * p.o_s1 + d8 * 8) = make_uint4(0, 0, 0, 0);
    }
    return;
  }
  const int t_begin = lo + split * chunk, t_end = min(L, t_begin + chunk);
  const int ntile = (t_end - t_begin + TT - 1) / TT;

  // Q（B 断片）: スレッド (r,c) はヘッド r の d = 64w + 16c + 2s + {0,1}（s = 0..7）
  uint32_t qf[8];
  if (r < G) {
    const half* qp = p.q + q0 * p.q_s0 + (hq0 + r) * p.q_s1 + 64 * w + 16 * c;
    uint4 a = *reinterpret_cast<const uint4*>(qp), b = *reinterpret_cast<const uint4*>(qp + 8);
    qf[0] = a.x; qf[1] = a.y; qf[2] = a.z; qf[3] = a.w; qf[4] = b.x; qf[5] = b.y; qf[6] = b.z; qf[7] = b.w;
  } else {
#pragma unroll
    for (int s = 0; s < 8; s++) qf[s] = 0;
  }

  const int8_t* kvb = p.kv;
  const int* bt = p.block_table + seq * p.bt_s0;
  const int bs = p.block_size;
  // レーン l はタイルの l 番目のトークンを受け持つ。block table は 1 タイル先に読んでおく（btv）
  auto lane_tok = [&](int it) {
    int tok = t_begin + it * TT + lane;
    return tok < t_end ? tok : t_end - 1;
  };
  auto bt_load = [&](int it) { return __ldg(bt + lane_tok(it) / bs); };
  struct Tile {
    uint4 k[2][2];
    uint2 v[4][2];
    float ks, vs;
  };
  // タイル it の K/V/scale を読み始める（btv はそのタイルの block 番号、前の周に読んである）
  auto issue = [&](Tile& T, int it, int btv) {
    int tc = lane_tok(it);
    int64_t o64 = (int64_t)btv * p.s_blk + (int64_t)(tc % bs) * p.s_slot + (int64_t)kvh * p.s_head;
    uint32_t off = (uint32_t)(o64 >> 4);
    T.ks = ldgf(kvb + o64 + p.ks_off);
    T.vs = ldgf(kvb + o64 + p.vs_off);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t o = __shfl_sync(0xffffffffu, off, 16 * i + r + 8 * h);
        T.k[i][h] = ldg16(kvb + (int64_t)o * 16 + p.k_off + 64 * w + 16 * c);
      }
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        uint32_t o = __shfl_sync(0xffffffffu, off, 8 * j + 2 * c + e);
        T.v[j][e] = ldg8(kvb + (int64_t)o * 16 + p.v_off + 64 * w + 8 * r);
      }
  };

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int k = 0; k < 4; k++) acc[i][k] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;
  const float sc = SOFTCAP ? p.scale / p.softcap : p.scale * LOG2E;

  auto compute = [&](const Tile& T, int it) {
    // ---- QK（このワープの 64 次元ぶんの部分和）
    float sacc[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int k = 0; k < 4; k++) sacc[i][k] = 0.f;
      const uint32_t* w0 = reinterpret_cast<const uint32_t*>(&T.k[i][0]);
      const uint32_t* w1 = reinterpret_cast<const uint32_t*>(&T.k[i][1]);
#pragma unroll
      for (int wd = 0; wd < 4; wd++) {
        uint32_t l0, h0, l1, h1;
        i8x4_h2(w0[wd], l0, h0);
        i8x4_h2(w1[wd], l1, h1);
        mma1688(sacc[i], l0, l1, qf[2 * wd]);
        mma1688(sacc[i], h0, h1, qf[2 * wd + 1]);
      }
    }
    float(*buf)[8][SROW] = sS[it & 1];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      buf[w][2 * c][16 * i + r] = sacc[i][0];
      buf[w][2 * c + 1][16 * i + r] = sacc[i][1];
      buf[w][2 * c][16 * i + r + 8] = sacc[i][2];
      buf[w][2 * c + 1][16 * i + r + 8] = sacc[i][3];
    }
    if (NW > 1) __syncthreads(); else __syncwarp();
    // ---- softmax（スレッド (r,c) はヘッド r、トークン 8j + 2c + e）
    float x[4][2];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 s = make_float2(0.f, 0.f);
#pragma unroll
      for (int ww = 0; ww < NW; ww++) {
        float2 t = *reinterpret_cast<const float2*>(&buf[ww][r][8 * j + 2 * c]);
        s.x += t.x; s.y += t.y;
      }
      x[j][0] = s.x; x[j][1] = s.y;
    }
    const int tbase = t_begin + it * TT;
    float mt = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int tt = 8 * j + 2 * c + e;
        float ks = __shfl_sync(0xffffffffu, T.ks, tt);
        float v = x[j][e] * ks * sc;
        if (SOFTCAP) v = p.softcap * tanhf(v) * LOG2E;
        v = (tbase + tt < t_end) ? v : -INFINITY;
        x[j][e] = v;
        mt = fmaxf(mt, v);
      }
    mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
    mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
    const float m_new = fmaxf(m_run, mt);
    const float alpha = exp2f(m_run - m_new);
    m_run = m_new;
    uint32_t pf[4];
    float ls = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float pv[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int tt = 8 * j + 2 * c + e;
        float vs = __shfl_sync(0xffffffffu, T.vs, tt);
        float pe = exp2f(x[j][e] - m_new);
        ls += pe;
        pv[e] = (tbase + tt < t_end) ? pe * vs : 0.f;
      }
      half2 h = __floats2half2_rn(pv[0], pv[1]);
      pf[j] = *reinterpret_cast<uint32_t*>(&h);
    }
    l_run = l_run * alpha + ls;
    const float a0 = __shfl_sync(0xffffffffu, alpha, 8 * c), a1 = __shfl_sync(0xffffffffu, alpha, 8 * c + 4);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      acc[i][0] *= a0; acc[i][1] *= a1; acc[i][2] *= a0; acc[i][3] *= a1;
    }
    // ---- PV
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int wd = 0; wd < 2; wd++) {
        uint32_t A = wd ? T.v[j][0].y : T.v[j][0].x, B = wd ? T.v[j][1].y : T.v[j][1].x;
        uint32_t xx = prmt(A, B, 0x5140u), yy = prmt(A, B, 0x7362u);
        uint32_t l0, h0, l1, h1;
        i8x4_h2(xx, l0, h0);
        i8x4_h2(yy, l1, h1);
        mma1688(acc[2 * wd], l0, h0, pf[j]);
        mma1688(acc[2 * wd + 1], l1, h1, pf[j]);
      }
    }
  };

  // 2 周ずつ展開し、レジスタの面 TA / TB を入れ替えて使う（移しで先読みの完了を待たないように）
  Tile TA, TB;
  int btv = bt_load(0);
  issue(TA, 0, btv);
  if (ntile > 1) btv = bt_load(1);
  for (int it = 0; it < ntile; it += 2) {
    if (it + 1 < ntile) {
      issue(TB, it + 1, btv);
      if (it + 2 < ntile) btv = bt_load(it + 2);
    }
    compute(TA, it);
    if (it + 1 >= ntile) break;
    if (it + 2 < ntile) {
      issue(TA, it + 2, btv);
      if (it + 3 < ntile) btv = bt_load(it + 3);
    }
    compute(TB, it + 1);
  }

  // ---- 後始末: l をヘッド r ごとに（c の 4 レーンで）合計
  l_run += __shfl_xor_sync(0xffffffffu, l_run, 1);
  l_run += __shfl_xor_sync(0xffffffffu, l_run, 2);
  // スレッド (r,c) の acc はヘッド 2c+e、d = 64w + 8r + {2i, 2i+1}
  const float lh[2] = {__shfl_sync(0xffffffffu, l_run, 8 * c), __shfl_sync(0xffffffffu, l_run, 8 * c + 4)};
  if (nsp == 1) {
#pragma unroll
    for (int e = 0; e < 2; e++) {
      int h = 2 * c + e;
      if (h < G) {
        float inv = 1.f / lh[e];
        half2 o[4];
#pragma unroll
        for (int i = 0; i < 4; i++) o[i] = __floats2half2_rn(acc[i][e] * inv, acc[i][2 + e] * inv);
        *reinterpret_cast<uint4*>(p.out + q0 * p.o_s0 + (hq0 + h) * p.o_s1 + 64 * w + 8 * r) =
            *reinterpret_cast<uint4*>(o);
      }
    }
    return;
  }
  // 部分を書く: ws_o[seq][hq][split][HD]、ws_ml[seq][hq][split][2]
#pragma unroll
  for (int e = 0; e < 2; e++) {
    int h = 2 * c + e;
    if (h < G) {
      float* o = p.ws_o + (((int64_t)seq * p.Hq + hq0 + h) * nsplit + split) * HD + 64 * w + 8 * r;
      *reinterpret_cast<float4*>(o) = make_float4(acc[0][e], acc[0][2 + e], acc[1][e], acc[1][2 + e]);
      *reinterpret_cast<float4*>(o + 4) = make_float4(acc[2][e], acc[2][2 + e], acc[3][e], acc[3][2 + e]);
    }
  }
  if (w == 0 && c == 0 && r < G) {
    float* ml = p.ws_ml + (((int64_t)seq * p.Hq + hq0 + r) * nsplit + split) * 2;
    ml[0] = m_run; ml[1] = l_run;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    int prev = atomicAdd(p.counters + seq * gridDim.y + vy, 1);
    sLast = (prev == nsp - 1);
  }
  __syncthreads();
  if (!sLast) return;
  __threadfence();
  if (tid == 0) p.counters[seq * gridDim.y + vy] = 0;  // 次の呼び出し（CUDA グラフの次の再生）のために戻す
  // 合成の重み: sW[h][s] = 2^(m_s - M) / Σ 2^(m_s - M) l_s。(m, l) はスレッドごとに 1 つずつ並べて読む
  for (int i = tid; i < G * nsp; i += blockDim.x) {
    int h = i / nsp, s = i % nsp;
    float2 ml = __ldcg(reinterpret_cast<const float2*>(p.ws_ml + (((int64_t)seq * p.Hq + hq0 + h) * nsplit + s) * 2));
    sW[h][s] = ml.x;
    sLs[h][s] = ml.y;
  }
  __syncthreads();
  if (tid < G) {
    float M = -INFINITY;
    for (int s = 0; s < nsp; s++) M = fmaxf(M, sW[tid][s]);
    float den = 0.f;
    for (int s = 0; s < nsp; s++) {
      float f = exp2f(sW[tid][s] - M);
      sW[tid][s] = f;
      den += f * sLs[tid][s];
    }
    float inv = 1.f / den;
    for (int s = 0; s < nsp; s++) sW[tid][s] *= inv;
  }
  __syncthreads();
  for (int i = tid; i < G * HD / 4; i += blockDim.x) {
    int h = i / (HD / 4), d4 = (i % (HD / 4)) * 4;
    const float* o = p.ws_o + ((int64_t)seq * p.Hq + hq0 + h) * nsplit * HD + d4;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s0 = 0; s0 < nsp; s0 += 8) {  // 8 split ぶんのロードをまとめて出す（1 つずつ待たない）
      float4 t[8];
#pragma unroll
      for (int u = 0; u < 8; u++)
        t[u] = (s0 + u < nsp) ? __ldcg(reinterpret_cast<const float4*>(o + (s0 + u) * HD)) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int u = 0; u < 8; u++) {
        float f = (s0 + u < nsp) ? sW[h][s0 + u] : 0.f;
        a.x += f * t[u].x; a.y += f * t[u].y; a.z += f * t[u].z; a.w += f * t[u].w;
      }
    }
    half2 h01 = __floats2half2_rn(a.x, a.y), h23 = __floats2half2_rn(a.z, a.w);
    uint2 u = make_uint2(*reinterpret_cast<uint32_t*>(&h01), *reinterpret_cast<uint32_t*>(&h23));
    *reinterpret_cast<uint2*>(p.out + q0 * p.o_s0 + (hq0 + h) * p.o_s1 + d4) = u;
  }
}

template <int HD, bool SC>
void launch(const Params& p, int nseq, int nkv, int nsplit, cudaStream_t st) {
  decode_kernel<HD, SC><<<dim3(nseq, nkv * p.gsplit, nsplit), HD / 2, 0, st>>>(p);
}

// q [T, Hq, HD] fp16、out 同じ、kv: 論理 (num_blocks, nkv, block_size, 2*phs) の int8
void decode(torch::Tensor q, torch::Tensor out, torch::Tensor kv, int64_t hs, torch::Tensor block_table,
            torch::Tensor seq_lens, torch::Tensor qsl, torch::Tensor ws_o, torch::Tensor ws_ml,
            torch::Tensor counters, int64_t window, double scale, double softcap, int64_t nsplit,
            int64_t min_chunk) {
  TORCH_CHECK(q.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf);
  TORCH_CHECK(kv.element_size() == 1 && kv.dim() == 4);
  TORCH_CHECK(q.stride(2) == 1 && out.stride(2) == 1);
  const int HD = q.size(2), Hq = q.size(1);
  const int nkv = kv.size(1), bs = kv.size(2), phs = kv.size(3) / 2;
  TORCH_CHECK(HD == hs && Hq % nkv == 0, "形が合わない");
  const int Gfull = Hq / nkv, gsplit = (Gfull + 7) / 8;
  TORCH_CHECK(Gfull % gsplit == 0, "GQA を 8 以下の組に等分できない");
  TORCH_CHECK(nsplit >= 1 && nsplit <= 32);
  Params p;
  p.q = reinterpret_cast<const half*>(q.data_ptr()); p.q_s0 = q.stride(0); p.q_s1 = q.stride(1);
  p.out = reinterpret_cast<half*>(out.data_ptr()); p.o_s0 = out.stride(0); p.o_s1 = out.stride(1);
  p.kv = reinterpret_cast<const int8_t*>(kv.data_ptr());
  p.s_blk = kv.stride(0); p.s_head = kv.stride(1); p.s_slot = kv.stride(2);
  TORCH_CHECK(kv.stride(3) == 1 && p.s_blk % 16 == 0 && p.s_head % 16 == 0 && p.s_slot % 16 == 0 && phs % 16 == 0 &&
                  (reinterpret_cast<uintptr_t>(p.kv) & 15) == 0,
              "KV の並びが 16 バイトにそろっていない");
  TORCH_CHECK((int64_t)kv.numel() / 16 < (int64_t)UINT32_MAX);
  p.k_off = 0; p.ks_off = hs; p.v_off = phs; p.vs_off = phs + hs;
  p.block_table = block_table.data_ptr<int>(); p.bt_s0 = block_table.stride(0); p.block_size = bs;
  p.seq_lens = seq_lens.data_ptr<int>(); p.qsl = qsl.data_ptr<int>();
  p.nkv = nkv; p.gsplit = gsplit; p.G = Gfull / gsplit; p.Hq = Hq;
  p.window = window; p.scale = scale; p.softcap = softcap; p.min_chunk = min_chunk;
  const int nseq = seq_lens.size(0);
  p.ws_o = ws_o.data_ptr<float>(); p.ws_ml = ws_ml.data_ptr<float>(); p.counters = counters.data_ptr<int>();
  TORCH_CHECK(ws_o.numel() >= (int64_t)nseq * Hq * nsplit * HD && ws_ml.numel() >= (int64_t)nseq * Hq * nsplit * 2 &&
              counters.numel() >= (int64_t)nseq * nkv * gsplit, "作業領域が小さい");
  TORCH_CHECK(qsl.size(0) >= nseq + 1);
  if (nseq == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  const bool sc = softcap > 0;
#define CA_L(H)                                                     \
  if (HD == H) {                                                    \
    if (sc) launch<H, true>(p, nseq, nkv, nsplit, st);              \
    else launch<H, false>(p, nseq, nkv, nsplit, st);                \
    return;                                                         \
  }
  CA_L(64) CA_L(128) CA_L(256) CA_L(512)
#undef CA_L
  TORCH_CHECK(false, "head_dim は 64/128/256/512 だけ");
}

}  // namespace cattn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &cattn::decode); }
