// Turing（sm_75）向けの W4A16 の行列積（group 32・対称・uint4b8・fp16 の活性・fp32 の累積）。
// C[M, N] = A[M, K] @ W[K, N]。W は vLLM の密な Marlin の並び（gptq_marlin_repack の出力 [K/16, 2N] の int32 と
// marlin_permute_scales の出力 [K/32, N] の fp16）をそのまま読む（読み込み時の詰め直しは要らない）。
//
// データの流れ（cuBLAS の turing_fp16_s1688gemm と同じレジスタ経由の二重バッファに、4 ビットを戻す処理を足した形）:
//   A: 大域メモリ → レジスタ（次の k の塊を先に読む）→ 共有メモリ（2 面、16B の塊を XOR で swizzle）→ ldmatrix.x4 → mma
//   B: 大域メモリ → レジスタ（次の k の塊を先に読む）→ lop3 で 0x6400 を重ねて fp16 に戻す → ×scale → mma（共有メモリを通らない）
//   1 周に __syncthreads は 1 回（共有メモリが 2 面なので「読みの後の書き」の同期が要らない）。
//
// Marlin の並び（16(k)×64(n) のタイルごとに int32 128 個 = 32 スレッド × int4）:
//   ニブルの番号 i（0..1023）のビット: i0=k8 i1=n8 i2=k1 | i3=n16 i4=n32 | i5=k2 i6=k4 | i7..9=n1 n2 n4
//   → スレッド lane（g = lane/4, t = lane%4）の int4 の j 番目の int32 は、列 16j + 8*n8 + g（n8 = 0, 1）の
//     k = 2t, 2t+1（k8 = 0）と 2t+8, 2t+9（k8 = 1）を持つ。これは mma.m16n8k8 の B の断片（b0 = B[2t..2t+1][g]）を
//     k8 = 0 / 1 の 2 回分、n8 の塊 2 つ分並べたものそのもの。列の向きは n8 の塊の単位で自然な順になる。
//   A の m16n8k16 ぶんの ldmatrix.x4 の 4 本は、k8 = 0 の {a0, a1} と k8 = 1 の {a0, a1} なので、A の並べ替えも要らない。
//   8 ビット（uint8b128、[K/16, 4N]、16×64 のタイルごとに int32 256 個、lane ごとに 8 個 = int4 2 つ）: lane の i 番目の int32 は
//     列 8i + g の k = 2t, 2t+8, 2t+1, 2t+9（バイト 0..3）。prmt 0x5250 / 0x5351 で k8 = 0 / 1 の b0 になる（probe8.py で実測）
//   scale（64 列ごとに scale_perm）: 列 16j + 8*n8 + g の scale は 64 列の中の位置 8g + 2j + n8 → スレッドは int4 1 つで 8 個読む。
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>


namespace cw4 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// 純粋なレジスタの計算なので volatile にしない（コンパイラが ldmatrix・LDG・戻す計算と自由に並べ替えられる）
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}

template <int LUT>
__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(LUT));
  return r;
}

__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<half2*>(&u); }

// int32 1 個（ニブル 8 個、uint4b8）→ B の断片 4 本（×scale 込み）。
// b[0] = 列の塊 n8=0 の k8=0、b[1] = n8=0 の k8=1、b[2] = n8=1 の k8=0、b[3] = n8=1 の k8=1。
// s = (n8=0 の scale, n8=1 の scale)
__device__ __forceinline__ void dequant(uint32_t q, uint32_t s, uint32_t (&b)[4]) {
  constexpr uint32_t LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
  constexpr int AND_OR = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
  const uint32_t t = q >> 8;
  half2 r0 = u2h(lop3<AND_OR>(q, LO, EX));  // 1024 + q（ニブル 0, 4 → k = 2t, 2t+1）
  half2 r1 = u2h(lop3<AND_OR>(q, HI, EX));  // 1024 + 16q（ニブル 1, 5 → k + 8）
  half2 r2 = u2h(lop3<AND_OR>(t, LO, EX));  // ニブル 2, 6（n8 = 1）
  half2 r3 = u2h(lop3<AND_OR>(t, HI, EX));  // ニブル 3, 7
  const half2 SUB = u2h(0x64086408u);       // 1032
  const half2 MUL = u2h(0x2c002c00u);       // 1/16
  const half2 ADD = u2h(0xd480d480u);       // -72
  r0 = __hsub2(r0, SUB);
  r1 = __hfma2(r1, MUL, ADD);
  r2 = __hsub2(r2, SUB);
  r3 = __hfma2(r3, MUL, ADD);
  const half2 sv = u2h(s);
  const half2 s0 = __low2half2(sv), s1 = __high2half2(sv);
  b[0] = h2u(__hmul2(r0, s0));
  b[1] = h2u(__hmul2(r1, s0));
  b[2] = h2u(__hmul2(r2, s1));
  b[3] = h2u(__hmul2(r3, s1));
}

// int32 1 個（バイト 4 個、uint8b128）→ 列の塊 1 つの B の断片 2 本（×scale 込み）。
// 8 ビットの Marlin の並び: バイト 0, 2 = k 2t, 2t+1（k8 = 0）、バイト 1, 3 = 2t+8, 2t+9（k8 = 1）。Marlin と同じ prmt で 0x64 を重ねる
__device__ __forceinline__ void dequant8(uint32_t q, half s, uint32_t (&b)[2]) {
  uint32_t lo, hi;
  const uint32_t EX = 0x64646464u;
  asm("prmt.b32 %0, %1, %2, 0x5250;\n" : "=r"(lo) : "r"(q), "r"(EX));
  asm("prmt.b32 %0, %1, %2, 0x5351;\n" : "=r"(hi) : "r"(q), "r"(EX));
  const half2 SUB = u2h(0x64806480u);  // 1152 = 1024 + 128
  const half2 s2 = __half2half2(s);
  b[0] = h2u(__hmul2(__hsub2(u2h(lo), SUB), s2));
  b[1] = h2u(__hmul2(__hsub2(u2h(hi), SUB), s2));
}

template <int N>
struct VecLoad;
template <>
struct VecLoad<4> {
  __device__ __forceinline__ static void ld(uint32_t* d, const void* p) {
    int4 v = __ldg(reinterpret_cast<const int4*>(p));
    d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
  }
};
template <>
struct VecLoad<2> {
  __device__ __forceinline__ static void ld(uint32_t* d, const void* p) {
    int2 v = __ldg(reinterpret_cast<const int2*>(p));
    d[0] = v.x; d[1] = v.y;
  }
};

// BITS（4 = uint4b8 / 8 = uint8b128）、BM×BN×BK のタイル、WM×WN 個のワープ（ワープのタイル TM×TN）。BK は 32 か 64、TN は 32 の倍数。
// SPLIT: gridDim.y 個に K を分ける（小さい M で SM を埋める用）。カーネルは 1 つで、0 埋めも fp16 への変換も別に要らない:
//   各切れは fp32 の部分和を自分の面 ws[切れ][M][N] に書き、タイルごとのチケット sems[blockIdx.x] を 1 つ進める。
//   最後に着いた CTA が全部の切れを切れの番号の順に足して（自分の分はレジスタから）fp16 で C に書き、チケットを 0 に戻す
//   （次の呼び出しのため、常に 0 で終わる）。誰も他を待たない・足す順が決まっていて決定的。
//   （CUTLASS の serial split-K のように切れの順に待つ形も試したが、書き出しが直列に並んで 0 埋め＋変換と同じだけかかった）
template <int BITS, int BM, int BN, int BK, int WM, int WN, int MINB, bool SPLIT>
__global__ void __launch_bounds__(WM* WN * 32, MINB)
    w4a16_kernel(const half* __restrict__ A, const uint32_t* __restrict__ W, const half* __restrict__ S,
                 half* __restrict__ C, float* __restrict__ ws, int* __restrict__ sems, int M, int N, int K, int lda,
                 int ldc, int group_m, int k_per) {
  constexpr int T = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int MT = TM / 16;   // ワープの m16 の塊
  constexpr int NB = TN / 8;    // ワープの n8 の塊
  constexpr int NI = TN * BITS / 64;  // k16 ごとにスレッドが持つ int32（B）
  constexpr int PL = BITS == 4 ? 4 : 8;      // 64 列のタイルで 1 スレッドが持つ int32（k16 ごと）
  constexpr int TS = BITS == 4 ? 128 : 256;  // 64 列のタイルの int32（k16 ごと）
  constexpr int NS = NB / 2;                 // scale の half2（n8 の塊 2 つずつ）
  constexpr int KS = BK / 16;   // k16 の切れ
  constexpr int KG = BK / 32;   // group（32）の数
  constexpr int CPR = BK / 8;   // A の 1 行の 16B の塊
  constexpr int ACH = BM * CPR;  // A の 1 面の 16B の塊
  constexpr int AL = (ACH + T - 1) / T;
  constexpr int VB = NI >= 4 ? 4 : NI;  // B の 1 回の読み（int32 の数）
  constexpr int VS = NS >= 4 ? 4 : NS;  // scale の 1 回の読み（half2 の数）
  static_assert(BITS == 4 || BITS == 8, "bits");
  static_assert(TM % 16 == 0 && TN % 32 == 0 && (BK == 32 || BK == 64), "tile");
  static_assert(BM % WM == 0 && BN % WN == 0, "warps");

  __shared__ __align__(128) half smA[2][BM * BK];

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;

  // タイルの割り当て（GROUP_M 行のタイルごとに n を回す。L2 での B の使い回し）
  const int num_m = (M + BM - 1) / BM, num_n = N / BN;
  const int pid = blockIdx.x;
  const int gsz = group_m * num_n;
  const int first_m = (pid / gsz) * group_m;
  const int size_m = min(num_m - first_m, group_m);
  const int pid_m = first_m + (pid % gsz) % size_m;
  const int pid_n = (pid % gsz) / size_m;
  const int m0 = pid_m * BM, n0 = pid_n * BN;

  const int KT = K / BK;
  int kt0 = 0, kt1 = KT;
  if (SPLIT) {
    kt0 = blockIdx.y * k_per;  // 切れの数 gridDim.y は空の切れが出ないように呼ぶ側で決める
    kt1 = min(KT, kt0 + k_per);
  }

  // A の大域メモリの読み（スレッドごとに AL 個の 16B の塊）。M の外の行は M-1 行目を読む（書かない）
  const half* a_src[AL];
  uint32_t a_dst[AL];
  bool a_ok[AL];
#pragma unroll
  for (int i = 0; i < AL; ++i) {
    const int c = tid + i * T;
    a_ok[i] = (ACH % T == 0) || c < ACH;
    const int r = c / CPR, ch = c % CPR;
    const int gr = min(m0 + r, M - 1);
    a_src[i] = A + (size_t)gr * lda + ch * 8;
    const int swz = CPR == 8 ? (r & 7) : ((r >> 1) & 3);
    a_dst[i] = (r * BK + ((ch ^ swz) * 8)) * 2;  // バイト（面の先頭から）
  }
  const uint32_t smA0 = smem_u32(&smA[0][0]);
  constexpr uint32_t PLANE = BM * BK * 2;

  // ldmatrix の行の番地（面の先頭から、ks ごとに塊が 2 つ進む）
  uint32_t a_frag_off[MT][KS];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int r = wm * TM + mt * 16 + (lane & 15);
    const int swz = CPR == 8 ? (r & 7) : ((r >> 1) & 3);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int ch = ks * 2 + (lane >> 4);
      a_frag_off[mt][ks] = (r * BK + ((ch ^ swz) * 8)) * 2;
    }
  }

  // B: ワープの最初の列 nw（32 の倍数）。int32 の番地 = 行（k16）* ldw + タイル*TS + lane*PL + j
  //   4 ビット: j 番目 = 列 16j + 8*n8 + g（n8 の塊 2 つ）、8 ビット: j 番目 = 列 8j + g（n8 の塊 1 つ）
  const int nw = n0 + wn * TN;
  const size_t ldw = (size_t)(BITS == 4 ? 2 : 4) * N;
  const uint32_t* w_base = W + (size_t)(nw / 64) * TS + lane * PL + (TN >= 64 ? 0 : ((nw % 64) / 32) * (PL / 2));
  // scale: 行（group）* N + タイル*64 + 8g + jj
  const int g = lane >> 2, t4 = lane & 3;
  const half* s_base = S + (size_t)(nw / 64) * 64 + 8 * g + (TN >= 64 ? 0 : 2 * ((nw % 64) / 16));

  uint32_t a_reg[AL][4];
  uint32_t b_raw[2][KS][NI];
  uint32_t s_raw[2][KG][NS];  // NS 個の half2（n8 の塊 2q, 2q+1 の scale）

  auto load_a = [&](int kt) {
#pragma unroll
    for (int i = 0; i < AL; ++i) {
      if (a_ok[i]) {
        int4 v = *reinterpret_cast<const int4*>(a_src[i] + (size_t)kt * BK);
        a_reg[i][0] = v.x; a_reg[i][1] = v.y; a_reg[i][2] = v.z; a_reg[i][3] = v.w;
      }
    }
  };
  auto store_a = [&](int plane) {
#pragma unroll
    for (int i = 0; i < AL; ++i) {
      if (a_ok[i]) {
        const uint32_t addr = smA0 + plane * PLANE + a_dst[i];
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(a_reg[i][0]), "r"(a_reg[i][1]),
                     "r"(a_reg[i][2]), "r"(a_reg[i][3]));
      }
    }
  };
  auto load_b = [&](int kt, int buf) {
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const uint32_t* p = w_base + (size_t)(kt * KS + ks) * ldw;
#pragma unroll
      for (int v = 0; v < NI / VB; ++v) VecLoad<VB>::ld(&b_raw[buf][ks][v * VB], p + (v * VB / PL) * TS + (v * VB % PL));
    }
#pragma unroll
    for (int kg = 0; kg < KG; ++kg) {
      const half* p = s_base + (size_t)(kt * KG + kg) * N;
#pragma unroll
      for (int v = 0; v < NS / VS; ++v) VecLoad<VS>::ld(&s_raw[buf][kg][v * VS], p + v * 64);
    }
  };

  float acc[MT][NB][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int b = 0; b < NB; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][b][e] = 0.f;

  auto compute = [&](int plane, int buf) {
    uint32_t af[2][MT][4];
    const uint32_t base = smA0 + plane * PLANE;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) ldsm_x4(af[0][mt], base + a_frag_off[mt][0]);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      if (ks + 1 < KS) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) ldsm_x4(af[(ks + 1) & 1][mt], base + a_frag_off[mt][ks + 1]);
      }
      if constexpr (BITS == 4) {
#pragma unroll
        for (int q = 0; q < NI; ++q) {
          uint32_t bf[4];
          dequant(b_raw[buf][ks][q], s_raw[buf][ks / 2][q], bf);
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma1688(acc[mt][2 * q], af[ks & 1][mt][0], af[ks & 1][mt][1], bf[0]);
            mma1688(acc[mt][2 * q + 1], af[ks & 1][mt][0], af[ks & 1][mt][1], bf[2]);
            mma1688(acc[mt][2 * q], af[ks & 1][mt][2], af[ks & 1][mt][3], bf[1]);
            mma1688(acc[mt][2 * q + 1], af[ks & 1][mt][2], af[ks & 1][mt][3], bf[3]);
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < NI; ++i) {
          uint32_t bf[2];
          const half2 sv = u2h(s_raw[buf][ks / 2][i / 2]);
          dequant8(b_raw[buf][ks][i], (i & 1) ? __high2half(sv) : __low2half(sv), bf);
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma1688(acc[mt][i], af[ks & 1][mt][0], af[ks & 1][mt][1], bf[0]);
            mma1688(acc[mt][i], af[ks & 1][mt][2], af[ks & 1][mt][3], bf[1]);
          }
        }
      }
    }
  };

  // 前置き: 最初の塊を読んで共有メモリへ
  load_a(kt0);
  load_b(kt0, 0);
  store_a(0);
  __syncthreads();

  // 2 周ずつ展開して、B のレジスタの面（buf）を定数にする（面の入れ替えの mov を出さない）
  int kt = kt0;
  int plane = 0;
  for (; kt + 1 < kt1; kt += 2) {
    // 周 kt（面 plane、buf 0）: 周 kt+1 を先に読む
    load_a(kt + 1);
    load_b(kt + 1, 1);
    compute(plane, 0);
    store_a(plane ^ 1);
    __syncthreads();
    plane ^= 1;
    // 周 kt+1（buf 1）: 周 kt+2 を先に読む（無ければ読まない）
    const bool more = kt + 2 < kt1;
    if (more) {
      load_a(kt + 2);
      load_b(kt + 2, 0);
    }
    compute(plane, 1);
    if (more) store_a(plane ^ 1);
    __syncthreads();
    plane ^= 1;
  }
  if (kt < kt1) compute(plane, 0);  // 残りが奇数の周

  // 書き出し: acc[mt][b] = 行 m0 + wm*TM + mt*16 + g (+8)、列 nw + 8b + 2t (+1)
  const int sk = blockIdx.y, nsk = gridDim.y;
  const size_t plane_sz = (size_t)M * N;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int r0 = m0 + wm * TM + mt * 16 + g;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const int col = nw + 8 * b + 2 * t4;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = r0 + 8 * h;
        if (r < M) {
          if (SPLIT) {
            __stcg(reinterpret_cast<float2*>(ws + sk * plane_sz + (size_t)r * N + col),
                   make_float2(acc[mt][b][2 * h], acc[mt][b][2 * h + 1]));
          } else {
            *reinterpret_cast<half2*>(C + (size_t)r * ldc + col) =
                __floats2half2_rn(acc[mt][b][2 * h], acc[mt][b][2 * h + 1]);
          }
        }
      }
    }
  }
  if (SPLIT) {
    __shared__ int last;
    __threadfence();  // 部分和を書き終えてからチケットを進める
    __syncthreads();
    if (tid == 0) {
      const int ticket = atomicAdd(sems + blockIdx.x, 1);
      last = ticket == nsk - 1;
      if (last) sems[blockIdx.x] = 0;  // 次の呼び出しのために 0 に戻す（このタイルの CTA は全部着いた）
    }
    __syncthreads();
    if (last) {
      __threadfence();
      // 切れのループを外側に、要素の読みを内側に並べる（読みが並んで出て待ちが重なる）。足す順は切れの番号の順（決定的）
      float2 sum[MT][NB][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
          for (int h = 0; h < 2; ++h) sum[mt][b][h] = make_float2(0.f, 0.f);
      for (int k = 0; k < nsk; ++k) {
        const float* pk = ws + k * plane_sz;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int r0 = m0 + wm * TM + mt * 16 + g;
#pragma unroll
          for (int b = 0; b < NB; ++b) {
            const int col = nw + 8 * b + 2 * t4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int r = r0 + 8 * h;
              float2 o = make_float2(acc[mt][b][2 * h], acc[mt][b][2 * h + 1]);
              if (k != sk && r < M) o = __ldcg(reinterpret_cast<const float2*>(pk + (size_t)r * N + col));
              sum[mt][b][h].x += o.x;
              sum[mt][b][h].y += o.y;
            }
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r0 = m0 + wm * TM + mt * 16 + g;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          const int col = nw + 8 * b + 2 * t4;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int r = r0 + 8 * h;
            if (r < M)
              *reinterpret_cast<half2*>(C + (size_t)r * ldc + col) = __floats2half2_rn(sum[mt][b][h].x, sum[mt][b][h].y);
          }
        }
      }
    }
  }
}

// 4 ビットの設定の一覧（BM, BN, BK, WM, WN, MINB）。増やすときはここに足す。split は BM ≤ 64 だけ
#define CW4_CONFIGS(X)    \
  X(128, 128, 32, 2, 4, 1) \
  X(128, 128, 32, 4, 2, 1) \
  X(128, 128, 64, 2, 4, 1) \
  X(128, 128, 64, 4, 2, 1) \
  X(128, 256, 32, 2, 4, 1) \
  X(128, 256, 64, 2, 4, 1) \
  X(256, 128, 32, 4, 2, 1) \
  X(64, 128, 32, 2, 2, 2)  \
  X(64, 128, 64, 2, 2, 2)  \
  X(64, 128, 32, 2, 4, 1)  \
  X(64, 128, 64, 2, 4, 1)  \
  X(64, 256, 32, 2, 4, 1)  \
  X(64, 256, 64, 2, 4, 1)  \
  X(128, 64, 32, 4, 1, 2)  \
  X(128, 64, 64, 4, 2, 2)  \
  X(32, 128, 64, 1, 4, 2)  \
  X(32, 256, 64, 1, 4, 2)  \
  X(32, 256, 64, 2, 4, 1)  \
  X(16, 128, 64, 1, 4, 2)  \
  X(16, 256, 64, 1, 4, 2)  \
  X(16, 128, 64, 1, 2, 4)  \
  X(128, 256, 32, 1, 8, 1) \
  X(128, 256, 64, 1, 8, 1) \
  X(128, 128, 32, 1, 4, 1) \
  X(128, 128, 32, 1, 4, 2) \
  X(256, 128, 32, 2, 4, 1) \
  X(64, 256, 32, 1, 8, 1)  \
  X(64, 256, 64, 1, 8, 1)  \
  X(64, 128, 64, 1, 4, 2)  \
  X(64, 256, 64, 1, 4, 1)  \
  X(64, 256, 32, 1, 4, 1)  \
  X(64, 128, 32, 1, 2, 3)  \
  X(128, 128, 32, 2, 2, 1) \
  X(64, 128, 64, 1, 4, 3)  \
  X(32, 128, 64, 1, 4, 4)

// 8 ビット（lm_head 用。M は小さく、帯域で決まる）
#define CW8_CONFIGS(X)   \
  X(16, 256, 64, 1, 4, 2) \
  X(16, 128, 64, 1, 4, 2) \
  X(16, 128, 64, 1, 2, 4) \
  X(16, 64, 64, 1, 2, 4)  \
  X(16, 256, 32, 1, 4, 2) \
  X(16, 128, 32, 1, 4, 4) \
  X(32, 256, 64, 1, 4, 2) \
  X(32, 128, 64, 1, 4, 2) \
  X(32, 128, 64, 1, 4, 4) \
  X(64, 128, 64, 1, 4, 2) \
  X(64, 256, 32, 1, 4, 1) \
  X(64, 128, 32, 1, 4, 3) \
  X(128, 256, 32, 1, 8, 1) \
  X(128, 256, 32, 2, 4, 1)

// split-K のチケット（タイルごと、端末ごとに 1 つを使い回す。最後に着いた CTA が 0 に戻すので常に 0 で終わる。
// 同じ端末で別々のストリームから同時に split-K を流すと取り合うので、ストリーム 1 本の前提）
constexpr int64_t SEM_CAP = 1 << 18;
inline int* sems_for(const at::Tensor& like) {
  static at::Tensor sems[64];
  const int d = like.get_device();
  if (!sems[d].defined()) sems[d] = at::zeros({SEM_CAP}, like.options().dtype(at::kInt));
  return sems[d].data_ptr<int>();
}

template <int BITS, int BM, int BN, int BK, int WM, int WN, int MINB>
void launch(const at::Tensor& a, const at::Tensor& w, const at::Tensor& s, at::Tensor& c, int split, int group_m,
            cudaStream_t st) {
  const int M = a.size(0), K = a.size(1), N = c.size(1);
  const int num_m = (M + BM - 1) / BM, num_n = N / BN;
  const int KT = K / BK;
  const int k_per = (KT + split - 1) / split;
  const int nsk = (KT + k_per - 1) / k_per;  // 空の切れを作らない
  dim3 block(WM * WN * 32);
  const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
  const uint32_t* W = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
  const half* S = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
  half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());
  if constexpr (BM <= 64) {
    if (nsk > 1) {
      TORCH_CHECK((int64_t)num_m * num_n <= SEM_CAP, "split-K のタイルが多すぎる");
      at::Tensor ws = at::empty({(int64_t)nsk * M * N}, a.options().dtype(at::kFloat));  // 0 埋めは要らない（各切れが書く）
      w4a16_kernel<BITS, BM, BN, BK, WM, WN, MINB, true><<<dim3(num_m * num_n, nsk), block, 0, st>>>(
          A, W, S, C, ws.data_ptr<float>(), sems_for(a), M, N, K, a.stride(0), c.stride(0), group_m, k_per);
      return;
    }
  } else {
    TORCH_CHECK(nsk == 1, "split-K は BM ≤ 64 の設定だけ");
  }
  w4a16_kernel<BITS, BM, BN, BK, WM, WN, MINB, false><<<dim3(num_m * num_n, 1), block, 0, st>>>(
      A, W, S, C, nullptr, nullptr, M, N, K, a.stride(0), c.stride(0), group_m, KT);
}

}  // namespace cw4

// a: [M, K] fp16（行が連続、stride(0) は 8 の倍数）、w: Marlin の並び [K/16, 2N] int32、s: Marlin の並び [K/32, N] fp16
// out: [M, N] fp16（列が連続）。cfg = (BM, BN, BK, WM, WN, MINB)
at::Tensor w4a16_gemm(const at::Tensor& a, const at::Tensor& w, const at::Tensor& s, c10::optional<at::Tensor> out,
                      int64_t BM, int64_t BN, int64_t BK, int64_t WM, int64_t WN, int64_t MINB, int64_t split,
                      int64_t group_m) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kHalf && a.dim() == 2 && a.stride(1) == 1 && a.stride(0) % 8 == 0);
  TORCH_CHECK(w.scalar_type() == at::kInt && w.is_contiguous() && s.scalar_type() == at::kHalf && s.is_contiguous());
  const int64_t M = a.size(0), K = a.size(1), N = s.size(1);
  TORCH_CHECK(w.size(0) * 16 == K && (w.size(1) == 2 * N || w.size(1) == 4 * N), "w の形が Marlin の並びでない");
  const int bits = w.size(1) == 2 * N ? 4 : 8;  // 4 ビット [K/16, 2N] / 8 ビット [K/16, 4N]
  TORCH_CHECK(s.size(0) * 32 == K, "group 32 のみ");
  TORCH_CHECK(K % BK == 0 && N % BN == 0, "K / N がタイルで割り切れない");
  TORCH_CHECK(M > 0);
  const c10::cuda::CUDAGuard guard(a.device());
  at::Tensor c = out.has_value() ? *out : at::empty({M, N}, a.options());
  TORCH_CHECK(c.stride(1) == 1 && c.size(0) == M && c.size(1) == N);
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
#define CW_DISPATCH(B, bm, bn, bk, wm, wn, minb)                                                       \
  if (bits == B && BM == bm && BN == bn && BK == bk && WM == wm && WN == wn && MINB == minb) {        \
    cw4::launch<B, bm, bn, bk, wm, wn, minb>(a, w, s, c, (int)split, (int)group_m, st);                \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                    \
    return c;                                                                                          \
  }
#define CW4_DISPATCH(...) CW_DISPATCH(4, __VA_ARGS__)
#define CW8_DISPATCH(...) CW_DISPATCH(8, __VA_ARGS__)
  CW4_CONFIGS(CW4_DISPATCH)
  CW8_CONFIGS(CW8_DISPATCH)
#undef CW4_DISPATCH
#undef CW8_DISPATCH
#undef CW_DISPATCH
  TORCH_CHECK(false, "その設定は作っていない: ", bits, "bit ", BM, "x", BN, "x", BK, " ", WM, "x", WN, " minb ", MINB);
}

std::vector<std::vector<int64_t>> configs(int64_t bits) {
  std::vector<std::vector<int64_t>> r;
#define CW_LIST(bm, bn, bk, wm, wn, minb) r.push_back({bm, bn, bk, wm, wn, minb});
  if (bits == 4) {
    CW4_CONFIGS(CW_LIST)
  } else {
    CW8_CONFIGS(CW_LIST)
  }
#undef CW_LIST
  return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &w4a16_gemm, "W4A16 GEMM（Marlin の並び、sm_75）", py::arg("a"), py::arg("w"), py::arg("s"),
        py::arg("out") = py::none(), py::arg("BM"), py::arg("BN"), py::arg("BK"), py::arg("WM"), py::arg("WN"),
        py::arg("MINB") = 1, py::arg("split") = 1, py::arg("group_m") = 8);
  m.def("configs", &configs, py::arg("bits") = 4);
}
