// TP2 の all-reduce を、2 プロセスで共有するピン留め・mapped のホストメモリ経由で行う（P2P の無い GeForce 向け）。
// CPU は番号を持たない: 呼び出しの番号は GPU のメモリ上のカウンタで持つので、CUDA グラフに捕獲して再生してよい
// （両ランクは同じ順で同じ大きさの all-reduce を呼ぶので、カウンタは両ランクで同じ値になる）。
//
// ホストメモリの並び（base からのバイト）:
//   A の面:  dataA(s, r, b) = ((s * 2 + r) * NBMAX + b) * BLK           s < NP, r < 2, b < NBMAX（NBMAX * BLK = max_bytes）
//   B の面:  dataB(r) = offB + r * max_bytes                              （面は 1 つ）
//   旗（int）: wA[2][NBMAX] rA[2][NBMAX] wB[2][MAXCH] rB[2]
//
// 方式 A（direct、1 カーネル、主に decode の小さいデータ向け）: ブロック b の受け持ちは常に [b*BLK, (b+1)*BLK) で、
//   その場所（面 s の dataA(s, ·, b)）はブロック b しか使わない → 面の使い回しはブロックごとの旗だけで守れる。
//   ブロック b、番号 n = ++cntA[b]、面 s = n % NP:
//     1. 相手が面 s の自分の場所を読み終えた（rA[相手][b] ≥ n - NP）のを待つ
//     2. 自分の入力（GPU）を dataA(s, 自分, b) に 16B ずつ書く → 各スレッド __threadfence_system → bar →
//        st.release.sys wA[自分][b] = n
//     3. ld.acquire.sys で wA[相手][b] ≥ n を待つ → bar → dataA(s, 相手, b) を ld.volatile（キャッシュしない）で読み、
//        自分 + 相手 を fp32 で足して fp16/bf16 に丸めて出力へ（2 項の和は可換なので両ランクで同じ値）
//     4. bar（読んだ値は出力に書いて使い終えている）→ fence → st.release.sys rA[自分][b] = n、cntA[b] = n
//   ブロックどうしは待ち合わせない（待つのは相手の GPU の同じ番号のブロックだけ）。どちらの GPU でも番号の小さいブロックから
//   起動されるので、両方で残っているうち最も小さい番号のブロックは必ず両方で動いていて、詰まらない。
//
// 方式 B（copy engine）: 主の stream で cntB += 1 → 送る stream: rB[相手] ≥ n-1 を待つカーネル → チャンクごとに
//   cudaMemcpyAsync D2H（dataB(自分)）→ wB[自分][c] = n を書くカーネル / 受ける stream: チャンクごとに wB[相手][c] ≥ n を
//   待つカーネル → cudaMemcpyAsync H2D → 最後に rB[自分] = n → 主の stream で 自分 + 相手。
//   コピーの番地は固定（面 1 つ）なので CUDA グラフに入る。旗の値はカーネルが GPU のカウンタから作る。
// 方式 C（ce_sm）: 送りは B と同じ（コピーエンジン）、受けはカーネルが wB[相手][c] を待って dataB(相手) を直接読んで足す
//   （H2D と足し算のカーネルが 1 つにまとまる）。**受けのカーネルは自分の送りが全部終わってから始める**（送りと同時に走らせると、
//   受けのカーネルが居座る間に同じ GPU の送り側の旗のカーネルが動けず、両ランクで詰まった。1 枚の GPU に 2 プロセスで再現）。
//   そのぶん自分の送りと受けは重ならない（相手の送りとは重なる）。グリッドは SM の数の 2 倍まで（旗を読む数を抑える）。
//   読み終えは、ブロックごとのチケット（GPU のメモリ）で最後のブロックが rB[自分] = n を立てる（B と同じ旗。呼び出しごとに
//   ブロックの数が違っても古い旗を待たない）。
//   方式 A は相手の同じ番号のブロックしか待たず、方式 B は送りの操作を全部先に積む（ハードウェアのキューが 1 本でも送りが先に済む）
//   ので、どちらも「同じ GPU の別のカーネルが同時に動くこと」に頼らない。
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace cear {

constexpr int MAXCH = 64;   // 方式 B・C のチャンクの最大数
constexpr int THREADS = 256;
constexpr int MAXPER = 8;   // 方式 A でスレッドが持つ 16B の数の上限（BLK ≤ THREADS * 16 * MAXPER = 32KB）

__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int4 ld_volatile(const int4* p) {
  int4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_volatile(int4* p, int4 v) {
  asm volatile("st.volatile.global.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
// 旗を待つ。休みなく読むと PCIe の読みの要求が溢れて相手・自分の DMA を押しのける（方式 C で 8MB が止まったように遅くなった）ので、
// 32ns から倍々に上限 SLEEP_MAX ns まで間を空けて読む
#ifndef CEAR_SLEEP_MAX
#define CEAR_SLEEP_MAX 1024
#endif
__device__ __forceinline__ void wait_geq(const int* p, int target) {
  unsigned ns = 32;
  while (ld_acquire_sys(p) < target) {
    __nanosleep(ns);
    if (ns < CEAR_SLEEP_MAX) ns <<= 1;
  }
}

// 16B（8 要素）どうしを fp32 で足して丸める（a + b == b + a なので両ランクで同じ）
template <typename T>
__device__ __forceinline__ int4 add8(int4 a, int4 b);
template <>
__device__ __forceinline__ int4 add8<half>(int4 a, int4 b) {
  int4 r;
  const half2* x = reinterpret_cast<const half2*>(&a);
  const half2* y = reinterpret_cast<const half2*>(&b);
  half2* z = reinterpret_cast<half2*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 fx = __half22float2(x[i]), fy = __half22float2(y[i]);
    z[i] = __floats2half2_rn(fx.x + fy.x, fx.y + fy.y);
  }
  return r;
}
template <>
__device__ __forceinline__ int4 add8<__nv_bfloat16>(int4 a, int4 b) {
  int4 r;
  const uint32_t* x = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* y = reinterpret_cast<const uint32_t*>(&b);
  uint32_t* z = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // bf16 → fp32 はビットの移動だけ。丸めは RN（sm_75 には bf16 の演算命令が無いので手で）
    const float lx = __uint_as_float(x[i] << 16), hx = __uint_as_float(x[i] & 0xffff0000u);
    const float ly = __uint_as_float(y[i] << 16), hy = __uint_as_float(y[i] & 0xffff0000u);
    const __nv_bfloat16 lo = __float2bfloat16_rn(lx + ly), hi = __float2bfloat16_rn(hx + hy);
    z[i] = (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
  }
  return r;
}
template <>
__device__ __forceinline__ int4 add8<float>(int4 a, int4 b) {  // fp32 は 4 要素
  return make_int4(__float_as_int(__int_as_float(a.x) + __int_as_float(b.x)),
                   __float_as_int(__int_as_float(a.y) + __int_as_float(b.y)),
                   __float_as_int(__int_as_float(a.z) + __int_as_float(b.z)),
                   __float_as_int(__int_as_float(a.w) + __int_as_float(b.w)));
}

struct Layout {
  char* hd;  // ホストメモリの GPU からの番地（cudaHostGetDevicePointer）
  int rank, np, blk, nbmax;
  int64_t max_bytes, offB, offF;
  __host__ __device__ char* dataA(int s, int r, int b) const {
    return hd + (((int64_t)s * 2 + r) * nbmax + b) * (int64_t)blk;
  }
  __host__ __device__ char* dataB(int r) const { return hd + offB + r * max_bytes; }
  __host__ __device__ int* wA(int r) const { return reinterpret_cast<int*>(hd + offF) + r * nbmax; }
  __host__ __device__ int* rA(int r) const { return wA(0) + 2 * nbmax + r * nbmax; }
  __host__ __device__ int* wB(int r) const { return wA(0) + 4 * nbmax + r * MAXCH; }
  __host__ __device__ int* rB(int r) const { return wA(0) + 4 * nbmax + 2 * MAXCH + r; }
};

// ---- 方式 A ----
template <typename T>
__global__ void __launch_bounds__(THREADS) direct_kernel(const int4* __restrict__ x, int4* __restrict__ out,
                                                         int* __restrict__ cnt, Layout L, int64_t nbytes) {
  __shared__ int sn;
  const int b = blockIdx.x, tid = threadIdx.x;
  const int me = L.rank, peer = 1 - L.rank;
  if (tid == 0) {
    const int n = cnt[b] + 1;
    sn = n;
    wait_geq(L.rA(peer) + b, n - L.np);  // 相手が面 s の自分の場所を読み終えた
  }
  __syncthreads();
  const int n = sn, s = n % L.np;
  const int64_t off = (int64_t)b * L.blk;
  const int nv = (int)(min((int64_t)L.blk, nbytes - off) / 16);
  int4* mine_h = reinterpret_cast<int4*>(L.dataA(s, me, b));
  const int4* peer_h = reinterpret_cast<const int4*>(L.dataA(s, peer, b));
  const int4* xb = x + off / 16;
  int4 v[MAXPER];
#pragma unroll
  for (int k = 0; k < MAXPER; ++k) {
    const int i = tid + k * THREADS;
    if (i < nv) {
      v[k] = xb[i];
      st_volatile(mine_h + i, v[k]);
    }
  }
  __threadfence_system();
  __syncthreads();
  if (tid == 0) {
    st_release_sys(L.wA(me) + b, n);
    wait_geq(L.wA(peer) + b, n);
  }
  __syncthreads();
  int4* ob = out + off / 16;
#pragma unroll
  for (int k = 0; k < MAXPER; ++k) {
    const int i = tid + k * THREADS;
    if (i < nv) ob[i] = add8<T>(v[k], ld_volatile(peer_h + i));
  }
  __syncthreads();  // 全員が相手の面を読み終えた（読んだ値は出力に書いた）
  if (tid == 0) {
    __threadfence_system();
    st_release_sys(L.rA(me) + b, n);
    cnt[b] = n;
  }
}

// ---- 方式 B・C の部品 ----
__global__ void bump_kernel(int* cnt) { cnt[0] += 1; }

// flags[i] ≥ cnt[0] + delta を待つ（i < k、スレッドごとに 1 つ）
__global__ void wait_kernel(const int* flags, int k, const int* cnt, int delta) {
  const int target = cnt[0] + delta;
  for (int i = threadIdx.x; i < k; i += blockDim.x) wait_geq(flags + i, target);
}

// *flag = cnt[0] + delta（前の D2H は stream の順で終わっている。PCIe の posted write は追い越さない）
__global__ void flag_kernel(int* flag, const int* cnt, int delta) {
  __threadfence_system();
  st_release_sys(flag, cnt[0] + delta);
}

template <typename T>
__global__ void add_kernel(const int4* __restrict__ x, const int4* __restrict__ y, int4* __restrict__ out, int64_t nv) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < nv; i += (int64_t)gridDim.x * blockDim.x)
    out[i] = add8<T>(x[i], y[i]);
}

// 方式 C の受け: 仮のブロック bb（BLK バイト）はチャンク c = bb*BLK / chunk の旗を待って、相手の dataB を直接読んで足す。
// グリッドは SM の数まで、ブロックは bb を順に回る（相手の旗はチャンクの順に立つので順に待ってよい）。cnt[0] = 番号、cnt[1] = チケット
template <typename T>
__global__ void __launch_bounds__(THREADS) cesm_recv_kernel(const int4* __restrict__ x, int4* __restrict__ out,
                                                            int* __restrict__ cnt, Layout L, int64_t nbytes,
                                                            int64_t chunk) {
  const int tid = threadIdx.x;
  const int me = L.rank, peer = 1 - L.rank;
  const int n = cnt[0];
  const int nblk = (int)((nbytes + L.blk - 1) / L.blk);
  for (int bb = blockIdx.x; bb < nblk; bb += gridDim.x) {
    const int64_t off = (int64_t)bb * L.blk;
    if (tid == 0) wait_geq(L.wB(peer) + (int)(off / chunk), n);
    __syncthreads();
    const int nv = (int)(min((int64_t)L.blk, nbytes - off) / 16);
    const int4* ph = reinterpret_cast<const int4*>(L.dataB(peer) + off);
    for (int i = tid; i < nv; i += THREADS) out[off / 16 + i] = add8<T>(x[off / 16 + i], ld_volatile(ph + i));
    __syncthreads();
  }
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(cnt + 1, 1) == (int)gridDim.x - 1) {  // 最後のブロック: 全部が読み終えた
      cnt[1] = 0;
      __threadfence_system();
      st_release_sys(L.rB(me), n);
    }
  }
}

}  // namespace cear

// 1 つのランクの状態（ホストメモリの番地・カウンタ・stream・event）
struct CeAr {
  cear::Layout L;
  char* hh;  // ホストの番地（cudaMemcpyAsync 用）
  at::Tensor cntA, cntB;
  c10::Device dev;
  c10::cuda::CUDAStream s_send, s_recv;
  cudaEvent_t ev_start, ev_send, ev_recv;
  int nsm = 1;

  // host_ptr: 2 プロセスで共有するホストメモリ（total_bytes 以上、0 で埋めたもの）。ここで cudaHostRegister（Portable | Mapped）する
  CeAr(int64_t host_ptr, int64_t rank, int64_t max_bytes, int64_t blk, int64_t np, int64_t device)
      : dev(c10::DeviceType::CUDA, (c10::DeviceIndex)device),
        s_send(c10::cuda::getStreamFromPool(true, (c10::DeviceIndex)device)),
        s_recv(c10::cuda::getStreamFromPool(true, (c10::DeviceIndex)device)) {
    TORCH_CHECK(rank == 0 || rank == 1);
    TORCH_CHECK(blk % 16 == 0 && blk <= cear::THREADS * 16 * cear::MAXPER && max_bytes % blk == 0 && np >= 2);
    const c10::cuda::CUDAGuard g(dev);
    hh = reinterpret_cast<char*>(host_ptr);
    C10_CUDA_CHECK(cudaHostRegister(hh, total_bytes(max_bytes, blk, np), cudaHostRegisterPortable | cudaHostRegisterMapped));
    void* dp = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&dp, hh, 0));
    L.hd = reinterpret_cast<char*>(dp);
    L.rank = (int)rank;
    L.np = (int)np;
    L.blk = (int)blk;
    L.nbmax = (int)(max_bytes / blk);
    L.max_bytes = max_bytes;
    L.offB = np * 2 * max_bytes;
    L.offF = L.offB + 2 * max_bytes;
    auto opt = at::TensorOptions().dtype(at::kInt).device(dev);
    cntA = at::zeros({L.nbmax}, opt);
    cntB = at::zeros({2}, opt);
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, (int)device));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev_start, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev_send, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev_recv, cudaEventDisableTiming));
  }

  // 共有メモリに要る大きさ（0 で埋めてから両ランクで登録する）
  static int64_t total_bytes(int64_t max_bytes, int64_t blk, int64_t np) {
    const int64_t nbmax = max_bytes / blk;
    return np * 2 * max_bytes + 2 * max_bytes + (4 * nbmax + 2 * cear::MAXCH + 2) * 4;
  }

  static void check_x(const at::Tensor& x, int64_t max_bytes) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat);
    const int64_t nb = x.numel() * x.element_size();
    TORCH_CHECK(nb % 16 == 0 && nb <= max_bytes, "大きさが 16 の倍数でないか max_bytes を超える");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  }

#define CEAR_TYPED(x, F)                                    \
  if ((x).scalar_type() == at::kHalf) {                     \
    F(half);                                                \
  } else if ((x).scalar_type() == at::kBFloat16) {          \
    F(__nv_bfloat16);                                       \
  } else {                                                  \
    F(float);                                               \
  }

  at::Tensor direct(const at::Tensor& x) {
    check_x(x, L.max_bytes);
    const c10::cuda::CUDAGuard g(x.device());
    at::Tensor out = at::empty_like(x);
    const int64_t nb = x.numel() * x.element_size();
    const int grid = (int)((nb + L.blk - 1) / L.blk);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
    const int4* xp = reinterpret_cast<const int4*>(x.data_ptr());
    int4* op = reinterpret_cast<int4*>(out.data_ptr());
#define CEAR_DIRECT(T) cear::direct_kernel<T><<<grid, cear::THREADS, 0, st>>>(xp, op, cntA.data_ptr<int>(), L, nb)
    CEAR_TYPED(x, CEAR_DIRECT)
#undef CEAR_DIRECT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  // 方式 B（sm_recv=false）・C（sm_recv=true）。chunk はチャンクの大きさ（BLK の倍数に切り上げ）
  at::Tensor ce(const at::Tensor& x, int64_t chunk, bool sm_recv) {
    check_x(x, L.max_bytes);
    const c10::cuda::CUDAGuard g(x.device());
    const int64_t nb = x.numel() * x.element_size();
    chunk = std::max<int64_t>(chunk, (nb + cear::MAXCH - 1) / cear::MAXCH);
    chunk = (chunk + L.blk - 1) / L.blk * L.blk;
    const int nch = (int)((nb + chunk - 1) / chunk);
    cudaStream_t main = c10::cuda::getCurrentCUDAStream().stream();
    cudaStream_t ss = s_send.stream(), sr = s_recv.stream();
    const int me = L.rank, peer = 1 - L.rank;
    int* cnt = cntB.data_ptr<int>();
    at::Tensor out = at::empty_like(x);
    at::Tensor recv;
    const int nblk = (int)((nb + L.blk - 1) / L.blk);

    cear::bump_kernel<<<1, 1, 0, main>>>(cnt);
    C10_CUDA_CHECK(cudaEventRecord(ev_start, main));
    // 送る: 相手が前回の自分の面を読み終えたのを待つ → チャンクごとに D2H → 旗
    C10_CUDA_CHECK(cudaStreamWaitEvent(ss, ev_start, 0));
    cear::wait_kernel<<<1, 32, 0, ss>>>(L.rB(peer), 1, cnt, -1);
    const char* xs = reinterpret_cast<const char*>(x.data_ptr());
    char* mine_hh = hh + L.offB + me * L.max_bytes;
    for (int c = 0; c < nch; ++c) {
      const int64_t o = c * chunk, sz = std::min(chunk, nb - o);
      C10_CUDA_CHECK(cudaMemcpyAsync(mine_hh + o, xs + o, sz, cudaMemcpyDeviceToHost, ss));
      cear::flag_kernel<<<1, 1, 0, ss>>>(L.wB(me) + c, cnt, 0);
    }
    C10_CUDA_CHECK(cudaEventRecord(ev_send, ss));
    if (sm_recv) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(main, ev_send, 0));  // 自分の送りが終わってから受ける（上の説明）
      // 受けて足す: 主の stream で（ブロックごとにチャンクの旗を待つ）
#define CEAR_CESM(T)                                                                                                \
  cear::cesm_recv_kernel<T><<<std::min(nblk, 2 * nsm), cear::THREADS, 0, main>>>(reinterpret_cast<const int4*>(x.data_ptr()),        \
                                                               reinterpret_cast<int4*>(out.data_ptr()), cnt, L, nb, \
                                                               chunk)
      CEAR_TYPED(x, CEAR_CESM)
#undef CEAR_CESM
    } else {
      recv = at::empty_like(x);  // 主の stream で確保（受ける stream は主の stream に合流するので後の再利用も安全）
      C10_CUDA_CHECK(cudaStreamWaitEvent(sr, ev_start, 0));
      char* rp = reinterpret_cast<char*>(recv.data_ptr());
      const char* peer_hh = hh + L.offB + peer * L.max_bytes;
      for (int c = 0; c < nch; ++c) {
        const int64_t o = c * chunk, sz = std::min(chunk, nb - o);
        cear::wait_kernel<<<1, 32, 0, sr>>>(L.wB(peer) + c, 1, cnt, 0);
        C10_CUDA_CHECK(cudaMemcpyAsync(rp + o, peer_hh + o, sz, cudaMemcpyHostToDevice, sr));
      }
      cear::flag_kernel<<<1, 1, 0, sr>>>(L.rB(me), cnt, 0);
      C10_CUDA_CHECK(cudaEventRecord(ev_recv, sr));
      C10_CUDA_CHECK(cudaStreamWaitEvent(main, ev_recv, 0));
      const int64_t nv = nb / 16;
      const int blocks = (int)std::min<int64_t>((nv + 255) / 256, 1024);
#define CEAR_ADD(T)                                                                                           \
  cear::add_kernel<T><<<blocks, 256, 0, main>>>(reinterpret_cast<const int4*>(x.data_ptr()),               \
                                                reinterpret_cast<const int4*>(recv.data_ptr()),            \
                                                reinterpret_cast<int4*>(out.data_ptr()), nv)
      CEAR_TYPED(x, CEAR_ADD)
#undef CEAR_ADD
    }
    C10_CUDA_CHECK(cudaStreamWaitEvent(main, ev_send, 0));  // x の D2H が終わるまで主の stream を進めない
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<CeAr>(m, "CeAr")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(), py::arg("host_ptr"), py::arg("rank"),
           py::arg("max_bytes"), py::arg("blk"), py::arg("np"), py::arg("device"))
      .def("direct", &CeAr::direct, "方式 A（1 カーネル、CUDA グラフ可）")
      .def("ce", &CeAr::ce, "方式 B（sm_recv=False）/ C（True）", py::arg("x"), py::arg("chunk"), py::arg("sm_recv") = false)
      .def_static("total_bytes", &CeAr::total_bytes);
}
