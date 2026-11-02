"""TP2 の all-reduce を、2 プロセスで共有するピン留め・mapped のホストメモリ経由で行う（P2P の無い GeForce 向け、CUDA C++ の拡張 ce_ar.cu）。

前の版（Python＋ctypes でドライバを 1 回に約 25 回呼ぶ）との違い:
- 呼び出しの番号を GPU のメモリ上のカウンタで持つので CPU は値を渡さない → CUDA グラフに捕獲して再生してよい（decode の小さい all-reduce にも効く）
- 1 回の all-reduce は C++ からの 1 回の呼び出しで全部出す（方式 A は カーネル 1 つ）
方式（ce_ar.cu の先頭に詳細）:
  direct（A）: 各ランクのカーネルが自分の面に直接書き、ブロックごとの旗で相手を待って、相手の面を直接読んで足す。カーネル 1 つ
  ce（B）    : コピーエンジンでチャンクごとに D2H / H2D、旗の書き込み・待ちは小さいカーネル、最後に足し算のカーネル
  cesm（C）  : 送りはコピーエンジン、受けはカーネルが相手の面を直接読んで足す

vLLM への差し込み: vllm/distributed/device_communicators/ にこのファイルを置き、cuda_communicator.py
（同じディレクトリの patched 版）の CudaCommunicator.__init__ で maybe_create、all_reduce の頭で maybe_all_reduce を呼ぶ。
環境変数:
  VLLM_CE_AR=1                 有効にする（既定は無効）
  VLLM_CE_AR_DIR=<dir>         ce_ar.cu と拡張（build/ce_ar_ext*.so）のある場所（このリポジトリでは benchmarks/turing/ce_allreduce）。無ければ ce_ar.cu から作る（nvcc が要る）
  VLLM_CE_AR_MIN_KB=0          これより小さい all-reduce は NCCL のまま
  VLLM_CE_AR_MAX_MB=16         これより大きい all-reduce は NCCL のまま（ホストメモリは (NP*2+2) × これ だけ使う）
  VLLM_CE_AR_SMALL_MAX_KB=1024 これ以下は direct（A）、超えると VLLM_CE_AR_LARGE の方式
  VLLM_CE_AR_LARGE=ce          大きいデータの方式: direct / ce / cesm
  VLLM_CE_AR_LARGE_MIN_KB=0    SMALL_MAX_KB を超えてこれより小さいものは NCCL のまま（RTX 2080 × 2 では 1〜4MB は NCCL が速い）
  VLLM_CE_AR_CHUNK_KB=1024     ce・cesm のチャンクの大きさ（チャンクは最大 64 個）
  VLLM_CE_AR_BLK_KB=8          direct・cesm の 1 ブロックの受け持ち（固定。2〜32）
  VLLM_CE_AR_NP=2              direct の面の数
  VLLM_CE_AR_GRAPH=1           0 にすると CUDA グラフの捕獲中は NCCL に戻す
"""
import ctypes
import glob
import importlib.util
import os
from multiprocessing import shared_memory

import torch

_EXT = None
_NAME = "ce_ar_ext"


def ext(src_dir=None):
    """拡張を読む（build/ にあればそれ、無ければ作る。sm_75 の SASS と compute_75 の PTX）"""
    global _EXT
    if _EXT is not None:
        return _EXT
    d = src_dir or os.environ.get("VLLM_CE_AR_DIR")
    if d is None:  # vLLM の中に mount したとき（このファイルだけが device_communicators にある）は /ce-ar を見る
        here = os.path.dirname(os.path.abspath(__file__))
        d = here if os.path.exists(os.path.join(here, "ce_ar.cu")) or not os.path.isdir("/ce-ar") else "/ce-ar"
    bd = os.environ.get("CE_AR_BUILD", os.path.join(d, "build"))
    so = glob.glob(os.path.join(bd, _NAME + "*.so"))
    src = os.path.join(d, "ce_ar.cu")
    stale = so and os.path.exists(src) and os.access(bd, os.W_OK) and os.path.getmtime(so[0]) < os.path.getmtime(src)
    if so and not stale and os.environ.get("CE_AR_REBUILD") != "1":
        spec = importlib.util.spec_from_file_location(_NAME, so[0])
        _EXT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXT)
        return _EXT
    from torch.utils.cpp_extension import load
    os.makedirs(bd, exist_ok=True)
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("CE_AR_ARCH", "7.5+PTX")
    try:
        _EXT = load(name=_NAME, sources=[src], build_directory=bd, extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-lineinfo"], verbose=os.environ.get("CE_AR_VERBOSE") == "1")
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _EXT


def _env(k, d):
    return os.environ.get(k, d)


class CEAllReduce:
    def __init__(self, rank: int, shm_name: str, create: bool, device: torch.device, max_bytes: int, blk: int,
                 np_: int, min_bytes: int, small_max: int, large: str, chunk: int, graph: bool, large_min: int = 0):
        assert large in ("direct", "ce", "cesm")
        self.rank, self.device = rank, device
        self.max_bytes, self.blk, self.np = max_bytes, blk, np_
        self.min_bytes, self.small_max, self.large, self.chunk, self.graph = min_bytes, small_max, large, chunk, graph
        self.large_min = large_min
        e = ext()
        self.total = e.CeAr.total_bytes(max_bytes, blk, np_)
        if create:
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.total)
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf)), 0, self.total)
        else:
            self.shm = shared_memory.SharedMemory(name=shm_name)
        from multiprocessing import resource_tracker
        resource_tracker.unregister(self.shm._name, "shared_memory")  # 名前は作った側が明示的に消す
        self.host = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        self.ar = e.CeAr(self.host, rank, max_bytes, blk, np_, device.index if device.index is not None else 0)
        self.calls = {"direct": 0, "ce": 0, "cesm": 0, "skip": 0}  # 確認用（グラフの再生は数えない）

    def flags(self):
        """旗の今の値（確かめる用。ホストメモリを CPU から読む）"""
        nbmax, maxch = self.max_bytes // self.blk, 64
        off = self.np * 2 * self.max_bytes + 2 * self.max_bytes
        n = 4 * nbmax + 2 * maxch + 2
        a = (ctypes.c_int * n).from_address(self.host + off)
        v = list(a)
        return {"wA": [v[0:nbmax], v[nbmax:2 * nbmax]], "rA": [v[2 * nbmax:3 * nbmax], v[3 * nbmax:4 * nbmax]],
                "wB": [v[4 * nbmax:4 * nbmax + maxch], v[4 * nbmax + maxch:4 * nbmax + 2 * maxch]],
                "rB": v[4 * nbmax + 2 * maxch:]}

    def maybe_all_reduce(self, x: torch.Tensor):
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not x.is_contiguous() or not x.is_cuda:
            return None
        nb = x.numel() * x.element_size()
        if nb % 16 or nb < self.min_bytes or nb > self.max_bytes or nb == 0 or x.data_ptr() % 16:
            return None
        if not self.graph and torch.cuda.is_current_stream_capturing():
            return None
        if self.small_max < nb < self.large_min:
            return None  # 中くらいは NCCL（大きさだけで決まるので両ランクで同じ）
        mode = "direct" if nb <= self.small_max else self.large
        self.calls[mode] += 1
        if mode == "direct":
            return self.ar.direct(x)
        return self.ar.ce(x, self.chunk, mode == "cesm")


_SEQ = [0]


def maybe_create(cpu_group, rank: int, world_size: int, device: torch.device):
    if _env("VLLM_CE_AR", "0") != "1" or world_size != 2:
        return None
    import torch.distributed as dist
    ranks = dist.get_process_group_ranks(cpu_group)
    _SEQ[0] += 1  # 同じプロセスで複数のグループ（TP・PP など）を作っても名前がぶつからないように
    name = [f"vllm_ce_ar_{os.getpid()}_{_SEQ[0]}" if rank == 0 else None]
    dist.broadcast_object_list(name, src=ranks[0], group=cpu_group)
    kw = dict(device=device, max_bytes=int(float(_env("VLLM_CE_AR_MAX_MB", "16")) * (1 << 20)),
              blk=int(float(_env("VLLM_CE_AR_BLK_KB", "8")) * 1024), np_=int(_env("VLLM_CE_AR_NP", "2")),
              min_bytes=int(float(_env("VLLM_CE_AR_MIN_KB", "0")) * 1024),
              small_max=int(float(_env("VLLM_CE_AR_SMALL_MAX_KB", "1024")) * 1024), large=_env("VLLM_CE_AR_LARGE", "ce"),
              chunk=int(float(_env("VLLM_CE_AR_CHUNK_KB", "1024")) * 1024), graph=_env("VLLM_CE_AR_GRAPH", "1") == "1",
              large_min=int(float(_env("VLLM_CE_AR_LARGE_MIN_KB", "0")) * 1024))
    kw["max_bytes"] = kw["max_bytes"] // kw["blk"] * kw["blk"]
    if rank == 0:
        ce = CEAllReduce(0, name[0], True, **kw)
    dist.barrier(group=cpu_group)
    if rank == 1:
        ce = CEAllReduce(1, name[0], False, **kw)
    dist.barrier(group=cpu_group)  # 両方が登録を終えた（0 で埋めた後）
    if rank == 0:
        import _posixshmem
        _posixshmem.shm_unlink(ce.shm._name)  # 両方が開いた後なので名前だけ消す（領域はプロセスが終わるまで残る）
    return ce
