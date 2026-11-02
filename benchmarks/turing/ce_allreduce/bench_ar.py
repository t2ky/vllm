"""TP2 の all-reduce のマイクロベンチ（P2P の無い GPU 2 枚用。1 枚でも CE_AR_TEST_GPUS=0,0 で動くが、待つカーネルが時分割になり参考にならない）。
大きさ（行 × 3840、fp16）ごとに NCCL・direct（A）・ce（B）・cesm（C）を交互に ROUNDS 周流して中央値（µs）と GB/s（片方向のデータの量 / 時間）。
- eager: 1 回ずつ Python から呼ぶ（CPU の起動の手間込み。両ランクの遅いほう）
- graph: REPS 回を CUDA グラフに捕獲して再生（GPU の時間だけ）
使い方: python bench_ar.py [行の並び] [結果の json]
環境変数: CE_AR_TEST_GPUS（"0,1"）ROUNDS（7）REPS（20）MODES（"nccl,direct,ce,cesm"）H（3840）CE_AR_TEST_PORT
          VLLM_CE_AR_CHUNK_KB・VLLM_CE_AR_BLK_KB・VLLM_CE_AR_NP（ce_allreduce と同じ）"""
import json
import os
import statistics as st
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# ce_allreduce.py lives in vllm/distributed/device_communicators/ of this repository
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "vllm", "distributed", "device_communicators")))
os.environ["VLLM_CE_AR"] = "1"
os.environ.setdefault("VLLM_CE_AR_DIR", HERE)


def w(rank, rows, out):
    gpus = [int(v) for v in os.environ.get("CE_AR_TEST_GPUS", "0,1").split(",")]
    dev = torch.device(f"cuda:{gpus[rank]}")
    torch.cuda.set_device(dev)
    port = int(os.environ.get("CE_AR_TEST_PORT", "29551"))
    same = gpus[0] == gpus[1]  # 1 枚に 2 プロセス: NCCL は同じ GPU では使えないので外す（数字は参考にならない）
    if same:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
        cpu = dist.group.WORLD
    else:
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, device_id=dev)
        cpu = dist.new_group(backend="gloo")
    import ce_allreduce
    ce = ce_allreduce.maybe_create(cpu, rank, 2, dev)
    H = int(os.environ.get("H", "3840"))
    ROUNDS, REPS = int(os.environ.get("ROUNDS", "7")), int(os.environ.get("REPS", "20"))
    modes = [m for m in os.environ.get("MODES", "nccl,direct,ce,cesm").split(",") if not (same and m == "nccl")]

    def op(mode, x):
        if mode == "nccl":
            y = x.clone()
            dist.all_reduce(y)
            return y
        if mode == "direct":
            return ce.ar.direct(x)
        return ce.ar.ce(x, ce.chunk, mode == "cesm")

    def eager_us(mode, x):
        dist.barrier(group=cpu)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS):
            op(mode, x)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / REPS * 1000

    graphs = {}

    def graph_us(mode, x):
        key = (mode, x.shape[0])
        if key not in graphs:
            op(mode, x)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(REPS):
                    op(mode, x)
            graphs[key] = g
        g = graphs[key]
        dist.barrier(group=cpu)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / REPS * 1000

    res = {}
    for M in rows:
        g = torch.Generator(device=dev).manual_seed(M)  # 両ランクで同じ元 × (rank+1)（和は元の 3 倍）
        x = torch.randn(M, H, device=dev, dtype=torch.half, generator=g) * (rank + 1)
        nb = x.numel() * 2
        if nb > ce.max_bytes:
            continue
        ref = op("nccl", x) if "nccl" in modes else (x.float() / (rank + 1) * 3).half()  # 1 枚のとき
        for m in modes:  # 正しさ（NCCL と同じ値。fp16 の 2 項の和なので一致するはず）
            if m != "nccl":
                y = op(m, x)
                if not torch.equal(y, ref) and rank == 0:
                    print(f"  不一致 {m} M={M} 差 {(y.float() - ref.float()).abs().max().item():.2e}", flush=True)
        t = {f"{k}/{m}": [] for k in ("eager", "graph") for m in modes}
        for _ in range(ROUNDS):
            for m in modes:
                t[f"eager/{m}"].append(eager_us(m, x))
                t[f"graph/{m}"].append(graph_us(m, x))
        med = {k: st.median(v) for k, v in t.items()}
        both = [None, None]
        dist.all_gather_object(both, med, group=cpu)
        med = {k: max(both[0][k], both[1][k]) for k in med}  # 遅いほうのランク
        res[M] = med
        if rank == 0:
            def f(k):
                return f"{med[k]:8.1f}µs({nb / med[k] / 1e3:5.2f}GB/s)"
            print(f"M={M:5d} {nb / 1024:8.0f}KB | eager " + " ".join(f"{m} {f('eager/' + m)}" for m in modes)
                  + " | graph " + " ".join(f"{m} {f('graph/' + m)}" for m in modes), flush=True)
    if rank == 0 and out:
        json.dump({"H": H, "res": res, "chunk": ce.chunk, "blk": ce.blk, "np": ce.np}, open(out, "w"), indent=1)
    dist.barrier(group=cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    rows = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4,8,16,32,64,128,256,512,1024,2048").split(",")]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    mp.spawn(w, args=(rows, out), nprocs=2)
