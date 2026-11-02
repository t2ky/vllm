"""2 プロセスを同じ GPU に載せて ce_ar（CUDA C++ の all-reduce）の正しさを試す（GPU 1 枚で 2 プロセス。2 枚でも動く）。
- 方式 direct / ce / cesm × dtype（fp16・bf16・fp32）× 大きさ（1〜2048 行 × 3840 など）を混ぜて N 回（面・カウンタの使い回し）
- ときどき片方を torch.cuda._sleep で遅らせる
- 期待値 = (x0.float() + x1.float()) を dtype に丸めたもの と**ビットまで一致**するか（両ランクとも一致すればランク間も一致）
- CUDA グラフ: いくつかの all-reduce を 1 本のグラフに捕獲し、入力を書き換えて何度も再生して、毎回一致するか
使い方: python test_ce_ar.py [回数=400] [グラフの再生回数=50]   環境変数 CE_AR_TEST_GPUS="0,0"（ランクごとの GPU）"""
import os
import random
import time
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# ce_allreduce.py lives in vllm/distributed/device_communicators/ of this repository
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "vllm", "distributed", "device_communicators")))
os.environ["VLLM_CE_AR"] = "1"
os.environ.setdefault("VLLM_CE_AR_DIR", HERE)
VERB = os.environ.get("CE_AR_TEST_VERBOSE") == "1"
DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def inputs(it, M, H, dt):
    xs = []
    for r in (0, 1):
        g = torch.Generator(device="cuda").manual_seed(it * 2 + r)
        xs.append((torch.randn(M, H, device="cuda", generator=g) * (1 + r)).to(DT[dt]))
    return xs


def w(rank, iters, replays):
    gpus = [int(v) for v in os.environ.get("CE_AR_TEST_GPUS", "0,0").split(",")]
    torch.cuda.set_device(gpus[rank])
    os.environ["VLLM_CE_AR_MAX_MB"] = "16"
    os.environ["VLLM_CE_AR_CHUNK_KB"] = "256"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{os.environ.get('CE_AR_TEST_PORT', '29541')}", rank=rank, world_size=2)
    import ce_allreduce
    ce = ce_allreduce.maybe_create(dist.group.WORLD, rank, 2, torch.device(f"cuda:{gpus[rank]}"))
    rng = random.Random(0)  # 両ランクで同じ並び
    bad, n = 0, 0
    for it in range(iters):
        M = rng.choice([1, 2, 16, 37, 128, 512, 1000, 2048])
        H = rng.choice([3840, 3840, 2048, 4096])
        dt = rng.choice(["fp16", "fp16", "bf16", "fp32"])
        mode = rng.choice(["direct", "ce", "cesm"])
        if M * H * DT[dt].itemsize > ce.max_bytes:
            M = ce.max_bytes // (H * DT[dt].itemsize)
        xs = inputs(it, M, H, dt)
        slow, who, cyc = rng.random() < 0.2, rng.randint(0, 1), rng.randint(0, 3_000_000)  # 乱数は両ランクで同じだけ引く
        if slow and who == rank:
            torch.cuda._sleep(cyc)  # 片方を遅らせる
        if VERB:
            print(f"{time.time() % 1000:8.3f} rank{rank} it={it} {mode} {dt} M={M} H={H}", flush=True)
        y = ce.ar.direct(xs[rank]) if mode == "direct" else ce.ar.ce(xs[rank], ce.chunk, mode == "cesm")
        ref = (xs[0].float() + xs[1].float()).to(DT[dt])
        if VERB:
            ev = torch.cuda.Event()
            ev.record()
            t0 = time.time()
            while not ev.query():
                if time.time() - t0 > 20:
                    f = ce.flags()
                    nch = 64
                    print(f"rank{rank} 止まった it={it}: wB0 {f['wB'][0][:nch]}\n wB1 {f['wB'][1][:nch]}\n rB {f['rB']}"
                          f"\n wA0 {f['wA'][0][:4]} wA1 {f['wA'][1][:4]} rA0 {f['rA'][0][:4]} rA1 {f['rA'][1][:4]}", flush=True)
                    os._exit(1)
                time.sleep(0.01)
        ok = torch.equal(y, ref)
        n += 1
        if not ok:
            bad += 1
            if bad <= 5:
                print(f"rank{rank} NG it={it} {mode} {dt} M={M} H={H} 差 {(y.float() - ref.float()).abs().max().item():.3e}",
                      flush=True)
    torch.cuda.synchronize()
    print(f"rank{rank} 逐次: {n} 回中 不一致 {bad}", flush=True)

    # CUDA グラフ: 12 個の all-reduce（方式・大きさを混ぜる）を捕獲して再生
    plan = [(rng.choice([1, 16, 128, 1024]), 3840, rng.choice(["fp16", "bf16"]), rng.choice(["direct", "ce", "cesm"]))
            for _ in range(12)]
    static = [torch.zeros(M, H, device="cuda", dtype=DT[dt]) for M, H, dt, _ in plan]
    outs = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for (M, H, dt, mode), x in zip(plan, static):
                outs.append(ce.ar.direct(x) if mode == "direct" else ce.ar.ce(x, ce.chunk, mode == "cesm"))
    torch.cuda.synchronize()
    gbad = 0
    for r in range(replays):
        refs = []
        for i, ((M, H, dt, mode), x) in enumerate(zip(plan, static)):
            xs = inputs(100000 + r * 100 + i, M, H, dt)
            x.copy_(xs[rank])
            refs.append((xs[0].float() + xs[1].float()).to(DT[dt]))
        if r % 5 == 0 and rank == r % 2:
            torch.cuda._sleep(2_000_000)
        g.replay()
        torch.cuda.synchronize()
        for o, ref in zip(outs, refs):
            gbad += int(not torch.equal(o, ref))
        if r % 3 == 0:  # グラフの間に逐次の呼び出しも挟む（カウンタがグラフの中と外で共通に進む）
            xs = inputs(200000 + r, 16, 3840, "fp16")
            y = ce.ar.direct(xs[rank])
            gbad += int(not torch.equal(y, (xs[0].float() + xs[1].float()).half()))
    print(f"rank{rank} グラフ: {replays} 回の再生（12 個ずつ）で不一致 {gbad}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    replays = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    mp.spawn(w, args=(iters, replays), nprocs=2)
