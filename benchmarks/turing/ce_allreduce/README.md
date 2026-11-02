# Host-memory TP=2 all-reduce for GPUs without P2P (reference, not adopted)

An experiment for TP=2 on two consumer Turing GPUs (RTX 2080) without NVLink and without PCIe P2P.
NCCL then goes through host shared memory. The idea was to do the all-reduce ourselves through pinned,
mapped host memory shared by the two ranks, using the copy engines or direct loads/stores over PCIe.
**It ended up no faster than NCCL, so it is kept on this branch for reference only and is not part of the
Turing image.**

## Files

- `ce_ar.cu` - CUDA C++ extension. The call counter lives in GPU memory (both ranks issue the same
  all-reduces in the same order), so the host passes no per-call values and the calls can be captured in
  CUDA graphs. Three modes:
  - `direct`: one kernel. Block b always owns the same BLK bytes (default 8 KB) of NP (default 2) buffers.
    Wait until the peer has read buffer `n % NP` -> write own data -> `fence.sys` + `st.release.sys` flag ->
    `ld.acquire.sys` wait for the peer's flag -> read the peer's data with volatile loads, add in fp32
    (a + b is commutative, so both ranks get identical results) -> release the buffer. Blocks never wait
    for each other, only for the same block on the peer.
  - `ce`: one buffer (fixed addresses so copies can be graph-captured), D2H / H2D per chunk on the copy
    engines; flag writes and waits are tiny kernels that derive values from the GPU counter.
  - `cesm`: send with the copy engine, receive with a kernel that reads the peer's buffer directly.
- `../../../vllm/distributed/device_communicators/ce_allreduce.py` - Python side (shared memory segment,
  `cudaHostRegister`, mode selection by size); `cuda_communicator.py` calls it at the start of
  `all_reduce` when `VLLM_CE_AR=1`. See the module docstring for all variables.
- `test_ce_ar.py` - correctness: 3 modes x fp16/bf16/fp32 x 1-2048 rows, one rank delayed at random,
  buffer reuse, CUDA graph replay. Expects bit-exact results on both ranks. Runs with two processes on one
  GPU (`CE_AR_TEST_GPUS=0,0`) or on two GPUs.
- `bench_ar.py`, `run.sh` - microbenchmark against NCCL (eager and in CUDA graphs) and sweeps.

## Lessons

- Running the `cesm` receive kernel concurrently with the same GPU's send-flag kernel deadlocked both
  ranks (the resident receive kernel blocked the flag kernel). The receive now starts after the own send.
  `direct` only waits for the same block on the peer, and `ce` enqueues all sends first.
- Polling flags across PCIe without pause steals bandwidth from the DMA: back off with `__nanosleep`
  (32 ns doubling up to 1 us).
- Do not `record_stream` the receive buffers: the CPU runs almost a step ahead, the frees are delayed by
  that much and the GPU runs out of memory. The main stream already waits for the side streams.
- An earlier Python + ctypes version needed about 25 driver calls per all-reduce and could not be graph
  captured; it also made the host-side gaps larger.

## Results (2x RTX 2080, PCIe gen3 x16, same CPU socket, no P2P)

Correctness: 400 sequential calls and 50 graph replays on two GPUs, no mismatches.

Microbenchmark inside CUDA graphs (one all-reduce, fp16):

| size | NCCL | this |
|---|---:|---:|
| 120 KB (decode) | 52.2 us | 41.9 us (`direct`, BLK 16 KB) |
| 480 KB | 128 us | 119 us |
| 1.9 MB | 377 us (best) | slower |
| 7.5 MB | 1,459 us | 1,378 us (`ce`, 2 MB chunks) |
| 15 MB | 2,916 us | 2,611 us |

End to end (Gemma 4 12B, TP=2, fixed load of 96 requests): NCCL 42.32 s, this (direct up to 480 KB, ce
from 7 MB, NCCL in between) 43.56 s, i.e. no difference. The median small all-reduce inside vLLM was
49.0 us with NCCL and 50.8 us with `direct`.

An earlier comparison that showed 887 -> 552 us for the large all-reduce was wrong: it compared medians
of different profiling regions whose size distributions differed. At equal sizes the copy engine is only
6-10% faster, and only from 7.5 MB up.

Also note that a GPU in a PCIe x8 slot makes every all-reduce (NCCL or this) 20-50% slower; pin TP pairs to
x16 slots on the same socket.

## Conclusion

Through host memory, NCCL is already close to the latency/bandwidth floor of PCIe plus host memory, so
there is little left to gain here. Only a direct GPU link (NVLink with P2P) would cut the communication
substantially. For reference, replacing the all-reduce with a no-op (wrong outputs, same compute) made the
same benchmark 22.6% faster, which bounds what any communication change can give.
