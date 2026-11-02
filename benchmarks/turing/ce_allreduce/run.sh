#!/bin/bash
# P2P の無い GPU 2 枚（例 RTX 2080 × 2）で ce_ar（ホストメモリ経由の TP2 all-reduce、CUDA C++）を確かめて測る。
# イメージは vllm-turing:v0.31.0-tt1。build/ に sm_75 の SASS 入りの拡張があるので nvcc は要らない（.cu が新しければ作り直す）。
# この ディレクトリ ごと Pod に置いて（例 /ce-ar）、GPU 2 枚が見える状態で:
#   bash run.sh test    正しさ（3 方式 × fp16/bf16/fp32 × 1〜2048 行、片方を遅らせる、面の使い回し、CUDA グラフの再生）。最初に 1 回
#   bash run.sh bench   マイクロベンチ（大きさごとの µs・GB/s、NCCL と比べる、eager と CUDA グラフ）
#   bash run.sh sweep   ce・cesm のチャンク（256KB / 1MB / 2MB）と direct の BLK（4 / 8 / 16KB）を変えて bench
# 結果は ./out/ に。vLLM で使うときは README 代わりに ce_allreduce.py の先頭の説明を見る（mount 3 つと環境変数）
set -e
MODE=${1:-bench}
cd "$(dirname "$0")"; mkdir -p out
export CE_AR_TEST_GPUS=${CE_AR_TEST_GPUS:-0,1}
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv
nvidia-smi topo -m || true
F="grep -v -i warn"
case $MODE in
  test)  python3 test_ce_ar.py 400 50 2>&1 | $F | tee out/test.log ;;
  bench) python3 bench_ar.py 1,2,4,8,16,32,64,128,256,512,1024,2048 out/bench.json 2>&1 | $F | tee out/bench.log ;;
  sweep)
    for C in 256 1024 2048; do
      VLLM_CE_AR_CHUNK_KB=$C MODES=nccl,ce,cesm python3 bench_ar.py 128,256,512,1024,2048 out/sweep_chunk$C.json 2>&1 | $F | tee out/sweep_chunk$C.log
    done
    for B in 4 8 16; do
      VLLM_CE_AR_BLK_KB=$B MODES=nccl,direct python3 bench_ar.py 1,4,16,64,256,1024,2048 out/sweep_blk$B.json 2>&1 | $F | tee out/sweep_blk$B.log
    done ;;
esac
