# CuPy matmul benchmark: 1 GPU vs 8 GPUs vs 16 GPUs (TAMU Vision)

16 float32 matrix multiplications (16384 x 16384) with CuPy. Each Slurm task
drives one GPU and takes its share: 1 GPU does all 16, 8 GPUs (1 node) do 2
each, 16 GPUs (2 nodes) do 1 each. The GPUs work independently (no
communication). Each result is checked with sum(A B) = colsum(A) . rowsum(B).

```bash
# from the repo root (environment: envs/exactbo.txt, see the repo README)
sbatch development/cupy_benchmark/bench.sbatch
```

Output: `logs/bench-<jobid>.out`, one `RESULT` line per GPU and the summary.
The time for a setup is its slowest GPU. Only the matmuls are timed, not
creating the matrices.

## Results (job 743164, dgx[021-022], H200)

| GPUs | matmuls per GPU | time | speedup |
|---:|---:|---:|---:|
| 1 | 16 | 2.606 s | 1.0x |
| 8 (1 node) | 2 | 0.326 s | 8.0x |
| 16 (2 nodes) | 1 | 0.163 s | 16.0x |

Every GPU ran at 54 TFLOPS (81% of the H200's 67 TFLOPS fp32 peak) and every
check passed.
