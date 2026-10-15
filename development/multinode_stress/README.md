# Multi-node GPU stress test (CuPy + NCCL, TAMU Vision)

General hardware stress test of Vision nodes (8x H200 141 GB, NVLink/NVSwitch
inside a node, InfiniBand between nodes), independent of `tamubo`. One process
per GPU, launched by `srun`; NCCL comes from the `nvidia-nccl-cu12` wheel in
`envs/tamubo_exactbo` (no MPI needed: rank 0 shares the NCCL id through a file
in the run directory).

```bash
# from the repo root (environment: envs/exactbo.txt, see the repo README)
sbatch development/multinode_stress/stress.sbatch                     # 2 nodes, ~20 min
sbatch --nodes=4 development/multinode_stress/stress.sbatch           # more nodes
sbatch --nodes=1 --time=00:15:00 development/multinode_stress/stress.sbatch --quick
sbatch development/multinode_stress/stress.sbatch --tests collectives matmul
```

## Tests (`stress.py`)

| test | what it does | reported |
|---|---|---|
| `gemm` | n=16384 matmul per GPU, fp64 / fp32 / tf32 / fp16, 10 s each | TFLOPS per GPU, % of H200 dense peak (67 / 67 / 495 / 989) |
| `hbm` | 8 GiB device-to-device copy per GPU | TB/s (read + write), % of 4.8 TB/s |
| `memtest` | fills 90% of free GPU memory with an index hash and its complement, verifies every word (8 passes) | mismatching words per GPU |
| `collectives` | all-reduce, all-gather, reduce-scatter, all-to-all, 8 KiB-2 GiB, at three scopes run concurrently: **node** (8 GPUs, NVLink), **rail** (GPU i of every node, one NIC each, all rails at once), **global** (all GPUs) | bus bandwidth (nccl-tests convention), worst group, result check |
| `matmul` | C = A B with 16384x16384 fp16 blocks on a ring of GPUs: each step multiplies the current B block while NCCL send/recv passes it on; run compute-only, comm-only and overlapped, node and global scope | total TFLOPS, overlap efficiency, ring bandwidth, per-block checksum |
| `burn` | 10 min of back-to-back fp16 GEMMs on every GPU plus a concurrent global 256 MiB all-reduce, in 30 s windows; each window recomputes one GEMM and compares it bit-for-bit with a reference | TFLOPS per window (min / median / slowest GPU), all-reduce bandwidth, mismatches, drift first -> last window |

GPUs below 90% of the median are listed by `host:gpuN`. `--quick` shortens
everything (1 s GEMMs, 3 message sizes, 60 s burn).

## Output

- `logs/stress-<jobid>.out`: the tables above, then `summarize.py`'s report:
  InfiniBand ports per node, NCCL transports (P2P/NVLink, NET/IB, GDRDMA), and
  per-GPU telemetry while busy (max temperature and power, SM clock, % of
  samples with power-cap / thermal / HW slowdown).
- `logs/<jobid>/results.jsonl`: one JSON row per GPU, group or window.
- `logs/<jobid>/smi_<host>.csv`: `nvidia-smi` every 2 s; `node_<host>.txt`:
  GPU serials, `nvidia-smi topo -m`, `ibstat`; `nccl.<host>.<pid>.log`.

## Results (job 716004, 2 nodes dgx[066,075], 16x H200, 13.7 min)

Everything passed: no memory mismatches, every collective and ring-matmul
checksum correct, and 10 min of burn with every GEMM bit-identical to its
reference. No GPU was below 90% of the median in any test.

Per GPU (min-max over the 16 GPUs):

| | measured | of H200 dense peak |
|---|---:|---:|
| fp64 GEMM | 53.2-55.8 TFLOPS | 82% |
| fp32 GEMM | 53.9-54.1 TFLOPS | 81% |
| tf32 GEMM | 357-376 TFLOPS | 74% |
| fp16 GEMM | 678-696 TFLOPS | 69% |
| HBM copy | 4.29-4.32 TB/s | 90% |
| memtest | 124 GiB per GPU, 8 passes | 0 errors |

NCCL bus bandwidth (GB/s, 2 GiB messages; rail = all 8 rails at once):

| | node (NVLink) | rail (IB, 1 NIC per GPU) | global (16 GPUs) |
|---|---:|---:|---:|
| all-reduce | 475 | 46-48 | 472 |
| all-gather | 350 | 40-42 | 324 |
| reduce-scatter | 352 | 45-46 | 345 |
| all-to-all | 340 | 39 | 91 |

- Each rail runs at ~47 GB/s = 376 Gb/s, 94% of its 400 Gb/s NDR link; NCCL
  uses the 8 GPU-local HCAs (mlx5_0,3,4,5,6,9,10,11) evenly with GPUDirect RDMA
  and never the storage/Ethernet ports (mlx5_1,2,7,8).
- Global all-reduce keeps node-level bandwidth (hierarchical over 8 rails).
  All-to-all is network-bound: half of each GPU's data crosses the IB link.
- Latency (8 KiB): 16 us all-reduce in a node, 29 us over 2 nodes.

Ring matmul (fp16, 16384^2 blocks): 646 TFLOPS per GPU over 16 GPUs
(10.3 PFLOPS total), 94% overlap efficiency; the inter-node hop (43.6 GB/s)
nearly matches the compute time (0.185 s comm vs 0.204 s compute), so a
smaller block size would be network-bound.

Burn (fp16 GEMM + global all-reduce, 10 min): 663-670 TFLOPS per GPU in every
window, drift first -> last window -0.1% (worst -0.2%), all-reduce 143 GB/s
under load. Every GPU sits at its 700 W limit (`sw power cap` in 83-84% of
busy samples, SM clock median 1425-1485 MHz vs 1980 max), max 70 C GPU and
64 C memory: power-limited, not thermally limited, which is why fp16 GEMM
reaches 69% of the boost-clock peak.
