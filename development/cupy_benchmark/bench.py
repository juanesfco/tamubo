"""16 matrix multiplications with CuPy, split over the GPUs of a Slurm step.

Each Slurm task drives one GPU and computes matmuls rank, rank + ntasks, ...
of the 16: 1 GPU does all 16, 8 GPUs do 2 each, 16 GPUs do 1 each. Matrix i
comes from seed i, so it is the same whichever GPU computes it.
"""

import argparse
import os
import socket
import time

import cupy as cp

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=16384, help="matrix size (n x n, float32)")
parser.add_argument("--count", type=int, default=16, help="number of matrix multiplications")
args = parser.parse_args()

rank = int(os.environ.get("SLURM_PROCID", 0))
ntasks = int(os.environ.get("SLURM_NTASKS", 1))
cp.cuda.Device(int(os.environ.get("SLURM_LOCALID", 0))).use()

# This GPU's share of the matrix pairs.
pairs = []
for i in range(rank, args.count, ntasks):
    rng = cp.random.default_rng(i)
    pairs.append((rng.random((args.n, args.n), dtype=cp.float32),
                  rng.random((args.n, args.n), dtype=cp.float32)))

cp.matmul(*pairs[0])  # warm-up (cuBLAS initialization)
cp.cuda.Device().synchronize()

start = time.perf_counter()
results = [cp.matmul(a, b) for a, b in pairs]
cp.cuda.Device().synchronize()
seconds = time.perf_counter() - start

# Check: sum(A @ B) = (column sums of A) . (row sums of B).
ok = True
for (a, b), c in zip(pairs, results):
    expected = float(a.sum(axis=0, dtype=cp.float64) @ b.sum(axis=1, dtype=cp.float64))
    ok &= abs(float(c.sum(dtype=cp.float64)) - expected) < 1e-4 * expected

tflops = 2 * args.n ** 3 * len(pairs) / seconds / 1e12
print(f"RESULT gpus={ntasks} rank={rank} host={socket.gethostname()} matmuls={len(pairs)} "
      f"seconds={seconds:.3f} tflops={tflops:.1f} check={'ok' if ok else 'FAILED'}", flush=True)
