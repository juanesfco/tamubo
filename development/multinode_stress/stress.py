"""Multi-node GPU stress test for TAMU Vision (CuPy + NCCL, one process per GPU).

Tests (``--tests``, run in this order):

- ``gemm``: per-GPU matrix-multiply throughput (fp64, fp32, tf32, fp16).
- ``hbm``: per-GPU device-to-device copy bandwidth.
- ``memtest``: fills most of each GPU's free memory with an index pattern and
  its complement and verifies every word.
- ``collectives``: NCCL all-reduce / all-gather / reduce-scatter / all-to-all
  bus bandwidth (nccl-tests convention) inside each node (NVLink), along each
  rail (the same local GPU on every node, InfiniBand) and over all GPUs.
- ``matmul``: distributed C = A B on a ring of GPUs (each B block is passed on
  with NCCL send/recv while the current one is multiplied), node and global
  scope, verified with per-block checksums.
- ``burn``: sustained fp16 GEMMs on every GPU with a concurrent global
  all-reduce, reported per time window; each window recomputes one GEMM and
  compares it bit-for-bit with a reference.

Launch with ``stress.sbatch`` (srun, one task per GPU). Rank 0 prints the
tables and writes ``results.jsonl`` to ``--out``; local rank 0 of every node
logs ``nvidia-smi`` telemetry and node info there (see ``summarize.py``).
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback

import cupy as cp
import numpy as np
from cupy._core import _routines_linalg as _linalg
from cupy.cuda import nccl

# H200 SXM dense peaks (TFLOPS) and HBM3e bandwidth (TB/s), NVIDIA datasheet.
PEAK_TFLOPS = {"fp64": 67.0, "fp32": 67.0, "tf32": 494.7, "fp16": 989.4}
PEAK_HBM_TBS = 4.8

# name -> (dtype, cuBLAS compute type, or None for CuPy's default)
GEMM_TYPES = {
    "fp64": (np.float64, None),
    "fp32": (np.float32, None),
    "tf32": (np.float32, _linalg.COMPUTE_TYPE_TF32),
    "fp16": (np.float16, None),
}

COLLECTIVES = ("allreduce", "allgather", "reducescatter", "alltoall")

_FILL = cp.ElementwiseKernel(
    "uint32 seed", "uint32 x",
    "x = ((unsigned int)((unsigned long long)i * 2654435761ull)) ^ seed",
    "stress_fill")
_CHECK = cp.ElementwiseKernel(
    "uint32 x, uint32 seed", "raw uint64 bad",
    "if (x != (((unsigned int)((unsigned long long)i * 2654435761ull)) ^ seed))"
    " atomicAdd(&bad[0], 1ull)",
    "stress_check")


class Ctx:
    """Ranks from Slurm; a global NCCL communicator plus node and rail splits."""

    def __init__(self, out_dir):
        env = os.environ
        self.rank, self.world = int(env["SLURM_PROCID"]), int(env["SLURM_NTASKS"])
        self.local, self.node = int(env["SLURM_LOCALID"]), int(env["SLURM_NODEID"])
        self.nnodes = int(env["SLURM_NNODES"])
        if self.world % self.nnodes:
            raise SystemExit("stress.py needs the same number of tasks on every node")
        self.host = socket.gethostname().split(".")[0]
        self.out = out_dir
        # All GPUs visible (no per-task binding): pick ours by local rank.
        ndev = cp.cuda.runtime.getDeviceCount()
        self.device = cp.cuda.Device(self.local if ndev > 1 else 0)
        self.device.use()
        cp.random.seed(1234 + self.rank)

        self.comm = self._init_comm()
        self.node_comm = self.comm.commSplit(self.node, self.local)
        self.rail_comm = self.comm.commSplit(self.local, self.node)
        self.info = self._exchange_info()
        self.labels = [f"{d['host']}:gpu{d['gpu']}" for d in self.info]
        self.results = (open(os.path.join(out_dir, "results.jsonl"), "a")
                        if self.rank == 0 else None)

    def _init_comm(self):
        path = os.path.join(self.out, "nccl_uid.json")
        if self.rank == 0:
            with open(path + ".tmp", "w") as f:
                uid = nccl.get_unique_id()  # bytes in CuPy >= 14, a tuple of ints before
                json.dump(bytes(b & 0xFF for b in uid).hex() if isinstance(uid, tuple)
                          else uid.hex(), f)
            os.replace(path + ".tmp", path)
        else:
            deadline = time.time() + 600
            while not os.path.exists(path):
                if time.time() > deadline:
                    raise TimeoutError(f"rank {self.rank}: no {path} after 600 s")
                time.sleep(0.2)
            time.sleep(0.5)
        with open(path) as f:
            uid = bytes.fromhex(json.load(f))
        return nccl.NcclCommunicator(self.world, uid, self.rank)

    def _exchange_info(self):
        props = cp.cuda.runtime.getDeviceProperties(self.device.id)
        uuid = props.get("uuid", b"")
        me = {"rank": self.rank, "host": self.host, "node": self.node, "gpu": self.local,
              "name": props["name"].decode(), "pci_bus": props["pciBusID"],
              "uuid": uuid.hex() if isinstance(uuid, bytes) else str(uuid)}
        rank_dir = os.path.join(self.out, "ranks")
        os.makedirs(rank_dir, exist_ok=True)
        with open(os.path.join(rank_dir, f"{self.rank:04d}.json"), "w") as f:
            json.dump(me, f)
        self.barrier()
        info = []
        for r in range(self.world):
            with open(os.path.join(rank_dir, f"{r:04d}.json")) as f:
                info.append(json.load(f))
        return info

    def barrier(self):
        x = cp.zeros(1, cp.float32)
        stream = cp.cuda.get_current_stream()
        self.comm.allReduce(x.data.ptr, x.data.ptr, 1, nccl.NCCL_FLOAT32,
                            nccl.NCCL_SUM, stream.ptr)
        stream.synchronize()

    def gather(self, values):
        """Every rank's float vector (same length on all ranks), as (world, len)."""
        x = cp.asarray(np.asarray(values, dtype=np.float64).ravel())
        out = cp.empty(self.world * x.size, cp.float64)
        stream = cp.cuda.get_current_stream()
        self.comm.allGather(x.data.ptr, out.data.ptr, x.size, nccl.NCCL_FLOAT64, stream.ptr)
        stream.synchronize()
        return cp.asnumpy(out).reshape(self.world, x.size)

    def log(self, *args):
        if self.rank == 0:
            print(*args, flush=True)

    def record(self, **row):
        if self.rank == 0:
            self.results.write(json.dumps(row, default=lambda o: o.item()) + "\n")
            self.results.flush()

    def close(self):
        for comm in (self.rail_comm, self.node_comm, self.comm):
            comm.destroy()
        if self.results:
            self.results.close()


def free_pool():
    cp.get_default_memory_pool().free_all_blocks()


def random_matrix(shape, dtype):
    x = cp.random.random(shape, dtype=np.float64 if dtype == np.float64 else np.float32)
    return x.astype(dtype, copy=False)


def time_loop(fn, seconds):
    """Seconds per call of ``fn`` on the current stream, one sample per ~0.25 s batch."""
    stream = cp.cuda.get_current_stream()
    for _ in range(2):
        fn()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    fn()
    stop.record(stream)
    stop.synchronize()
    batch = max(1, int(0.25 / max(cp.cuda.get_elapsed_time(start, stop) / 1e3, 1e-6)))
    samples = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end or len(samples) < 3:
        start.record(stream)
        for _ in range(batch):
            fn()
        stop.record(stream)
        stop.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start, stop) / 1e3 / batch)
    return samples


def report_per_gpu(ctx, test, key, values, unit, peak=None, **extra):
    """Rank 0: one summary line over GPUs, flag GPUs below 90% of the median."""
    if ctx.rank:
        return
    for r, v in enumerate(values):
        ctx.record(test=test, key=key, gpu=ctx.labels[r], value=float(v), unit=unit,
                   **{k: float(e[r]) for k, e in extra.items()})
    med = float(np.median(values))
    line = (f"  {key:<8} min {values.min():9.2f}  median {med:9.2f}  "
            f"max {values.max():9.2f} {unit}")
    if peak:
        line += f"   ({100 * med / peak:.0f}% of peak {peak:g})"
    ctx.log(line)
    slow = [f"{ctx.labels[r]} {v:.2f}" for r, v in enumerate(values) if v < 0.9 * med]
    if slow:
        ctx.log(f"    below 90% of median: {', '.join(slow)}")


# --- per-GPU tests -----------------------------------------------------------------

def test_gemm(ctx, args):
    n = args.gemm_n
    ctx.log(f"\n== GEMM per GPU (n={n}, {args.gemm_seconds:g} s per type)")
    flops = 2.0 * n ** 3
    for name in args.gemm_types:
        dtype, compute = GEMM_TYPES[name]
        a, b = random_matrix((n, n), dtype), random_matrix((n, n), dtype)
        c = cp.empty((n, n), dtype)
        prev = _linalg.get_compute_type(dtype)
        if compute is not None:
            _linalg.set_compute_type(dtype, compute)
        try:
            samples = time_loop(lambda: cp.matmul(a, b, out=c), args.gemm_seconds)
        finally:
            _linalg.set_compute_type(dtype, prev)
        per = ctx.gather([flops / np.median(samples) / 1e12, flops / max(samples) / 1e12])
        report_per_gpu(ctx, "gemm", name, per[:, 0], "TFLOPS", PEAK_TFLOPS[name],
                       worst_batch=per[:, 1])
        del a, b, c
        free_pool()


def test_hbm(ctx, args):
    nbytes = args.hbm_gib << 30
    ctx.log(f"\n== HBM device-to-device copy per GPU ({args.hbm_gib} GiB buffers)")
    src = cp.ones(nbytes, cp.uint8)
    dst = cp.empty_like(src)
    samples = time_loop(lambda: dst.data.copy_from_device_async(src.data, nbytes),
                        args.gemm_seconds)
    per = ctx.gather([2 * nbytes / np.median(samples) / 1e12])
    report_per_gpu(ctx, "hbm", "copy", per[:, 0], "TB/s", PEAK_HBM_TBS)
    del src, dst
    free_pool()


def test_memtest(ctx, args):
    free_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    words = 1 << 28  # 1 GiB chunks of uint32
    chunks = []
    try:
        for _ in range(int(free * args.mem_frac) // (words * 4)):
            chunks.append(cp.empty(words, cp.uint32))
    except cp.cuda.memory.OutOfMemoryError:
        pass
    ctx.log(f"\n== Memory fill/verify per GPU ({args.mem_passes} passes, "
            f"{args.mem_frac:.0%} of free memory)")
    bad = cp.zeros(1, cp.uint64)
    t0 = time.perf_counter()
    for p in range(args.mem_passes):
        flip = 0xFFFFFFFF if p % 2 else 0  # odd passes write the complement
        seeds = [np.uint32(((k * 0x9E3779B9) & 0xFFFFFFFF) ^ flip) for k in range(len(chunks))]
        for x, s in zip(chunks, seeds):
            _FILL(s, x)
        for x, s in zip(chunks, seeds):
            _CHECK(x, s, bad)
    cp.cuda.get_current_stream().synchronize()
    seconds = time.perf_counter() - t0
    gib = float(len(chunks))
    per = ctx.gather([gib, float(bad.get()[0]), total / 2 ** 30, seconds])
    del chunks
    free_pool()
    if ctx.rank == 0:
        for r in range(ctx.world):
            ctx.record(test="memtest", gpu=ctx.labels[r], gib=per[r, 0],
                       errors=int(per[r, 1]), total_gib=per[r, 2], seconds=per[r, 3])
        ctx.log(f"  tested {per[:, 0].min():.0f}-{per[:, 0].max():.0f} GiB per GPU "
                f"(of {per[0, 2]:.0f} GiB), {per[:, 3].max():.1f} s")
        bad_gpus = [f"{ctx.labels[r]} ({int(per[r, 1])} words)"
                    for r in range(ctx.world) if per[r, 1]]
        ctx.log(f"  mismatches: {', '.join(bad_gpus) if bad_gpus else 'none'}")


# --- collectives -------------------------------------------------------------------

def parse_size(text):
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    text = text.upper().rstrip("B")
    return int(float(text[:-1]) * units[text[-1]]) if text[-1] in units else int(text)


def fmt_size(nbytes):
    for unit, scale in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if nbytes >= scale:
            return f"{nbytes / scale:g}{unit}"
    return f"{nbytes}B"


def scopes(ctx):
    """(name, communicator, this rank's group id, group label per group id)."""
    out = [("node", ctx.node_comm, ctx.node,
            {d["node"]: d["host"] for d in ctx.info})]
    if ctx.nnodes > 1:
        out.append(("rail", ctx.rail_comm, ctx.local,
                    {d["gpu"]: f"gpu{d['gpu']} x{ctx.nnodes} nodes" for d in ctx.info}))
        out.append(("global", ctx.comm, 0, {0: f"{ctx.world} GPUs"}))
    return out


def group_of(ctx, scope, r):
    d = ctx.info[r]
    return {"node": d["node"], "rail": d["gpu"], "global": 0}[scope]


def run_collective(op, comm, send, recv, count, stream):
    n, ptr, f32 = comm.size(), stream.ptr, nccl.NCCL_FLOAT32
    if op == "allreduce":
        comm.allReduce(send.data.ptr, recv.data.ptr, count, f32, nccl.NCCL_SUM, ptr)
    elif op == "allgather":
        comm.allGather(send.data.ptr, recv.data.ptr, count // n, f32, ptr)
    elif op == "reducescatter":
        comm.reduceScatter(send.data.ptr, recv.data.ptr, count // n, f32, nccl.NCCL_SUM, ptr)
    elif op == "alltoall":
        chunk = count // n
        nccl.groupStart()
        for peer in range(n):
            comm.send(send.data.ptr + 4 * peer * chunk, chunk, f32, peer, ptr)
            comm.recv(recv.data.ptr + 4 * peer * chunk, chunk, f32, peer, ptr)
        nccl.groupEnd()


def prepare_and_check(op, comm, send, recv, count):
    """Fill ``send`` for ``op``; return a function checking ``recv`` afterwards."""
    n, r = comm.size(), comm.rank_id()
    idx = cp.arange(n, dtype=cp.float32)[:, None]
    if op in ("allreduce", "reducescatter"):
        send.fill(1)
        size = count if op == "allreduce" else count // n
        return lambda: bool(cp.all(recv[:size] == n))
    if op == "allgather":
        send[:count // n] = r
        return lambda: bool(cp.all(recv[:count].reshape(n, -1) == idx))
    send[:count].reshape(n, -1)[...] = r * n + idx
    return lambda: bool(cp.all(recv[:count].reshape(n, -1) == idx * n + r))


def test_collectives(ctx, args):
    sizes = [parse_size(s) for s in args.coll_sizes]
    stream = cp.cuda.Stream(non_blocking=True)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    for scope, comm, _, labels in scopes(ctx):
        n = comm.size()
        ctx.log(f"\n== NCCL {scope} ({n} GPUs per group, {len(labels)} groups at once), "
                f"bus bandwidth GB/s over groups")
        ctx.log(f"  {'op':<14}{'size':>7}{'min':>9}{'median':>9}{'max':>9}"
                f"{'time (us)':>12}  worst group / check")
        for op in COLLECTIVES:
            for nbytes in sizes:
                count = max(n, (nbytes // 4) // n * n)
                with stream:
                    send = cp.empty(count, cp.float32)
                    recv = cp.empty(count, cp.float32)
                    check = prepare_and_check(op, comm, send, recv, count)
                    iters = int(np.clip((1 << 31) // (4 * count), 5, 200))
                    for _ in range(3):
                        run_collective(op, comm, send, recv, count, stream)
                    start.record(stream)
                    for _ in range(iters):
                        run_collective(op, comm, send, recv, count, stream)
                    stop.record(stream)
                    stop.synchronize()
                    t = cp.cuda.get_elapsed_time(start, stop) / 1e3 / iters
                    ok = check()
                del send, recv
                per = ctx.gather([t, ok])
                if ctx.rank:
                    continue
                factor = 2 * (n - 1) / n if op == "allreduce" else (n - 1) / n
                groups = {}
                for r in range(ctx.world):
                    g = group_of(ctx, scope, r)
                    groups[g] = max(groups.get(g, 0.0), per[r, 0])
                bw = {g: 4 * count / tg * factor / 1e9 for g, tg in groups.items()}
                vals = np.array(list(bw.values()))
                worst = min(bw, key=bw.get)
                failed = [ctx.labels[r] for r in range(ctx.world) if not per[r, 1]]
                for g, b in bw.items():
                    ctx.record(test="collective", scope=scope, op=op, bytes=4 * count,
                               group=labels[g], busbw_gbs=b, time_s=groups[g],
                               ok=not any(group_of(ctx, scope, r) == g and not per[r, 1]
                                          for r in range(ctx.world)))
                note = "" if len(bw) == 1 else labels[worst]
                note += ("  FAILED on " + ", ".join(failed)) if failed else ("  ok" if note else "ok")
                ctx.log(f"  {op:<14}{fmt_size(4 * count):>7}{vals.min():9.1f}"
                        f"{np.median(vals):9.1f}{vals.max():9.1f}"
                        f"{1e6 * max(groups.values()):12.1f}  {note}")
        free_pool()


# --- distributed matmul ------------------------------------------------------------

def ring_matmul(comm, a, b0, bufs, out, s_comp, s_comm, compute=True, communicate=True):
    """out[j] = a @ B_j for every rank's block B_j, passing blocks around the ring."""
    P, r = comm.size(), comm.rank_id()
    cur, nxt = bufs
    cur.data.copy_from_device_async(b0.data, b0.nbytes, s_comp)
    s_comp.synchronize()
    for step in range(P):
        if communicate and step < P - 1:
            nccl.groupStart()
            comm.send(cur.data.ptr, cur.size, nccl.NCCL_FLOAT16, (r + 1) % P, s_comm.ptr)
            comm.recv(nxt.data.ptr, nxt.size, nccl.NCCL_FLOAT16, (r - 1) % P, s_comm.ptr)
            nccl.groupEnd()
        if compute:
            with s_comp:
                cp.matmul(a, cur, out=out[(r - step) % P])
        s_comm.synchronize()
        s_comp.synchronize()
        cur, nxt = nxt, cur


def ring_checksum_error(comm, a, b0, out):
    """Max relative error of sum(out[j]) against colsum(a) . rowsum(B_j)."""
    P, k = comm.size(), b0.shape[0]
    col_a = a.sum(axis=0, dtype=cp.float64)
    row_b = b0.sum(axis=1, dtype=cp.float64)
    all_rows = cp.empty((P, k), cp.float64)
    stream = cp.cuda.get_current_stream()
    comm.allGather(row_b.data.ptr, all_rows.data.ptr, k, nccl.NCCL_FLOAT64, stream.ptr)
    stream.synchronize()
    expected = all_rows @ col_a
    got = out.reshape(P, -1).sum(axis=1, dtype=cp.float64)
    return float(cp.max(cp.abs(got - expected) / expected))


def test_matmul(ctx, args):
    m = args.mm_n
    ctx.log(f"\n== Distributed ring matmul (fp16, {m}x{m} blocks per GPU, "
            f"{args.mm_reps} reps, median)")
    ctx.log(f"  {'scope':<8}{'GPUs':>5}{'TFLOPS total':>14}{'per GPU':>9}{'compute s':>11}"
            f"{'comm s':>9}{'overlap s':>11}{'overlap eff':>13}{'ring GB/s':>11}  check")
    a, b0 = random_matrix((m, m), np.float16), random_matrix((m, m), np.float16)
    bufs = [cp.empty_like(b0), cp.empty_like(b0)]
    ctx.device.synchronize()  # inputs come from the default stream, used on s_comp/s_comm
    s_comp = cp.cuda.Stream(non_blocking=True)
    s_comm = cp.cuda.Stream(non_blocking=True)
    for scope, comm, _, labels in scopes(ctx):
        if scope == "rail":
            continue
        P = comm.size()
        out = cp.empty((P, m, m), cp.float16)
        modes = {"compute": dict(communicate=False), "comm": dict(compute=False),
                 "overlap": {}}
        times = {}
        for mode, kw in modes.items():
            ring_matmul(comm, a, b0, bufs, out, s_comp, s_comm, **kw)  # warm-up
            reps = []
            for _ in range(args.mm_reps):
                ctx.barrier()
                t0 = time.perf_counter()
                ring_matmul(comm, a, b0, bufs, out, s_comp, s_comm, **kw)
                reps.append(time.perf_counter() - t0)
            times[mode] = float(np.median(reps))
        err = ring_checksum_error(comm, a, b0, out)
        del out
        free_pool()
        per = ctx.gather([times["compute"], times["comm"], times["overlap"], err])
        if ctx.rank:
            continue
        groups = {}
        for r in range(ctx.world):
            groups.setdefault(group_of(ctx, scope, r), []).append(r)
        for g, ranks in groups.items():
            tc, tm, to = (per[ranks, i].max() for i in range(3))
            err_g = per[ranks, 3].max()
            total = P * P * 2.0 * m ** 3 / to / 1e12
            ring_bw = (P - 1) * m * m * 2 / tm / 1e9
            ok = err_g < 1e-3
            ctx.record(test="matmul", scope=scope, group=labels[g], gpus=P, block=m,
                       tflops_total=total, compute_s=tc, comm_s=tm, overlap_s=to,
                       ring_gbs=ring_bw, rel_err=err_g, ok=ok)
            name = scope if len(groups) == 1 else labels[g]
            ctx.log(f"  {name:<8}{P:>5}{total:14.1f}{total / P:9.1f}{tc:11.3f}{tm:9.3f}"
                    f"{to:11.3f}{tc / to:12.0%} {ring_bw:11.1f}  "
                    f"{'ok' if ok else 'FAILED'} (rel err {err_g:.1e})")


# --- burn --------------------------------------------------------------------------

def test_burn(ctx, args):
    n, W = args.burn_n, ctx.world
    a, b = random_matrix((n, n), np.float16), random_matrix((n, n), np.float16)
    c, ref = cp.empty((n, n), cp.float16), cp.empty((n, n), cp.float16)
    s_comp = cp.cuda.Stream(non_blocking=True)
    s_comm = cp.cuda.Stream(non_blocking=True)
    count = (args.burn_comm_mib << 20) // 4
    send, recv = cp.ones(count, cp.float32), cp.empty(count, cp.float32)
    ctx.device.synchronize()  # inputs come from the default stream, used on s_comp/s_comm
    with s_comp:
        cp.matmul(a, b, out=ref)

    def allreduce():
        ctx.comm.allReduce(send.data.ptr, recv.data.ptr, count, nccl.NCCL_FLOAT32,
                           nccl.NCCL_SUM, s_comm.ptr)

    # Same number of all-reduces per window on every rank, sized from a calibration.
    for _ in range(3):
        allreduce()
    s_comm.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        allreduce()
    s_comm.synchronize()
    t_ar = float(ctx.gather([(time.perf_counter() - t0) / 5]).max())
    ops = max(1, int(args.burn_comm_load * args.burn_window / t_ar))
    windows = max(1, int(round(args.burn_seconds / args.burn_window)))
    ctx.log(f"\n== Burn: fp16 GEMM n={n} on every GPU + global all-reduce "
            f"({args.burn_comm_mib} MiB x {ops} per window), {windows} x {args.burn_window:g} s")
    ctx.log(f"  {'t (s)':>6}{'TFLOPS min':>12}{'median':>9}  {'slowest GPU':<18}"
            f"{'allreduce GB/s':>15}  GEMM check")
    flops = 2.0 * n ** 3
    e_c0, e_c1, e_m0, e_m1 = (cp.cuda.Event() for _ in range(4))
    history = []
    for w in range(windows):
        e_m0.record(s_comm)
        for _ in range(ops):
            allreduce()
        e_m1.record(s_comm)
        e_c0.record(s_comp)
        k, prev = 0, None
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < args.burn_window:
            with s_comp:
                for _ in range(args.burn_batch):
                    cp.matmul(a, b, out=c)
            ev = cp.cuda.Event()
            ev.record(s_comp)
            k += args.burn_batch
            if prev is not None:
                prev.synchronize()  # keep one batch queued so the GPU never idles
            prev = ev
        e_c1.record(s_comp)
        with s_comp:
            cp.matmul(a, b, out=c)
            ok = bool(cp.array_equal(c, ref))
            diff = 0.0 if ok else float(cp.abs(c.astype(cp.float32) - ref.astype(cp.float32)).max())
        s_comm.synchronize()
        tflops = flops * k / (cp.cuda.get_elapsed_time(e_c0, e_c1) / 1e3) / 1e12
        busbw = (ops * 4 * count * 2 * (W - 1) / W
                 / (cp.cuda.get_elapsed_time(e_m0, e_m1) / 1e3) / 1e9)
        per = ctx.gather([tflops, ok, busbw, diff])
        history.append(per[:, 0])
        if ctx.rank:
            continue
        for r in range(W):
            ctx.record(test="burn", window=w, gpu=ctx.labels[r], tflops=per[r, 0],
                       allreduce_busbw_gbs=per[r, 2], check_ok=bool(per[r, 1]),
                       max_abs_diff=per[r, 3])
        slow = int(per[:, 0].argmin())
        failed = [f"{ctx.labels[r]} (max diff {per[r, 3]:g})" for r in range(W) if not per[r, 1]]
        ctx.log(f"  {(w + 1) * args.burn_window:6.0f}{per[:, 0].min():12.1f}"
                f"{np.median(per[:, 0]):9.1f}  {ctx.labels[slow]:<18}{per[:, 2].min():15.1f}  "
                f"{'MISMATCH ' + ', '.join(failed) if failed else 'ok'}")
    if ctx.rank == 0 and len(history) > 1:
        h = np.array(history)  # windows x GPUs
        drop = 1 - h[-1] / h[0]
        ctx.log(f"  per-GPU change first -> last window: median {-np.median(drop):+.1%}, "
                f"worst {ctx.labels[int(drop.argmax())]} {-drop.max():+.1%}")
    del a, b, c, ref, send, recv
    free_pool()


# --- node logging ------------------------------------------------------------------

def start_node_logging(ctx):
    """Local rank 0: node info file and an nvidia-smi telemetry process."""
    if ctx.local != 0:
        return None
    with open(os.path.join(ctx.out, f"node_{ctx.host}.txt"), "w") as f:
        for cmd in (["nvidia-smi", "--format=csv", "--query-gpu=index,name,serial,pci.bus_id,"
                     "driver_version,power.limit,clocks.max.sm,clocks.max.mem"],
                    ["nvidia-smi", "topo", "-m"], ["ibstat"]):
            f.write("$ " + " ".join(cmd) + "\n")
            try:
                f.write(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
            except (OSError, subprocess.SubprocessError) as exc:
                f.write(f"failed: {exc}\n")
    fields = ("timestamp,index,temperature.gpu,temperature.memory,power.draw,clocks.sm,"
              "clocks.mem,utilization.gpu,memory.used")
    try:
        help_text = subprocess.run(["nvidia-smi", "--help-query-gpu"], capture_output=True,
                                   text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        help_text = ""
    fields += (",clocks_event_reasons.active" if "clocks_event_reasons.active" in help_text
               else ",clocks_throttle_reasons.active")
    log = open(os.path.join(ctx.out, f"smi_{ctx.host}.csv"), "w")
    return subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,nounits",
                             "-l", "2"], stdout=log, stderr=subprocess.STDOUT)


TESTS = {"gemm": test_gemm, "hbm": test_hbm, "memtest": test_memtest,
         "collectives": test_collectives, "matmul": test_matmul, "burn": test_burn}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="run directory (shared filesystem)")
    p.add_argument("--tests", nargs="+", choices=list(TESTS), default=list(TESTS))
    p.add_argument("--quick", action="store_true", help="short smoke-test settings")
    p.add_argument("--gemm-n", type=int, default=16384)
    p.add_argument("--gemm-types", nargs="+", choices=list(GEMM_TYPES), default=list(GEMM_TYPES))
    p.add_argument("--gemm-seconds", type=float, default=10.0,
                   help="timing per GEMM type (and for the HBM copy)")
    p.add_argument("--hbm-gib", type=int, default=8)
    p.add_argument("--mem-frac", type=float, default=0.9)
    p.add_argument("--mem-passes", type=int, default=8)
    p.add_argument("--coll-sizes", nargs="+", default=["8K", "1M", "8M", "64M", "512M", "2G"])
    p.add_argument("--mm-n", type=int, default=16384)
    p.add_argument("--mm-reps", type=int, default=5)
    p.add_argument("--burn-seconds", type=float, default=600.0)
    p.add_argument("--burn-window", type=float, default=30.0)
    p.add_argument("--burn-n", type=int, default=16384)
    p.add_argument("--burn-batch", type=int, default=4)
    p.add_argument("--burn-comm-mib", type=int, default=256)
    p.add_argument("--burn-comm-load", type=float, default=0.25,
                   help="fraction of each window the all-reduces would take alone")
    args = p.parse_args()
    if args.quick:
        args.gemm_seconds, args.mem_passes, args.mm_reps = 1.0, 2, 1
        args.coll_sizes = ["8K", "64M", "512M"]
        args.burn_seconds, args.burn_window = 60.0, 15.0

    os.makedirs(args.out, exist_ok=True)
    ctx = Ctx(args.out)
    smi = start_node_logging(ctx)
    try:
        d = ctx.info[0]
        ctx.log(f"{ctx.world} GPUs ({d['name']}) on {ctx.nnodes} nodes: "
                f"{' '.join(sorted({i['host'] for i in ctx.info}))}")
        ctx.log(f"NCCL {nccl.get_version()}, CuPy {cp.__version__}, "
                f"CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}")
        for name in args.tests:
            t0 = time.perf_counter()
            TESTS[name](ctx, args)
            ctx.barrier()
            ctx.log(f"  [{name}: {time.perf_counter() - t0:.0f} s]")
    finally:
        if smi is not None:
            smi.terminate()
    ctx.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Peers would wait forever in NCCL; exit hard so srun kills the whole step.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
