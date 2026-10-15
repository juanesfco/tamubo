"""Summarise a stress-test run directory: InfiniBand links, NCCL transports, GPU telemetry.

    python summarize.py logs/<jobid>
"""

import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# nvidia-smi clocks_event_reasons bits that mean the GPU was slowed down.
SLOWDOWN = {0x4: "sw power cap", 0x8: "hw slowdown", 0x20: "sw thermal",
            0x40: "hw thermal", 0x80: "hw power brake"}


def infiniband(run_dir):
    print("== InfiniBand ports per node (ibstat)")
    for path in sorted(glob.glob(os.path.join(run_dir, "node_*.txt"))):
        host = os.path.basename(path)[5:-4]
        text = open(path).read().split("$ ibstat", 1)[-1]
        ports = Counter()
        for block in re.split(r"\nCA '", text)[1:]:
            get = lambda key: (re.search(rf"{key}:\s*(.+)", block) or [None, "?"])[1].strip()
            ports[(get("Link layer"), get("Rate"), get("State"))] += 1
        print(f"  {host}: " + ", ".join(f"{k}x {layer} {rate} Gb/s {state}"
                                        for (layer, rate, state), k in sorted(ports.items())))


def nccl_transports(run_dir):
    files = glob.glob(os.path.join(run_dir, "nccl.*.log"))
    if not files:
        return
    via, nics, nets = Counter(), Counter(), set()
    for path in files:
        host = os.path.basename(path).split(".")[1]
        for line in open(path, errors="replace"):
            msg = line.split("NCCL INFO ", 1)[-1].strip()
            m = re.search(r" via (\S+)", msg)
            if m:
                kind = m.group(1)
                via[re.sub(r"/\d+", "", kind)] += 1  # drop NIC numbers
                n = re.match(r"NET/\w+/(\d+)", kind)
                if n:
                    nics[(host, int(n.group(1)))] += 1
            if msg.startswith(("NET/IB : Using", "Using network", "NCCL version")):
                nets.add(msg)
    print("\n== NCCL (from NCCL_DEBUG logs)")
    names = {}
    for msg in sorted(nets):
        print(f"  {msg}")
        names.update({int(i): n for i, n in re.findall(r"\[(\d+)\](mlx5_\d+)", msg)})
    for kind, k in via.most_common():
        print(f"  connections via {kind}: {k}")
    if nics:
        used = Counter()
        for (host, dev), k in nics.items():
            used[names.get(dev, f"dev{dev}")] += k
        print("  NIC use (connections, all nodes): "
              + ", ".join(f"{n} {k}" for n, k in sorted(used.items())))


def telemetry(run_dir):
    rows = defaultdict(list)  # (host, gpu) -> rows while busy
    for path in sorted(glob.glob(os.path.join(run_dir, "smi_*.csv"))):
        host = os.path.basename(path)[4:-4]
        with open(path) as f:
            reader = csv.reader(f, skipinitialspace=True)
            header = [re.sub(r"\s*\[.*\]", "", h) for h in next(reader, [])]
            for rec in reader:
                if len(rec) != len(header) or rec[0] == header[0]:
                    continue
                d = dict(zip(header, rec))
                try:
                    if float(d["utilization.gpu"]) < 90:
                        continue
                    reason = d.get("clocks_event_reasons.active",
                                   d.get("clocks_throttle_reasons.active", "0"))
                    sm = d.get("clocks.sm", d.get("clocks.current.sm"))
                    row = (float(d["temperature.gpu"]), _num(d["temperature.memory"]),
                           float(d["power.draw"]), float(sm), int(reason, 16))
                except (KeyError, TypeError, ValueError):
                    continue
                rows[(host, int(d["index"]))].append(row)
    if not rows:
        print("\n== GPU telemetry: no busy samples found")
        return
    print("\n== GPU telemetry while busy (utilization >= 90%, 2 s samples)")
    print(f"  {'GPU':<16}{'samples':>8}{'max C':>7}{'mem C':>7}{'max W':>7}"
          f"{'SM MHz med':>11}{'min':>6}  slowdown reasons (% of samples)")
    med_all = np.median([np.median([r[3] for r in v]) for v in rows.values()])
    for (host, gpu), v in sorted(rows.items()):
        a = np.array([r[:4] for r in v], dtype=float)
        reasons = [r[4] for r in v]
        seen = ", ".join(f"{name} {100 * sum(bool(x & bit) for x in reasons) / len(v):.0f}%"
                         for bit, name in SLOWDOWN.items() if any(x & bit for x in reasons))
        med = np.median(a[:, 3])
        flag = "  <- slow clock" if med < 0.95 * med_all else ""
        print(f"  {host + ':gpu' + str(gpu):<16}{len(v):>8}{a[:, 0].max():7.0f}"
              f"{np.nanmax(a[:, 1]):7.0f}{a[:, 2].max():7.0f}{med:11.0f}{a[:, 3].min():6.0f}"
              f"  {seen or 'none'}{flag}")


def _num(text):
    try:
        return float(text)
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    run = sys.argv[1]
    infiniband(run)
    nccl_transports(run)
    telemetry(run)
