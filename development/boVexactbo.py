# Run with (example): python boVexactbo.py 21 51 101
print("Starting")

import os
import re
import sys
import time
import subprocess
import numpy as np

print("Libraries loaded")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _extract_floats(text):
    return np.array(
        [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)],
        dtype=float,
    )

def _parse_final_xy(output, dim):
    match_x = re.search(r"Final data points \(X\):\s*\n([\s\S]*?)\nFinal data points \(y\):", output)
    if not match_x:
        raise ValueError("Could not parse X from output.")
    x_block = match_x.group(1)
    y_block = output.split("Final data points (y):", 1)[-1]

    x_vals = _extract_floats(x_block)
    if x_vals.size % dim != 0:
        raise ValueError("Parsed X values do not align with expected dimension.")
    X = x_vals.reshape(-1, dim)

    y_vals = _extract_floats(y_block)
    return X, y_vals

def _parse_true_minimum(output):
    match_point = re.search(r"Best point:\s*\[([^\]]+)\]", output)
    match_val = re.search(r"Best value:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if not match_point or not match_val:
        raise ValueError("Could not parse true minimum output.")
    point = _extract_floats(match_point.group(1))
    value = float(match_val.group(1))
    return point, value

def _run_command(cmd, cwd, env=None):
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout, elapsed

def _best_from_data(X, y):
    idx = np.argmin(y)
    return X[idx], y[idx]

def _format_row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

def main():
    # Resolutions for BO/ExactBO
    if len(sys.argv) > 1:
        resolutions = [int(x) for x in sys.argv[1:]]
    else:
        resolutions = [21, 51, 101]

    dim = 2

    # True minimum (high-resolution grid)
    true_n = 10001
    print(f"Computing true minimum using true_minimum.py on {true_n}x{true_n} grid...")
    true_cmd = [sys.executable, "true_minimum.py", str(true_n)]
    true_out, true_time = _run_command(true_cmd, cwd=SCRIPT_DIR)
    x_true, y_true = _parse_true_minimum(true_out)
    print(f"True minimum: x={x_true}, y={y_true}, time={true_time:.2f}s")

    results = []

    # ExactBO via legate
    env = os.environ.copy()
    env["CUPYNUMERIC_REPORT_COVERAGE"] = "1"
    for N in resolutions:
        print(f"Running ExactBO (legate) with N={N}...")
        exact_cmd = [
            "legate",
            "--cpus", "1",
            "--gpus", "1",
            "--show-config",
            "run_exactbo_cupynumeric.py",
            str(N),
        ]
        out, t_exact = _run_command(exact_cmd, cwd=SCRIPT_DIR, env=env)
        X_data, y_data = _parse_final_xy(out, dim)
        x_best, y_best = _best_from_data(X_data, y_data)
        results.append({
            "method": "ExactBO",
            "N": N,
            "x_best": x_best,
            "y_best": y_best,
            "y_gap": y_best - y_true,
            "x_dist": float(np.linalg.norm(x_best - x_true)),
            "time_s": t_exact,
        })

    # Regular BO via run_bo.py
    for N in resolutions:
        print(f"Running BO (numpy) with N={N}...")
        bo_cmd = [sys.executable, "run_bo.py", str(N)]
        out, t_bo = _run_command(bo_cmd, cwd=SCRIPT_DIR)
        X_data, y_data = _parse_final_xy(out, dim)
        x_best, y_best = _best_from_data(X_data, y_data)
        results.append({
            "method": "BO",
            "N": N,
            "x_best": x_best,
            "y_best": y_best,
            "y_gap": y_best - y_true,
            "x_dist": float(np.linalg.norm(x_best - x_true)),
            "time_s": t_bo,
        })

    # Print results table
    widths = [8, 6, 20, 14, 12, 12, 10]
    headers = ["Method", "N", "x_best", "y_best", "y_gap", "x_dist", "time_s"]
    print("\nResults")
    print(_format_row(headers, widths))
    print(_format_row(["-" * w for w in widths], widths))
    for r in results:
        cols = [
            r["method"],
            r["N"],
            np.array2string(r["x_best"], precision=5, separator=","),
            f"{r['y_best']:.6f}",
            f"{r['y_gap']:.6f}",
            f"{r['x_dist']:.6f}",
            f"{r['time_s']:.2f}",
        ]
        print(_format_row(cols, widths))

if __name__ == "__main__":
    main()
