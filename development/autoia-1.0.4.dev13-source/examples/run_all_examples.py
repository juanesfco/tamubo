"""Run the complete numbered AutoIA tutorial sequence in fresh subprocesses."""
from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = sorted(HERE.glob("[0-9][0-9]_*.py"))


if __name__ == "__main__":
    failed = []
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    for path in SCRIPTS:
        print("\n" + "=" * 78, flush=True)
        print(path.name, flush=True)
        print("=" * 78, flush=True)
        proc = subprocess.run([sys.executable, str(path)], env=env)
        if proc.returncode:
            failed.append(path.name)
    if failed:
        print("FAILED:", failed)
        raise SystemExit(1)
    print(f"\nAll {len(SCRIPTS)} tutorial examples completed successfully.")
