"""Audit a release tree for a prohibited downstream package token.

The target token is assembled from fragments so this source file does not itself
contain the prohibited byte sequence.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

TARGET = ("cer" + "topt").encode("ascii")
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".tex", ".toml", ".cfg", ".ini", ".json", ".rst", ".csv",
}


def _contains(data: bytes) -> bool:
    return TARGET in data.lower()


def scan_tree(root: Path) -> list[str]:
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _contains(rel.encode("utf-8", errors="ignore")):
            hits.append(rel + " [name]")
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            try:
                if _contains(path.read_bytes()):
                    hits.append(rel)
            except OSError:
                hits.append(rel + " [unreadable]")
    return hits


def scan_zip(path: Path) -> list[str]:
    hits: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if _contains(info.filename.encode("utf-8", errors="ignore")):
                hits.append(info.filename + " [name]")
                continue
            if Path(info.filename).suffix.lower() in TEXT_EXTENSIONS:
                if _contains(zf.read(info)):
                    hits.append(info.filename)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ns = ap.parse_args()
    total: list[tuple[Path, str]] = []
    for path in ns.paths:
        hits = scan_zip(path) if path.suffix.lower() == ".zip" else scan_tree(path)
        total.extend((path, hit) for hit in hits)
    if total:
        for base, hit in total:
            print(f"FAIL: {base}: {hit}")
        return 1
    print(f"PASS: prohibited-token occurrences = 0 across {len(ns.paths)} input(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
