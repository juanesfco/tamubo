"""24 — Compact and streaming many-box workflows.

AutoIA's many-box layer is a general repeated-enclosure capability, not a
global optimizer.  This example demonstrates three equivalent ways to certify
many independent boxes:

1. top-level ``enclose_many`` returning full result objects;
2. prepared ``enclose_many_bounds`` returning compact stacked arrays;
3. prepared ``iter_bounds_many`` streaming boxes in bounded chunks.

The first route uses a process pool; the compact and streaming routes run
serially so their exact agreement also demonstrates that scheduling does not
change the proof result. Every returned enclosure is produced by the trusted
CPU binary64 backend.
"""
import numpy as np
from autoia import Interval, enclose_many, prepare


def engineering_model(p):
    x, y = p
    return np.array([
        np.exp(-x*x) + x*y,
        np.sin(y) + 0.25*x*x,
    ], dtype=object)


def make_boxes():
    for a in np.linspace(-1.0, 0.75, 8):
        yield [Interval(float(a), float(a + 0.25)), Interval(-0.15, 0.20)]


if __name__ == "__main__":
    boxes = list(make_boxes())

    # One-shot public API.
    full, info_full = enclose_many(
        engineering_model, boxes,
        workers=2, parallel="process", return_info=True,
    )
    print("one-shot execution:", info_full)
    print("first full enclosure:", full[0].lower, full[0].upper)

    # Prepare once, then reuse the prepared model and return only compact bounds.
    session = prepare(engineering_model)
    lower, upper, info_compact = session.enclose_many_bounds(
        boxes, workers=1, parallel="serial", return_info=True,
        use_cache=False,
    )
    print("\ncompact execution:", info_compact)
    print("lower array shape:", lower.shape)
    print("upper array shape:", upper.shape)

    # Stream boxes without retaining all full AutoIAResult objects.
    streamed = list(session.iter_bounds_many(
        make_boxes(), workers=1, parallel="serial",
        chunk_size=2, use_cache=False,
    ))
    stream_lower = np.vstack([lo for lo, _ in streamed])
    stream_upper = np.vstack([hi for _, hi in streamed])

    # All three routes are independent executions of the same rigorous semantics.
    full_lower = np.vstack([r.lower for r in full])
    full_upper = np.vstack([r.upper for r in full])
    assert np.array_equal(lower, full_lower)
    assert np.array_equal(upper, full_upper)
    assert np.array_equal(stream_lower, full_lower)
    assert np.array_equal(stream_upper, full_upper)
    assert info_full.machine_rigorous and info_compact.machine_rigorous

    print("streamed boxes:", len(streamed))
    print("all three many-box routes agree exactly: True")
    print("machine rigorous: True")
