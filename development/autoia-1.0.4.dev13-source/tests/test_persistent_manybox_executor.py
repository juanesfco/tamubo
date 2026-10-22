import importlib.util
import numpy as np
import pytest

from autoia import Interval, prepare


def _smooth_model(p):
    x, y = p
    return x * x + 0.25 * x * y + y


def _boxes():
    return [
        [Interval(-0.8, -0.5), Interval(-0.2, 0.1)],
        [Interval(-0.4, -0.1), Interval(0.0, 0.3)],
        [Interval(0.0, 0.25), Interval(-0.1, 0.2)],
        [Interval(0.2, 0.5), Interval(0.1, 0.4)],
        [Interval(0.45, 0.7), Interval(-0.3, 0.0)],
        [Interval(0.65, 0.9), Interval(0.0, 0.25)],
    ]


def _serial_reference(boxes):
    s = prepare(_smooth_model, order=4, cache_max_entries=0)
    return s.enclose_many_bounds(boxes, workers=1, parallel="serial", use_cache=False)


def test_persistent_manybox_executor_preserves_successive_batch_results():
    boxes = _boxes()
    ref_lo, ref_hi = _serial_reference(boxes)
    s = prepare(_smooth_model, order=4, cache_max_entries=0)
    with s.manybox_executor(workers=1, parallel="serial", chunk_size=2, use_cache=False) as ex:
        lo1, hi1 = ex.enclose_many_bounds(boxes[:3])
        lo2, hi2 = ex.enclose_many_bounds(boxes[3:])
        assert ex.calls == 2
        assert ex.boxes_certified == 6
    assert np.array_equal(np.vstack([lo1, lo2]), ref_lo)
    assert np.array_equal(np.vstack([hi1, hi2]), ref_hi)


@pytest.mark.skipif(importlib.util.find_spec("cloudpickle") is None,
                    reason="spawn process transport uses optional cloudpickle")
def test_persistent_process_executor_matches_serial_across_two_batches():
    boxes = _boxes()
    ref_lo, ref_hi = _serial_reference(boxes)
    s = prepare(_smooth_model, order=4, cache_max_entries=0)
    with s.manybox_executor(workers=2, parallel="process", chunk_size=1, use_cache=False) as ex:
        lo1, hi1, info1 = ex.enclose_many_bounds(boxes[:3], return_info=True)
        lo2, hi2, info2 = ex.enclose_many_bounds(boxes[3:], return_info=True)
        assert info1.resolved_parallel == "process"
        assert info2.resolved_parallel == "process"
        assert info1.scheduling_note == "persistent-manybox-executor"
        assert info2.scheduling_note == "persistent-manybox-executor"
    assert np.array_equal(np.vstack([lo1, lo2]), ref_lo)
    assert np.array_equal(np.vstack([hi1, hi2]), ref_hi)
