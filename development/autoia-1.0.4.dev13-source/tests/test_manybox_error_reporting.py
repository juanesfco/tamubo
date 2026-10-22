"""Per-box fail-closed reporting for prepared many-box execution."""

from __future__ import annotations

import importlib.util
import numpy as np
import pytest
from scipy.optimize import linprog

from autoia import (
    Interval,
    prepare,
    ManyBoxCertificationReport,
    ManyBoxBoundsItemReport,
    CertificationError,
)


def _mixed_feasibility_model(p):
    x = p[0]
    # min z subject to z >= x and 0 <= z <= 1.
    # Feasible at x=.3,.7; infeasible at x=5.
    return linprog(
        c=[1.0],
        A_ub=[[-1.0]],
        b_ub=[-x],
        bounds=[(0.0, 1.0)],
    ).fun


def _boxes():
    return [
        [Interval(0.3, 0.3)],
        [Interval(5.0, 5.0)],
        [Interval(0.7, 0.7)],
    ]


def test_default_manybox_behavior_still_raises():
    s = prepare(_mixed_feasibility_model, order=4, cache_max_entries=0)
    with pytest.raises(CertificationError):
        s.enclose_many_bounds(_boxes(), workers=1, parallel="serial")


def test_bounds_report_mode_preserves_successful_siblings():
    s = prepare(_mixed_feasibility_model, order=4, cache_max_entries=0)
    lo, hi, ok = s.enclose_many_bounds(
        _boxes(), workers=1, parallel="serial", on_error="report"
    )
    assert ok.tolist() == [True, False, True]
    assert lo.shape == hi.shape == (3, 1)
    assert lo[0, 0] <= 0.3 <= hi[0, 0]
    assert np.isnan(lo[1, 0]) and np.isnan(hi[1, 0])
    assert lo[2, 0] <= 0.7 <= hi[2, 0]


def test_detailed_report_carries_failure_code():
    s = prepare(_mixed_feasibility_model, order=4, cache_max_entries=0)
    items = s.certify_many(_boxes(), on_error="report")
    assert all(isinstance(x, ManyBoxCertificationReport) for x in items)
    assert [x.ok for x in items] == [True, False, True]
    assert items[1].failure.error_type == "CertificationError"
    assert items[1].failure.error_code == "certification-failed"
    assert items[1].result is None


def test_iter_bounds_report_continues_past_failed_box():
    s = prepare(_mixed_feasibility_model, order=4, cache_max_entries=0)
    items = list(s.iter_bounds_many(_boxes(), on_error="report"))
    assert all(isinstance(x, ManyBoxBoundsItemReport) for x in items)
    assert [x.index for x in items] == [0, 1, 2]
    assert [x.ok for x in items] == [True, False, True]
    assert items[1].failure.error_code == "certification-failed"
    assert items[2].lower[0] <= 0.7 <= items[2].upper[0]


@pytest.mark.skipif(importlib.util.find_spec("cloudpickle") is None,
                    reason="closure/process transport uses optional cloudpickle")
def test_persistent_process_report_mode_does_not_lose_siblings():
    # Deliberately capture a local value so this exercises cloudpickle too.
    cap = 1.0
    def model(p):
        x = p[0]
        return linprog(
            c=[cap], A_ub=[[-1.0]], b_ub=[-x], bounds=[(0.0, cap)]
        ).fun

    s = prepare(model, order=4, cache_max_entries=0)
    with s.manybox_executor(
        workers=2, parallel="process", chunk_size=3,
        use_cache=False, on_error="report",
    ) as ex:
        lo, hi, ok, info = ex.enclose_many_bounds(_boxes(), return_info=True)
    assert ok.tolist() == [True, False, True]
    assert lo[0, 0] <= 0.3 <= hi[0, 0]
    assert np.isnan(lo[1, 0]) and np.isnan(hi[1, 0])
    assert lo[2, 0] <= 0.7 <= hi[2, 0]
    assert info.n_succeeded == 2
    assert info.n_failed == 1
    assert info.resolved_parallel == "process"
