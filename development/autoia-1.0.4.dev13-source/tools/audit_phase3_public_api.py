"""Step 20a Phase-3 public API freeze audit.

This audit checks public routing and reproducibility semantics only.  It does not
replace the trusted arithmetic audits.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from autoia import (
    Interval, autoia, prepare, ConfigurationPolicy,
    AutoIAConfiguredResult, PreparedAutoIA, PreparedConfiguredAutoIA, __version__,
)


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def main():
    checks = 0

    models = [
        (lambda p: p[0]*p[0] + 0.2*p[0], [Interval(-0.4, 0.7)]),
        (lambda p: np.exp(0.6*p[0]), [Interval(-0.5, 0.8)]),
        (lambda p: np.exp(3.0*p[0]), [Interval(-1.0, 1.0)]),
    ]

    for model, box in models:
        novice = autoia(model, box)
        explicit_auto = autoia(
            model, box, configuration_policy=ConfigurationPolicy(mode="auto")
        )
        assert isinstance(novice, AutoIAConfiguredResult)
        assert novice.configuration_policy.mode == "auto"
        assert novice.configuration_diagnostics["baseline_order"] == 4
        assert novice.configuration_diagnostics["baseline_representation"] == "polynomial"
        assert np.array_equal(bits(novice.lower), bits(explicit_auto.lower))
        assert np.array_equal(bits(novice.upper), bits(explicit_auto.upper))
        assert "AutoIAResultV11" not in repr(novice)
        assert "bern_cost" not in repr(novice)
        checks += 8

        fixed_keyword = autoia(model, box, order=4)
        fixed_policy = autoia(
            model, box, order=4,
            configuration_policy=ConfigurationPolicy(mode="fixed"),
        )
        assert not isinstance(fixed_keyword, AutoIAConfiguredResult)
        assert np.array_equal(bits(fixed_keyword.lower), bits(fixed_policy.lower))
        assert np.array_equal(bits(fixed_keyword.upper), bits(fixed_policy.upper))
        assert "AutoIAResultV11" not in repr(fixed_keyword)
        assert "bern_cost" not in repr(fixed_keyword)
        checks += 5

    # Explicit representation is also a fixed expert request.
    fixed_gen = autoia(
        lambda p: p[0]*p[0], [Interval(-1.0, 1.0)], representation="generator"
    )
    assert not isinstance(fixed_gen, AutoIAConfiguredResult)
    assert fixed_gen.requested_representation == "generator"
    checks += 2

    # Prepared routing follows the same novice/expert rule.
    def prepared_model(p):
        return np.exp(3.0*p[0])

    novice_session = prepare(prepared_model)
    fixed_session = prepare(prepared_model, order=4)
    assert isinstance(novice_session, PreparedConfiguredAutoIA)
    assert isinstance(fixed_session, PreparedAutoIA)
    box = [Interval(-0.6, 0.6)]
    a = novice_session.certify(box)
    b = autoia(prepared_model, box)
    assert np.array_equal(bits(a.lower), bits(b.lower))
    assert np.array_equal(bits(a.upper), bits(b.upper))
    assert "Attempt 1" in a.explain_configuration()
    checks += 5

    out = {
        "autoia_version": __version__,
        "checks": checks,
        "status": "PASS",
        "contract": "zero-knob automatic; explicit order/representation fixed unless policy explicit",
    }
    Path("STEP20A_PHASE3_PUBLIC_API_AUDIT.json").write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
