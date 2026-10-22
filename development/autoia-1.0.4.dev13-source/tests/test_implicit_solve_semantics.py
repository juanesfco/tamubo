import numpy as np
import pytest

scipy_opt = pytest.importorskip("scipy.optimize")
from scipy.optimize import root_scalar, fsolve

from autoia import autoia, Interval
from autoia.exceptions import UnsupportedOperationError, CertificationError


def colebrook_resid(f, Re, rel_roughness):
    if f <= 0:
        return np.nan
    term = (rel_roughness / 3.7) + (2.51 / (Re * np.sqrt(f)))
    return 1.0 / np.sqrt(f) + 2.0 * np.log10(term)


def colebrook_model(p):
    result = root_scalar(
        colebrook_resid,
        bracket=[0.0008, 0.1],
        args=(p[0], p[1]),
    )
    return result.root


def test_root_scalar_colebrook_certifies_and_contains_grid():
    result = autoia(
        colebrook_model,
        [Interval(90000.0, 110000.0), Interval(0.0008, 0.0012)],
    )
    assert result.machine_rigorous
    assert result.lower[0] < result.upper[0]
    certs = [c for c in result.certificates if c.get("operation") == "scipy.optimize.root_scalar"]
    assert certs
    cert = certs[-1]
    assert cert["unique_in_bracket"] is True
    assert cert["rho_upper"] < 1.0
    assert cert["root_range"].lo >= 0.0008
    assert cert["root_range"].hi <= 0.1

    for Re in np.linspace(90000.0, 110000.0, 9):
        for rr in np.linspace(0.0008, 0.0012, 9):
            nominal = root_scalar(
                colebrook_resid, bracket=[0.0008, 0.1], args=(float(Re), float(rr))
            ).root
            assert result.lower[0] <= nominal <= result.upper[0]


def sqrt_resid(y, a):
    return y * y - a


def sqrt_model_with_downstream_residual(p):
    y = root_scalar(sqrt_resid, args=(p[0],), bracket=[0.5, 2.0]).root
    return np.array([y, y * y - p[0]])


def test_root_scalar_preserves_dependency_downstream():
    result = autoia(sqrt_model_with_downstream_residual, [Interval(1.0, 1.21)])
    assert result.machine_rigorous
    assert result.lower[0] <= 1.0
    assert result.upper[0] >= 1.1
    assert result.lower[1] <= 0.0 <= result.upper[1]
    # A dependency-blind root interval squared minus the input would be much
    # wider.  This regression checks that the implicit predictor survives into
    # downstream arithmetic.
    assert result.width[1] < 0.02


def sqrt_model_positional_args(p):
    return root_scalar(sqrt_resid, (p[0],), None, [0.5, 2.0]).root


def test_root_scalar_common_positional_signature_is_supported():
    result = autoia(sqrt_model_positional_args, [Interval(1.0, 1.21)])
    assert result.machine_rigorous
    assert result.lower[0] <= 1.0
    assert result.upper[0] >= 1.1


def root_without_bracket_model(p):
    return root_scalar(sqrt_resid, args=(p[0],), x0=1.0, x1=1.2).root


def test_validated_root_scalar_without_bracket_fails_explicitly():
    with pytest.raises(UnsupportedOperationError):
        autoia(root_without_bracket_model, [Interval(1.0, 1.21)])


def bad_bracket_model(p):
    return root_scalar(sqrt_resid, args=(p[0],), bracket=[1.5, 2.0]).root


def test_root_scalar_unprovable_bracket_is_certification_failure():
    with pytest.raises(CertificationError):
        autoia(bad_bracket_model, [Interval(1.0, 1.21)])


def nonlinear_system(y, a, b):
    return np.array([
        y[0] + 0.2 * y[1] - a,
        0.1 * y[0] + y[1] + 0.05 * y[0] * y[1] - b,
    ])


def fsolve_model(p):
    y = fsolve(nonlinear_system, [1.0, 1.5], args=(p[0], p[1]))
    return np.array([y[0], y[1], y[0] + p[0] * y[1]])


def test_fsolve_square_system_certifies_and_contains_grid():
    result = autoia(
        fsolve_model,
        [Interval(0.9, 1.1), Interval(1.4, 1.6)],
    )
    assert result.machine_rigorous
    certs = [c for c in result.certificates if c.get("operation") == "scipy.optimize.fsolve"]
    assert certs
    assert certs[-1]["rho_upper"] < 1.0

    for a in np.linspace(0.9, 1.1, 7):
        for b in np.linspace(1.4, 1.6, 7):
            y = fsolve(nonlinear_system, [1.0, 1.5], args=(float(a), float(b)))
            vals = np.array([y[0], y[1], y[0] + a * y[1]])
            assert np.all(result.lower <= vals)
            assert np.all(vals <= result.upper)


def fsolve_residual_model(p):
    y = fsolve(nonlinear_system, [1.0, 1.5], (p[0], p[1]))
    return nonlinear_system(y, p[0], p[1])


def test_fsolve_preserves_dependency_into_residual_and_positional_args():
    result = autoia(
        fsolve_residual_model,
        [Interval(0.9, 1.1), Interval(1.4, 1.6)],
    )
    assert result.machine_rigorous
    assert np.all(result.lower <= 0.0)
    assert np.all(result.upper >= 0.0)
    assert np.max(result.width) < 0.01


def fsolve_full_output_model(p):
    y, info, ier, mesg = fsolve(
        nonlinear_system, [1.0, 1.5], args=(p[0], p[1]), full_output=True
    )
    assert ier == 1
    assert "autoia_certificate" in info
    return y


def test_fsolve_full_output_preserves_common_scipy_return_shape():
    result = autoia(
        fsolve_full_output_model,
        [Interval(0.95, 1.05), Interval(1.45, 1.55)],
    )
    assert result.machine_rigorous
    assert result.output_shape == (2,)


def singular_system(y, a):
    return np.array([y[0] * y[0] - a])


def singular_fsolve_model(p):
    # The intended root is near zero and the Jacobian is singular at a=0.
    return fsolve(singular_system, [0.0], args=(p[0],))[0]


def test_fsolve_singular_case_never_returns_false_certificate():
    with pytest.raises((CertificationError, Exception)) as excinfo:
        autoia(singular_fsolve_model, [Interval(0.0, 1e-8)])
    # Most importantly, no AutoIAResult is returned.  Keep this assertion loose
    # across SciPy versions because the nominal proposal may itself fail first.
    assert excinfo.value is not None


def test_root_scalar_nested_closure_parameters_are_supported():
    def model(p):
        target = p[0]
        residual = lambda y: y * y - target
        return root_scalar(residual, bracket=[0.5, 2.0]).root

    result = autoia(model, [Interval(1.0, 1.21)])
    assert result.machine_rigorous
    assert result.lower[0] <= 1.0
    assert result.upper[0] >= 1.1


def test_fsolve_nested_closure_parameters_are_supported():
    def model(p):
        a, b = p[0], p[1]
        def residual(y):
            return np.array([
                y[0] + 0.2 * y[1] - a,
                0.1 * y[0] + y[1] + 0.05 * y[0] * y[1] - b,
            ])
        return fsolve(residual, [1.0, 1.5])

    result = autoia(model, [Interval(0.9, 1.1), Interval(1.4, 1.6)])
    assert result.machine_rigorous
    assert result.output_shape == (2,)


def elementary_system(y, a, b):
    return np.array([
        np.exp(y[0]) + 0.2 * y[1] - a,
        y[0] + np.sin(y[1]) - b,
    ])


def elementary_fsolve_model(p):
    return fsolve(elementary_system, [0.0, 0.5], args=(p[0], p[1]))


def test_fsolve_elementary_nonlinearity_certifies():
    result = autoia(
        elementary_fsolve_model,
        [Interval(1.15, 1.17), Interval(0.44, 0.46)],
    )
    assert result.machine_rigorous
    cert = [c for c in result.certificates if c.get("operation") == "scipy.optimize.fsolve"][-1]
    assert cert["rho_upper"] < 0.1



def test_imported_pure_python_root_callback_is_prepared_locally(tmp_path):
    helper = tmp_path / "implicit_helper.py"
    helper.write_text(
        "import numpy as np\n"
        "def residual(y, a):\n"
        "    return np.log(y) - a\n"
    )

    import importlib.util
    spec = importlib.util.spec_from_file_location("autoia_test_implicit_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    imported_residual = module.residual

    def model(p):
        return root_scalar(imported_residual, args=(p[0],), bracket=[0.5, 4.0]).root

    result = autoia(model, [Interval(0.0, np.log(2.0))])
    assert result.machine_rigorous
    assert result.lower[0] <= 1.0
    assert result.upper[0] >= 2.0


def test_scipy_optimize_functions_are_not_globally_mutated():
    import scipy.optimize as opt
    real_root_scalar = opt.root_scalar
    real_fsolve = opt.fsolve
    _ = autoia(sqrt_model_with_downstream_residual, [Interval(1.0, 1.01)])
    _ = autoia(fsolve_model, [Interval(0.99, 1.01), Interval(1.49, 1.51)])
    assert opt.root_scalar is real_root_scalar
    assert opt.fsolve is real_fsolve


def test_imported_pure_python_module_helper_with_root_scalar_is_prepared_locally(tmp_path):
    helper = tmp_path / "implicit_module_helper.py"
    helper.write_text(
        "import numpy as np\n"
        "from scipy.optimize import root_scalar\n"
        "def residual(y, a):\n"
        "    return np.log(y) - a\n"
        "def solve(a):\n"
        "    return root_scalar(residual, bracket=[0.5, 4.0], args=(a,)).root\n"
    )

    import importlib.util
    spec = importlib.util.spec_from_file_location("autoia_test_implicit_module_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Build the wrapper exactly like an ordinary module-level engineering model:
    # the imported helper module is a global, not a closure cell.
    namespace = {"module": module}
    exec("def model(p):\n    return module.solve(p[0])\n", namespace)
    model = namespace["model"]

    result = autoia(model, [Interval(0.0, np.log(2.0))])
    assert result.machine_rigorous
    assert result.lower[0] <= 1.0
    assert result.upper[0] >= 2.0
    cert = [c for c in result.certificates if c.get("operation") == "scipy.optimize.root_scalar"][-1]
    assert cert["unique_in_bracket"]


def test_imported_pure_python_module_helper_with_fsolve_is_prepared_locally(tmp_path):
    helper = tmp_path / "implicit_fsolve_module_helper.py"
    helper.write_text(
        "import numpy as np\n"
        "from scipy.optimize import fsolve\n"
        "def residual(y, a, b):\n"
        "    return np.array([y[0] + 0.2*y[1] - a, 0.1*y[0] + y[1] + 0.05*y[0]*y[1] - b])\n"
        "def solve(a, b):\n"
        "    return fsolve(residual, [1.0, 1.5], args=(a, b))\n"
    )

    import importlib.util
    spec = importlib.util.spec_from_file_location("autoia_test_implicit_fsolve_module_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    namespace = {"module": module}
    exec("def model(p):\n    return module.solve(p[0], p[1])\n", namespace)
    model = namespace["model"]

    result = autoia(model, [Interval(0.9, 1.1), Interval(1.4, 1.6)])
    assert result.machine_rigorous
    assert result.output_shape == (2,)
    cert = [c for c in result.certificates if c.get("operation") == "scipy.optimize.fsolve"][-1]
    assert cert["rho_upper"] < 1.0
