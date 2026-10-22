import numpy as np
import pytest
import scipy
from tests.support import interpolation_helper as _IMPORTED_HELPER

scipy_interpolate = pytest.importorskip("scipy.interpolate")
scipy_linalg = pytest.importorskip("scipy.linalg")
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh

from autoia import autoia, Interval
from autoia.exceptions import UnsupportedOperationError, ModelExecutionError


def _table_2d(xgrid, ygrid):
    values = np.empty((len(xgrid), len(ygrid)))
    for i, x in enumerate(xgrid):
        for j, y in enumerate(ygrid):
            values[i, j] = 1.0 + 2.0*x + 3.0*y + 4.0*x*y
    return values


_X = np.array([0.0, 1.0, 2.0])
_Y = np.array([0.0, 2.0])
_VALUES = _table_2d(_X, _Y)
_RGI = RegularGridInterpolator((_X, _Y), _VALUES, method="linear")


def _global_rgi_model(p):
    return _RGI((p[0], p[1]))


def test_regular_grid_interpolator_single_cell_contains_dense_samples():
    result = autoia(
        _global_rgi_model,
        [Interval(0.2, 0.8), Interval(0.4, 1.6)],
    )
    assert result.machine_rigorous
    for x in np.linspace(0.2, 0.8, 15):
        for y in np.linspace(0.4, 1.6, 15):
            value = float(_RGI((float(x), float(y))))
            assert result.lower[0] <= value <= result.upper[0]


def test_regular_grid_interpolator_crosses_grid_boundary_pathwise():
    result = autoia(
        _global_rgi_model,
        [Interval(0.2, 1.4), Interval(0.5, 1.5)],
    )
    assert result.machine_rigorous
    assert len(getattr(result, "branch_results", ())) >= 2
    for x in np.linspace(0.2, 1.4, 21):
        for y in np.linspace(0.5, 1.5, 11):
            value = float(_RGI((float(x), float(y))))
            assert result.lower[0] <= value <= result.upper[0]


def test_regular_grid_interpolator_constructor_inside_model_is_prepared():
    def model(p):
        rgi = RegularGridInterpolator((_X, _Y), _VALUES, method="linear")
        return rgi((p[0], p[1]))

    result = autoia(model, [Interval(0.1, 0.9), Interval(0.2, 1.4)])
    assert result.machine_rigorous


def test_regular_grid_interpolator_descending_axis_and_vector_values():
    xdesc = _X[::-1]
    values = np.empty((len(xdesc), len(_Y), 2))
    for i, x in enumerate(xdesc):
        for j, y in enumerate(_Y):
            values[i, j, 0] = 1.0 + 2.0*x + 3.0*y + 4.0*x*y
            values[i, j, 1] = -2.0 + x - 0.5*y
    rgi = RegularGridInterpolator((xdesc, _Y), values, method="linear")

    def model(p):
        local = RegularGridInterpolator((xdesc, _Y), values, method="linear")
        return local((p[0], p[1]))

    result = autoia(model, [Interval(0.2, 1.4), Interval(0.5, 1.5)])
    assert result.machine_rigorous
    assert result.output_shape == (2,)
    for x in np.linspace(0.2, 1.4, 9):
        for y in np.linspace(0.5, 1.5, 9):
            v = np.asarray(rgi((float(x), float(y))), dtype=float)
            assert np.all(result.lower <= v)
            assert np.all(v <= result.upper)


def test_regular_grid_interpolator_linear_extrapolation_fill_none():
    rgi = RegularGridInterpolator(
        (np.array([0.0, 1.0]),), np.array([0.0, 2.0]),
        method="linear", bounds_error=False, fill_value=None,
    )
    def model(p):
        local = RegularGridInterpolator(
            (np.array([0.0, 1.0]),), np.array([0.0, 2.0]),
            method="linear", bounds_error=False, fill_value=None,
        )
        return local((p[0],))
    result = autoia(model, [Interval(-0.5, 0.5)])
    assert result.machine_rigorous
    assert result.lower[0] <= -1.0
    assert result.upper[0] >= 1.0


def test_regular_grid_interpolator_finite_fill_value_across_boundary():
    def model(p):
        local = RegularGridInterpolator(
            (np.array([0.0, 1.0]),), np.array([0.0, 2.0]),
            method="linear", bounds_error=False, fill_value=-7.0,
        )
        return local((p[0],))
    result = autoia(model, [Interval(-0.5, 0.5)])
    assert result.machine_rigorous
    assert result.lower[0] <= -7.0
    assert result.upper[0] >= 1.0


def test_regular_grid_interpolator_bounds_error_fails_when_box_leaves_grid():
    def model(p):
        local = RegularGridInterpolator(
            (np.array([0.0, 1.0]),), np.array([0.0, 2.0]),
            method="linear", bounds_error=True,
        )
        return local((p[0],))
    with pytest.raises(ModelExecutionError):
        autoia(model, [Interval(-0.1, 0.5)])


def test_regular_grid_interpolator_non_linear_method_fails_explicitly():
    def model(p):
        local = RegularGridInterpolator(
            (np.array([0.0, 1.0]),), np.array([0.0, 2.0]), method="nearest"
        )
        return local((p[0],))
    with pytest.raises(UnsupportedOperationError, match="method='linear'"):
        autoia(model, [Interval(0.2, 0.5)])


def _eigh_alias_model(p):
    K = np.array([[2.0 + p[0], -1.0], [-1.0, 2.0]])
    return eigh(K, eigvals_only=True)[0]


def test_scipy_linalg_eigh_fails_with_explicit_semantic_message():
    with pytest.raises(UnsupportedOperationError, match="scipy.linalg.eigh") as exc:
        autoia(_eigh_alias_model, [Interval(0.0, 0.2)])
    assert "No certificate was produced" in str(exc.value)


def test_scipy_linalg_eigh_numeric_calls_remain_ordinary_scipy():
    K = np.array([[2.0, -1.0], [-1.0, 2.0]])
    w = eigh(K, eigvals_only=True)
    assert np.allclose(w, [1.0, 3.0])


def test_regular_grid_interpolator_imported_helper_module_is_prepared():
    def model(p):
        return _IMPORTED_HELPER.evaluate(p[0], p[1])

    result = autoia(model, [Interval(0.2, 0.8), Interval(-0.6, 0.6)])
    assert result.machine_rigorous
    for x in np.linspace(0.2, 0.8, 9):
        for y in np.linspace(-0.6, 0.6, 9):
            v = float(_IMPORTED_HELPER.evaluate(float(x), float(y)))
            assert result.lower[0] <= v <= result.upper[0]


def test_regular_grid_interpolator_uncertain_table_is_explicitly_unsupported():
    def model(p):
        values = np.array([0.0, p[0]])
        local = RegularGridInterpolator((np.array([0.0, 1.0]),), values, method="linear")
        return local((0.5,))

    with pytest.raises(UnsupportedOperationError, match="fixed numeric table values"):
        autoia(model, [Interval(0.9, 1.1)])


def test_scipy_module_eigh_path_fails_explicitly():
    def model(p):
        K = np.array([[2.0 + p[0], -1.0], [-1.0, 2.0]])
        return scipy.linalg.eigh(K, eigvals_only=True)[0]

    with pytest.raises(UnsupportedOperationError, match="scipy.linalg.eigh"):
        autoia(model, [Interval(0.0, 0.2)])
