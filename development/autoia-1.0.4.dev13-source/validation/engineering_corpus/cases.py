"""Case definitions for the Version-1 engineering validation corpus.

This module contains validation metadata only.  The engineering model modules
remain independent of AutoIA and can be executed with ordinary float arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models.cantilever_beam import cantilever_beam
from .models.pressure_vessel import pressure_vessel
from .models.thermal_rc import thermal_rc
from .models.nozzle import nozzle
from .models.pipe_flow import pipe_pressure_drop
from .models.thermal_network import thermal_network
from .models.heat_exchanger import heat_exchanger
from .models.modal_2dof import modal_2dof


@dataclass(frozen=True)
class EngineeringCase:
    name: str
    model: Callable
    bounds: tuple[tuple[float, float], ...]
    order: int
    representation: str
    expected: str
    features: tuple[str, ...]
    description: str
    expected_exception: str | None = None
    expected_message: str | None = None
    invalid_witness: tuple[float, ...] | None = None


CASES = (
    EngineeringCase(
        name="cantilever_beam",
        model=cantilever_beam,
        bounds=(
            (900.0, 1100.0),       # force [N]
            (1.9, 2.1),            # length [m]
            (0.048, 0.052),        # width [m]
            (0.095, 0.105),        # height [m]
            (190.0e9, 210.0e9),    # Young's modulus [Pa]
            (7600.0, 8000.0),      # density [kg/m^3]
        ),
        order=4,
        representation="polynomial",
        expected="certified",
        features=("array-output", "sqrt", "real-powers", "rational-algebra"),
        description="Rectangular Euler-Bernoulli cantilever stress, deflection, and first-frequency calculation.",
    ),
    EngineeringCase(
        name="pressure_vessel_generator",
        model=pressure_vessel,
        bounds=((1.8e6, 2.2e6), (0.48, 0.52), (0.0095, 0.0105)),
        order=4,
        representation="generator",
        expected="certified",
        features=("shared-generators", "sqrt", "von-mises", "array-output"),
        description="Thin-walled pressure-vessel membrane and von-Mises stress calculation.",
    ),
    EngineeringCase(
        name="thermal_rc_transient",
        model=thermal_rc,
        bounds=((20.0, 30.0), (90.0, 110.0), (0.45, 0.55), (9000.0, 11000.0), (500.0, 700.0)),
        order=4,
        representation="polynomial",
        expected="certified",
        features=("exp", "rational-algebra", "array-output"),
        description="First-order lumped thermal transient with uncertain resistance and capacitance.",
    ),
    EngineeringCase(
        name="isentropic_nozzle",
        model=nozzle,
        bounds=((760.0, 840.0), (480.0e3, 520.0e3), (0.35, 0.45)),
        order=4,
        representation="polynomial",
        expected="certified",
        features=("sqrt", "noninteger-power", "ideal-gas", "array-output"),
        description="Ideal-gas isentropic nozzle temperature, velocity, density, and mass-flux relations.",
    ),
    EngineeringCase(
        name="pipe_flow_regime_branch",
        model=pipe_pressure_drop,
        bounds=((990.0, 1010.0), (0.035, 0.060), (0.00095, 0.00105), (0.048, 0.052), (9.5, 10.5)),
        order=4,
        representation="polynomial",
        expected="certified",
        features=("uncertain-if", "path-aware", "negative-power", "darcy-weisbach"),
        description="Darcy-Weisbach pressure loss with an uncertain laminar/turbulent Reynolds-number branch.",
    ),
    EngineeringCase(
        name="thermal_network_solve",
        model=thermal_network,
        bounds=(
            (9.0, 11.0), (7.2, 8.8), (2.7, 3.3), (3.6, 4.4),
            (90.0, 110.0), (45.0, 55.0), (20.0, 30.0), (18.0, 22.0),
        ),
        order=3,
        representation="polynomial",
        expected="certified",
        features=("numpy-array-assembly", "semantic-linear-solve", "parameter-dependent-matrix"),
        description="Three-node steady thermal network assembled with NumPy and solved by np.linalg.solve.",
    ),
    EngineeringCase(
        name="heat_exchanger_lmtd",
        model=heat_exchanger,
        bounds=((180.0, 190.0), (110.0, 115.0), (25.0, 30.0), (70.0, 75.0), (450.0, 550.0), (9.0, 11.0)),
        order=4,
        representation="polynomial",
        expected="certified",
        features=("log", "domain-proof", "rational-algebra", "lmtd"),
        description="Counterflow heat-exchanger log-mean-temperature-difference calculation on a valid separated box.",
    ),
    EngineeringCase(
        name="heat_exchanger_lmtd_singular_box",
        model=heat_exchanger,
        bounds=((175.0, 185.0), (115.0, 125.0), (25.0, 35.0), (75.0, 85.0), (450.0, 550.0), (9.0, 11.0)),
        order=4,
        representation="polynomial",
        expected="certification-failure",
        expected_exception="CertificationError",
        expected_message="zero",
        invalid_witness=(175.0, 115.0, 25.0, 85.0, 500.0, 10.0),
        features=("log", "division-by-zero", "fail-closed", "physical-domain-invalid"),
        description="Same LMTD source over a box containing deltaT1=deltaT2, where the literal source formula has 0/0.",
    ),
    EngineeringCase(
        name="modal_2dof_eigen",
        model=modal_2dof,
        bounds=((9.5, 10.5), (11.5, 12.5), (950.0, 1050.0), (1150.0, 1250.0), (450.0, 550.0)),
        order=3,
        representation="polynomial",
        expected="unsupported",
        expected_exception="UnsupportedOperationError",
        expected_message="np.linalg.eigvals",
        features=("numpy-linalg", "eigenvalues", "deferred-v1-capability"),
        description="Conventional 2-DOF modal analysis retained as an explicit Version-1 unsupported boundary.",
    ),
)

CASE_BY_NAME = {case.name: case for case in CASES}
