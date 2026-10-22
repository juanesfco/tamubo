"""Manifest for the pre-existing external engineering-code validation tranche."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .adapters import ExternalVectorAdapter
from .external_sources.fluids_extract import (
    Clift_Gauvin,
    Morsi_Alexander,
    Churchill_1977,
    Swamee_Jain_1976,
)
from .external_sources.ht_extract import cylindrical_heat_transfer
from .external_sources.grheat_extract import PlaneInstantaneousExternal


@dataclass(frozen=True)
class ExternalEngineeringCase:
    name: str
    project: str
    license: str
    provenance: str
    upstream_symbol: str
    model: Callable
    bounds: tuple[tuple[float, float], ...]
    order: int
    expected: str
    features: tuple[str, ...]
    description: str
    expected_exception: str | None = None
    expected_message: str | None = None


def _one(p):
    return (p[0],), {}


def _two(p):
    return (p[0], p[1]), {}


def _ht_args(p):
    # Only the scalar uncertainty interface is adapted.  The external function's
    # list loops, dictionary assembly, logarithm, and thermal-resistance logic
    # execute unchanged.
    Ti, To, hi, ho, Di = p
    return (Ti, To, hi, ho, Di, [0.015, 0.025], [0.20, 0.045]), {}


def _ht_output(ans):
    return (ans["Q"], ans["q"], ans["UA"], ans["U_outer"], ans["U_inner"])


_plane = PlaneInstantaneousExternal(
    zp=0.001,
    tp=0.0,
    diffusivity=1.4e-7,
    capacity=4.18e6,
    boundary="adiabatic",
)


def _plane_args(p):
    return (p[0], p[1], 0.0), {}


CASES = (
    ExternalEngineeringCase(
        name="fluids_clift_gauvin_drag",
        project="fluids",
        license="MIT",
        provenance="fluids 1.3.1; external function body from fluids/drag.py",
        upstream_symbol="fluids.drag.Clift_Gauvin",
        model=ExternalVectorAdapter(Clift_Gauvin, _one),
        bounds=((500.0, 1500.0),),
        order=4,
        expected="certified",
        features=("external-source", "real-powers", "rational-algebra"),
        description="Particle drag correlation from the pre-existing fluids engineering library.",
    ),
    ExternalEngineeringCase(
        name="fluids_morsi_alexander_branch",
        project="fluids",
        license="MIT",
        provenance="fluids 1.3.1; external function body from fluids/drag.py",
        upstream_symbol="fluids.drag.Morsi_Alexander",
        model=ExternalVectorAdapter(Morsi_Alexander, _one),
        bounds=((800.0, 1200.0),),
        order=4,
        expected="certified",
        features=("external-source", "uncertain-if", "path-aware", "piecewise-correlation"),
        description="Piecewise Morsi-Alexander drag correlation crossing the Re=1000 branch boundary.",
    ),
    ExternalEngineeringCase(
        name="fluids_churchill_friction",
        project="fluids",
        license="MIT",
        provenance="fluids 1.3.1; external function body from fluids/friction.py",
        upstream_symbol="fluids.friction.Churchill_1977",
        model=ExternalVectorAdapter(Churchill_1977, _two),
        bounds=((8.0e4, 1.2e5), (5.0e-5, 2.0e-4)),
        order=5,
        expected="certified",
        features=("external-source", "log", "real-powers", "friction-factor"),
        description="Churchill all-regime Darcy friction factor using natural logarithm; order 5 is retained because order 4 is rigorous but impractically loose on this box.",
    ),
    ExternalEngineeringCase(
        name="fluids_swamee_jain_log10",
        project="fluids",
        license="MIT",
        provenance="fluids 1.3.1; external function body from fluids/friction.py",
        upstream_symbol="fluids.friction.Swamee_Jain_1976",
        model=ExternalVectorAdapter(Swamee_Jain_1976, _two),
        bounds=((8.0e4, 1.2e5), (5.0e-5, 2.0e-4)),
        order=4,
        expected="certified",
        features=("external-source", "log10", "real-powers", "friction-factor"),
        description="Swamee-Jain Darcy friction factor deliberately probes common math.log10 compatibility.",
    ),
    ExternalEngineeringCase(
        name="ht_cylindrical_heat_transfer",
        project="ht",
        license="MIT",
        provenance="pre-existing ht/conduction.py cylindrical_heat_transfer function",
        upstream_symbol="ht.conduction.cylindrical_heat_transfer",
        model=ExternalVectorAdapter(cylindrical_heat_transfer, _ht_args, _ht_output),
        bounds=((370.0, 390.0), (290.0, 310.0), (80.0, 120.0), (15.0, 25.0), (0.045, 0.055)),
        order=4,
        expected="certified",
        features=("external-source", "for-loop", "lists", "dict", "log", "multi-output"),
        description="Multilayer cylindrical conduction/convection routine from the pre-existing ht library.",
    ),
    ExternalEngineeringCase(
        name="grheat_plane_instantaneous",
        project="grheat",
        license="MIT",
        provenance="grheat 0.5.1-era source; Plane._instantaneous from grheat/plane_source.py",
        upstream_symbol="grheat.Plane._instantaneous",
        model=ExternalVectorAdapter(_plane._instantaneous, _plane_args),
        bounds=((0.0009, 0.0011), (0.095, 0.105)),
        order=4,
        expected="certified",
        features=("external-source", "bound-method", "numpy", "sqrt", "exp", "constant-branch"),
        description="Planar Green-function instantaneous heat source from pre-existing grheat.",
    ),
)

CASE_BY_NAME = {case.name: case for case in CASES}
