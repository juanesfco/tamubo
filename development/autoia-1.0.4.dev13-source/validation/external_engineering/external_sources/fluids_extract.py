"""Selected verbatim computational bodies from the external ``fluids`` project.

Provenance
----------
Project: fluids, Caleb Bell
License: MIT
Pinned release: 1.3.1
Pinned source commit used for the core module: e49c8b0
Upstream files: fluids/drag.py and fluids/friction.py

Only compact functions needed by the validation corpus are vendored here.  Their
computational statements are retained verbatim.  Docstrings are shortened in
this validation copy; that shortening does not alter executable source logic.
"""

from math import log, log10, sqrt


def Clift_Gauvin(Re: float) -> float:
    """External drag correlation from fluids.drag.Clift_Gauvin."""
    return 24./Re*(1 + 0.152*Re**0.677) + 0.417/(1 + 5070*Re**-0.94)


def Morsi_Alexander(Re: float) -> float:
    """External piecewise drag correlation from fluids.drag.Morsi_Alexander."""
    if Re < 0.1:
        return 24./Re
    elif Re < 1:
        return 22.73/Re + 0.0903/Re**2 + 3.69
    elif Re < 10:
        return 29.1667/Re - 3.8889/Re**2 + 1.2220
    elif Re < 100:
        return 46.5/Re - 116.67/Re**2 + 0.6167
    elif Re < 1000:
        return 98.33/Re - 2778./Re**2 + 0.3644
    elif Re < 5000:
        return 148.62/Re - 4.75E4/Re**2 + 0.3570
    elif Re < 10000:
        return -490.546/Re + 57.87E4/Re**2 + 0.46
    else:
        return -1662.5/Re + 5.4167E6/Re**2 + 0.5191


def Churchill_1977(Re: float, eD: float) -> float:
    """External all-regime friction-factor correlation from fluids.friction."""
    A3 = (37530/Re)**16
    A2 = (2.457*log((7./Re)**0.9 + 0.27*eD))**16
    ff = 2.0*((8.0/Re)**12 + 1.0/(A2+A3)**1.5)**(1.0/12.)
    return 4.0*ff


def Swamee_Jain_1976(Re: float, eD: float) -> float:
    """External friction-factor correlation using base-10 logarithm."""
    term = (-4.0*log10((6.97/Re)**0.9 + eD*(1.0/3.7)))
    return 4.0/(term*term)
