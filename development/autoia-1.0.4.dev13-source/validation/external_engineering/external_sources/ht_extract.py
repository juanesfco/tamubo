"""Selected external function from the MIT-licensed ``ht`` project.

Project: ht, Caleb Bell
Upstream file: ht/conduction.py
License: MIT

The executable body of ``cylindrical_heat_transfer`` below is retained from the
upstream function.  Its long documentation string is omitted in this validation
copy so the corpus remains compact.
"""

from math import log, pi


def cylindrical_heat_transfer(Ti, To, hi, ho, Di, ts, ks):
    length = 1.0 # basis
    external_diameter = Di + 2.0*sum(ts)
    A_external = pi*external_diameter*length
    A_internal = pi*Di*length
    Rs = []
    Do_running = Di
    R_layers = 0.0
    for i in range(len(ts)):
        Do_running, Di_running = 2.0*ts[i]+Do_running, Do_running
        Ri = 0.5*external_diameter*log(Do_running/Di_running)/ks[i]
        R_layers += Ri
        Rs.append(Ri)
    D_ratio = external_diameter/Di
    inv_term = D_ratio/hi + R_layers + 1.0/ho
    U_external = 1.0/inv_term
    UA = A_external*U_external
    dT = Ti - To
    Q = UA*dT
    q = Q/A_external
    Ts = [Ti]
    for Ri in Rs:
        Ts.append(Ts[-1] - q*Ri)
    ans = {"Q": Q, "q": q, "UA": UA, "U_outer": U_external, "U_inner": UA/A_internal, "Ts": Ts, "Rs": Rs}
    return ans
