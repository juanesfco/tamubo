"""Independent 80-digit containment audit for frozen supported GP posteriors."""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np
import mpmath as mp

from autoia import autoia, Interval
from autoia.integrations import freeze_gpytorch_exact_gp, frozen_gp_posterior

ZeroMean = type("ZeroMean", (), {})
RBFKernel = type("RBFKernel", (), {})
MaternKernel = type("MaternKernel", (), {})
ScaleKernel = type("ScaleKernel", (), {})
GaussianLikelihood = type("GaussianLikelihood", (), {})

class StructuralGP:
    __autoia_supported_exact_gp__ = True
    def __init__(self, profile):
        self.train_inputs = (np.array([[0.0], [0.22], [0.55], [0.83], [1.0]], dtype=float),)
        self.train_targets = np.array([0.1, 0.7, -0.2, 0.4, 0.0], dtype=float)
        self.mean_module = ZeroMean()
        if profile == "rbf":
            k = RBFKernel(); k.lengthscale = np.array([0.31])
        else:
            k = MaternKernel(); k.lengthscale = np.array([0.31]); k.nu = {"m12":0.5,"m32":1.5,"m52":2.5}[profile]
        s = ScaleKernel(); s.base_kernel = k; s.outputscale = np.array(1.17)
        self.covar_module = s
        like = GaussianLikelihood(); like.noise = np.array([0.013])
        self.likelihood = like

def mpf64(x):
    f = Fraction.from_float(float(x))
    return mp.mpf(f.numerator) / mp.mpf(f.denominator)

def hp_point(x, state):
    x = mpf64(x)
    ell = mpf64(state.lengthscale[0]); os = mpf64(state.outputscale)
    kvals=[]
    for xi in state.train_x[:,0]:
        r=abs((x-mpf64(xi))/ell); r2=r*r
        if state.kernel_kind == "RBFKernel": base=mp.e**(-mp.mpf('0.5')*r2)
        elif state.matern_nu == 0.5: base=mp.e**(-r)
        elif state.matern_nu == 1.5:
            c=mp.sqrt(3); base=(1+c*r)*mp.e**(-c*r)
        else:
            c=mp.sqrt(5); base=(1+c*r+mp.mpf(5)/3*r2)*mp.e**(-c*r)
        kvals.append(os*base)
    mu=mpf64(state.mean_constant)
    for a,k in zip(state.alpha,kvals): mu += mpf64(a)*k
    v=[]
    for i,k in enumerate(kvals):
        rhs=k
        for j in range(i): rhs -= mpf64(state.chol[i,j])*v[j]
        v.append(rhs/mpf64(state.chol[i,i]))
    var=os-sum(z*z for z in v)
    return mu,var

def main():
    mp.mp.dps=80
    checks=0
    for profile in ("rbf","m12","m32","m52"):
        state=freeze_gpytorch_exact_gp(StructuralGP(profile))
        def f(p):
            m,v=frozen_gp_posterior([[p[0]]],state)
            return np.array([m[0],v[0]],dtype=object)
        r=autoia(f,[Interval.enclosing("0.20","0.30")],representation="interval")
        lo=np.asarray(r.lower,float); hi=np.asarray(r.upper,float)
        for x in np.linspace(0.2,0.3,63):
            m,v=hp_point(float(x),state)
            vals=(m,v)
            for j,val in enumerate(vals):
                if not (mpf64(lo[j]) <= val <= mpf64(hi[j])):
                    raise AssertionError((profile,x,j,val,lo[j],hi[j]))
                checks += 1
    print("profiles = 4")
    print(f"high_precision_posterior_containment_checks = {checks}")
    print("status = PASS")

if __name__ == "__main__": main()
