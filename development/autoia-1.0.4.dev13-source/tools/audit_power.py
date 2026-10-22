#!/usr/bin/env python3
"""Independent high-precision audit of AutoIA trusted real-power kernels."""
from __future__ import annotations

from pathlib import Path
import sys
import argparse, random
import mpmath as mp
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia.trusted import RigorousBinary64Backend, real_power_point_interval


def mp_exact_float(x):
    n,d=float(x).as_integer_ratio(); return mp.mpf(n)/mp.mpf(d)

def contains(I,v):
    return mp_exact_float(I.lo) <= v <= mp_exact_float(I.hi)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cases',type=int,default=300); ap.add_argument('--dps',type=int,default=120); args=ap.parse_args()
    mp.mp.dps=args.dps; b=RigorousBinary64Backend(); rng=random.Random(20260829); passed=0
    fixed=[(0.25,.5),(4.,.5),(.5,1.25),(2.,-.25),(3.,-2.),(-2.,3.),(-2.,-3.)]
    for x,p in fixed:
        I=real_power_point_interval(x,p,b); exact=mp.power(mp_exact_float(x),mp_exact_float(p));
        if not contains(I,exact): raise AssertionError((x,p,I,exact))
        passed+=1
    for _ in range(args.cases):
        x=10.0**rng.uniform(-8,8); p=rng.uniform(-5,5)
        if float(p).is_integer(): p += 0.125
        I=real_power_point_interval(x,p,b); exact=mp.power(mp_exact_float(x),mp_exact_float(p))
        if not contains(I,exact): raise AssertionError((x,p,I,exact))
        passed+=1
    print(f'Power high-precision checks passed: {passed}')
    print(f'Random positive-base cases: {args.cases}')
    print(f'mpmath precision: {args.dps} decimal digits')
if __name__=='__main__': main()
