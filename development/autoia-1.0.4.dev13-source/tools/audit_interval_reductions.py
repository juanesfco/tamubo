#!/usr/bin/env python3
"""Reproducible Step-9b soundness/performance audit."""
from __future__ import annotations
import argparse, time, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from autoia import autoia, Interval, RigorousBinary64Backend
from autoia.trusted import rigorous_dot_point_interval, rigorous_dot_interval, rigorous_matmul_point, rigorous_matmul_point_interval, rigorous_dot_point
from tests.support.exact_binary64 import exact_float, lower_endpoint_le_exact, upper_endpoint_ge_exact

def exact_pi(a,x):
    lo=Fraction(0); hi=Fraction(0)
    for ai,I in zip(a,x):
        aq=exact_float(ai); l=exact_float(I.lo); h=exact_float(I.hi)
        if aq>=0: lo+=aq*l; hi+=aq*h
        else: lo+=aq*h; hi+=aq*l
    return lo,hi

def exact_ii(a,b):
    lo=Fraction(0); hi=Fraction(0)
    for A,B in zip(a,b):
        vals=[exact_float(x)*exact_float(y) for x in (A.lo,A.hi) for y in (B.lo,B.hi)]
        lo+=min(vals); hi+=max(vals)
    return lo,hi

def contains(I,lo,hi): return lower_endpoint_le_exact(I.lo,lo) and upper_endpoint_ge_exact(I.hi,hi)
def best(fn,reps=3):
    vals=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
    return min(vals)

def make_intervals(rng,n,scale=.2):
    lo=rng.normal(size=n); hi=lo+np.abs(rng.normal(size=n))*scale
    return np.array([Interval(l,h) for l,h in zip(lo,hi)],dtype=object)

def make_model(n):
    def model(p):
        x,y=p; A=np.zeros((n,n)); b=np.zeros(n)
        for i in range(n):
            A[i,i]=3.0+0.03*x*x+0.01*(i+1)/n*y
            if i+1<n:
                A[i,i+1]=0.02*x; A[i+1,i]=-0.015*y
            b[i]=1.0+0.005*(i+1)*x*y+0.002*y*y
        return np.linalg.solve(A,b)
    return model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--exact-cases',type=int,default=300); ap.add_argument('--size',type=int,default=28); ap.add_argument('--seed',type=int,default=2026082920); args=ap.parse_args()
    rng=np.random.default_rng(args.seed); bvec=RigorousBinary64Backend(reduction_method='vectorized')
    pi_checks=ii_checks=0
    for _ in range(args.exact_cases):
        n=int(rng.integers(16,257)); a=rng.normal(size=n); x=make_intervals(rng,n); y=make_intervals(rng,n)
        I=rigorous_dot_point_interval(a,x,bvec); lo,hi=exact_pi(a,x)
        if not contains(I,lo,hi): raise AssertionError(('point-interval',n,I,lo,hi))
        pi_checks+=1
        J=rigorous_dot_interval(x,y,bvec); lo,hi=exact_ii(x,y)
        if not contains(J,lo,hi): raise AssertionError(('interval-interval',n,J,lo,hi))
        ii_checks+=1
    print(f'Exact-rational point×interval checks passed: {pi_checks:,}')
    print(f'Exact-rational interval×interval checks passed: {ii_checks:,}')
    print('Reduction speedups (reference/vectorized):')
    for n in (64,256,1024):
        a=rng.normal(size=n); x=make_intervals(rng,n); y=make_intervals(rng,n)
        br=RigorousBinary64Backend(reduction_method='reference'); bv=RigorousBinary64Backend(reduction_method='vectorized')
        reps=max(3,1500//n)
        tr=best(lambda:[rigorous_dot_point_interval(a,x,br) for _ in range(reps)],2)
        tv=best(lambda:[rigorous_dot_point_interval(a,x,bv) for _ in range(reps)],2)
        ir=best(lambda:[rigorous_dot_interval(x,y,br) for _ in range(reps)],2)
        iv=best(lambda:[rigorous_dot_interval(x,y,bv) for _ in range(reps)],2)
        print(f'  n={n:4d}: point×interval {tr/tv:8.2f}x; interval×interval {ir/iv:8.2f}x')
    # Matrix batching compared with the Step-9 per-entry vectorized-dot architecture.
    n=args.size; A=rng.normal(size=(n,n)); B=rng.normal(size=(n,n)); BI=np.empty((n,n),dtype=object)
    for idx in np.ndindex(BI.shape):
        c=float(rng.normal()); BI[idx]=Interval(c-0.03,c+0.03)
    bv=RigorousBinary64Backend(reduction_method='vectorized')
    def old_point_matmul():
        out=np.empty((n,n),dtype=object)
        for i in range(n):
            for j in range(n): out[i,j]=rigorous_dot_point(A[i],B[:,j],bv)
        return out
    def old_pi_matmul():
        out=np.empty((n,n),dtype=object)
        for i in range(n):
            for j in range(n): out[i,j]=rigorous_dot_point_interval(A[i],BI[:,j],bv)
        return out
    oldp=best(old_point_matmul,2); newp=best(lambda:rigorous_matmul_point(A,B,bv),3)
    oldi=best(old_pi_matmul,2); newi=best(lambda:rigorous_matmul_point_interval(A,BI,bv),3)
    print(f'Batched {n}x{n} point matmul speedup vs per-entry vectorized dots: {oldp/newp:.2f}x ({oldp:.6f}s -> {newp:.6f}s)')
    print(f'Batched {n}x{n} point×interval matmul speedup: {oldi/newi:.2f}x ({oldi:.6f}s -> {newi:.6f}s)')
    box=[Interval(-.5,.5),Interval(-.4,.4)]; model=make_model(n)
    bref=RigorousBinary64Backend(reduction_method='reference'); baut=RigorousBinary64Backend(reduction_method='auto')
    t=time.perf_counter(); rr=autoia(model,box,order=2,backend=bref); tr=time.perf_counter()-t
    t=time.perf_counter(); ra=autoia(model,box,order=2,backend=baut); ta=time.perf_counter()-t
    same=np.array_equal(rr.lower.view(np.uint64),ra.lower.view(np.uint64)) and np.array_equal(rr.upper.view(np.uint64),ra.upper.view(np.uint64))
    print(f'End-to-end {n}x{n} solve: reference={tr:.6f}s auto={ta:.6f}s speedup={tr/ta:.2f}x')
    print(f'End-to-end endpoints bit-identical in this benchmark: {same}')
    print(f'Auto reduction stats: {baut.reduction_stats}')
if __name__=='__main__': main()
