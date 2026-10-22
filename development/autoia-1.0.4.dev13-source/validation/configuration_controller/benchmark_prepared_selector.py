"""Prepared/many-box benchmark for Step 20a Phase 2."""
from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import numpy as np
from autoia import prepare, Interval, ConfigurationPolicy, __version__


def mild_model(p):
    return np.exp(p[0]) + 0.1*p[0]


def escalation_model(p):
    return np.exp(3.0*p[0])


def boxes_for(centers, halfwidth):
    return [[Interval(float(c-halfwidth), float(c+halfwidth))] for c in centers]


def run_one(name, model, boxes):
    fixed=prepare(model, order=4)
    auto=prepare(model, order=4, configuration_policy=ConfigurationPolicy(mode="auto"))
    t0=time.perf_counter(); rf=fixed.certify_many(boxes); tf=time.perf_counter()-t0
    t0=time.perf_counter(); ra=auto.certify_many(boxes); ta=time.perf_counter()-t0
    fixed_w=np.vstack([r.width for r in rf])
    auto_w=np.vstack([r.width for r in ra])
    # Exact-repeat pass should use the already-populated per-configuration caches.
    before=auto.stats.engineering_model_executions
    t0=time.perf_counter(); auto.certify_many(boxes); repeat_time=time.perf_counter()-t0
    after=auto.stats.engineering_model_executions
    return {
        "name":name,"boxes":len(boxes),
        "fixed_seconds":tf,"auto_seconds":ta,"runtime_ratio":ta/max(tf,np.finfo(float).tiny),
        "max_width_ratio":float(np.max(np.divide(auto_w,fixed_w,out=np.ones_like(auto_w),where=fixed_w>0))),
        "min_width_ratio":float(np.min(np.divide(auto_w,fixed_w,out=np.ones_like(auto_w),where=fixed_w>0))),
        "auto_stats_first_pass":auto.stats.__dict__,
        "repeat_seconds":repeat_time,"repeat_added_model_executions":int(after-before),
    }


def main():
    rows=[
        run_one("no-escalation", mild_model, boxes_for(np.linspace(-0.4,0.4,24),0.05)),
        run_one("higher-order-escalation", escalation_model, boxes_for(np.linspace(-0.15,0.15,16),1.0)),
    ]
    out={"autoia_version":__version__,"platform":{"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),"numpy":np.__version__},"cases":rows}
    Path("STEP20A_PREPARED_SELECTOR_RESULTS.json").write_text(json.dumps(out,indent=2)+"\n")
    lines=[f"AutoIA {__version__} prepared automatic-selector benchmark"]
    for r in rows:
        lines.append(f"{r['name']}: boxes={r['boxes']} runtime_ratio={r['runtime_ratio']:.6g} width_ratio=[{r['min_width_ratio']:.6g},{r['max_width_ratio']:.6g}] candidate_sessions={r['auto_stats_first_pass']['candidate_sessions']} model_exec={r['auto_stats_first_pass']['engineering_model_executions']} repeat_added_exec={r['repeat_added_model_executions']}")
    Path("STEP20A_PREPARED_SELECTOR_RESULTS.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=='__main__': main()
