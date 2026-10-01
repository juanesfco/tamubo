"""
Check that CuPy and JAX agree on GPU identity, as ExactBO's AutoBound path assumes.

autobound_bounds.taylor_mu_q_bounds puts the GP arrays on
jax.devices("gpu")[cupy.cuda.Device().id] and hands the boxes over by DLPack
(which keeps them on their physical GPU). For every visible GPU, from a worker
thread like exactbo's sharding uses, this checks that a DLPack array from CuPy
device i lands on JAX device i, and that taylor_mu_q_bounds on device i returns
results on device i equal to those computed on device 0.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment3"))

from common import make_gp, random_boxes  # noqa: E402

from tamubo.exactbo.autobound_bounds import taylor_mu_q_bounds  # noqa: E402
from tamubo.exactbo.bounds import sigma_bound_factors  # noqa: E402

import jax  # noqa: E402  (imported by autobound_bounds first, which configures it)


def main() -> None:
    gp, X, _ = make_gp("problem10d", 32, 0)
    params = gp.kernel_.get_params()
    L_inv, _ = sigma_bound_factors(gp.L_)
    arrays = dict(
        X_train=X, alpha=np.asarray(gp.alpha_).ravel(), K_inv=L_inv.T @ L_inv,
        length_scale=np.asarray(params["k1__k2__length_scale"]), sigma_f_2=float(params["k1__k1__constant_value"]),
    )
    lo, hi = random_boxes(3000, X.shape[1], 15, np.random.default_rng(0))
    n_gpus = cp.cuda.runtime.getDeviceCount()
    jax_gpus = jax.devices("gpu")
    print(f"CuPy GPUs: {n_gpus}, JAX GPUs: {len(jax_gpus)}")

    def on_device(i: int):
        with cp.cuda.Device(i):
            imported = jax.dlpack.from_dlpack(cp.ones(4))
            (jax_device,) = imported.devices()
            out = taylor_mu_q_bounds(cp.asarray(lo), cp.asarray(hi), xp=cp, **arrays)
            out_devices = {int(a.device.id) for a in out}
            values = [cp.asnumpy(a) for a in out]
            pci = cp.cuda.Device(i).pci_bus_id
        return jax_device, out_devices, values, pci

    ok = True
    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        results = list(pool.map(on_device, range(n_gpus)))
    reference = results[0][2]
    for i, (jax_device, out_devices, values, pci) in enumerate(results):
        same_device = jax_device == jax_gpus[i] and out_devices == {i}
        same_values = all(np.array_equal(a, b) for a, b in zip(values, reference))
        ok &= same_device and same_values
        print(f"cupy {i} ({pci}): DLPack -> {jax_device}, outputs on cupy {sorted(out_devices)}, "
              f"device match {same_device}, values equal to GPU 0 {same_values}")
    print("DEVICE CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
