"""NVIDIA CUDA smoke check for AutoIA V1 acceleration.

Run this on a CUDA machine (including DGX Spark or a data-center NVIDIA GPU).
CUDA is used only for batched proposal/assist values.  The same boxes are then
independently certified by AutoIA's trusted CPU binary64 backend.
"""
import numpy as np
from autoia import Interval, prepare, available_accelerators, proposal_map


def engineering_model(p):
    x, y = p
    return np.exp(-x*x) + x*y + 0.1*y*y


def main():
    info = available_accelerators()
    print("accelerator report:")
    for k, v in info.items():
        print(k, v)
    if not info["cuda"].available:
        print("\nCUDA is not available in this Python/PyTorch environment.")
        return 2

    import torch

    def torch_model(x):
        return torch.exp(-x[:, 0]**2) + x[:, 0]*x[:, 1] + 0.1*x[:, 1]**2

    boxes = [
        [Interval(-0.8, -0.4), Interval(-0.2, 0.2)],
        [Interval(-0.4, 0.0), Interval(-0.2, 0.2)],
        [Interval(0.0, 0.4), Interval(-0.2, 0.2)],
        [Interval(0.4, 0.8), Interval(-0.2, 0.2)],
    ]
    centers = np.array([[0.5*(b[0].lo+b[0].hi), 0.5*(b[1].lo+b[1].hi)] for b in boxes])
    proposal = proposal_map(torch_model, centers, device="cuda", dtype="float64")
    print("\nCUDA proposal:", proposal)
    if proposal.machine_rigorous:
        raise RuntimeError("CUDA proposal was incorrectly labeled rigorous")

    session = prepare(engineering_model)
    certs, execution = session.enclose_many(
        boxes, workers="auto", parallel="auto", return_info=True, use_cache=False
    )
    print("\ntrusted CPU execution:", execution)
    for i, (center, value, cert) in enumerate(zip(centers, proposal.values, certs)):
        scalar = float(np.asarray(value))
        print(i, "center", center, "CUDA proposal", scalar,
              "CPU certificate", cert.lower[0], cert.upper[0])
        if not (cert.lower[0] <= scalar <= cert.upper[0]):
            raise RuntimeError(f"CUDA center proposal {i} not contained in CPU box certificate")
        if not cert.machine_rigorous:
            raise RuntimeError("CPU certificate was not machine rigorous")
    print("\nCUDA proposal + CPU verification smoke check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
