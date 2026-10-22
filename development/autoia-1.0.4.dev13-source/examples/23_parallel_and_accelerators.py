"""23 — Parallel CPU proof and optional MPS/CUDA proposal acceleration.

The rigorous many-box path is CPU binary64 and can use multiple CPU workers.
Apple MPS and NVIDIA CUDA are optional *proposal/assist* devices: their values
are explicitly non-rigorous and never replace the independent CPU certificate.
"""
import numpy as np
from autoia import (
    Interval, prepare, available_accelerators, proposal_map,
)


def engineering_model(p):
    x, y = p
    return np.exp(-x*x) + 0.25*y*y + x*y


if __name__ == "__main__":
    boxes = [
        [Interval(-1.0, -0.5), Interval(-0.2, 0.2)],
        [Interval(-0.5, 0.0), Interval(-0.2, 0.2)],
        [Interval(0.0, 0.5), Interval(-0.2, 0.2)],
        [Interval(0.5, 1.0), Interval(-0.2, 0.2)],
    ]

    session = prepare(engineering_model)
    results, info = session.enclose_many(
        boxes, workers=2, parallel="process", return_info=True,
        use_cache=False,
    )
    print("rigorous CPU many-box execution:")
    print(info)
    for i, r in enumerate(results):
        print(i, r.lower, r.upper, "rigorous=", r.machine_rigorous)

    print("\navailable proposal accelerators:")
    for name, item in available_accelerators().items():
        print(name, item)

    # Optional GPU proposal example.  V1 never silently selects a GPU for
    # device='auto': proposal performance depends on arithmetic intensity.
    # This example opts in explicitly when an accelerator is available.
    try:
        import torch

        def torch_proposal(x):
            return torch.exp(-x[:, 0]**2) + 0.25*x[:, 1]**2 + x[:, 0]*x[:, 1]

        centers = np.array([
            [0.5*(b[0].lo+b[0].hi), 0.5*(b[1].lo+b[1].hi)] for b in boxes
        ])
        accel = available_accelerators()
        device = "mps" if accel["mps"].available else (
            "cuda" if accel["cuda"].available else "cpu"
        )
        proposal = proposal_map(torch_proposal, centers, device=device)
        print("\nproposal device:", proposal.device)
        print("proposal dtype:", proposal.dtype)
        print("proposal values:", proposal.values)
        print("proposal machine rigorous:", proposal.machine_rigorous)
        assert proposal.machine_rigorous is False
    except Exception as exc:
        print("\noptional accelerator proposal not run:", type(exc).__name__, exc)
