"""04 — Ordinary Python uncertain control flow."""
from autoia import Interval, autoia


def piecewise_model(p):
    x = p[0]
    if x >= 0.0:
        return x * x + x
    return x * x - 2.0 * x


if __name__ == "__main__":
    result = autoia(piecewise_model, [Interval(-1.0, 1.0)])
    print(result)
    print("branch count:", len(result.branch_results))
    for i, (box, decisions, branch) in enumerate(
        zip(result.branch_boxes, result.branch_decisions, result.branch_results)
    ):
        print(
            f"branch {i}: box={box}, decisions={decisions}, "
            f"bounds=({branch.lower}, {branch.upper})"
        )
