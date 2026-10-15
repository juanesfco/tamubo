"""
Tests for ExactBO's box splitting (``tamubo.exactbo.partition.split_boxes``).

Every split must tile its parent exactly (children inside the parent, volumes
summing to it, every point of the parent in exactly one child), whichever
dimensions are split. Also checks which dimensions ``split_ratio`` and
``min_width`` select, that NumPy and CuPy agree, and that ExactBO's partition
loop, which rebuilds each partition's boxes from their parents in chunks, does
not depend on the chunk size:

    python -m pytest experiments/exactbo/experiment5/test_partition.py
"""
from __future__ import annotations

import numpy as np
import pytest

import tamubo.exactbo.partition as partition
import tamubo.exactbo.run as run
from tamubo.exactbo.partition import split_boxes, split_child_counts
from tamubo.utils import has_cupy, to_numpy

BACKENDS = ["numpy"] + (["cupy"] if has_cupy() else [])


def _xp(backend):
    if backend == "cupy":
        import cupy

        return cupy
    return np


def _split(backend, L, U, mask, scale, **kwargs):
    xp = _xp(backend)
    n, d = L.shape
    out = split_boxes(xp.asarray(L), xp.asarray(U), xp.asarray(mask), xp.asarray(scale), n, d,
                      return_counts=True, backend=backend, **kwargs)
    return tuple(to_numpy(a) for a in out)


@pytest.fixture(scope="module")
def boxes():
    rng = np.random.default_rng(0)
    n, d = 200, 5
    L = rng.random((n, d)) * 0.5
    U = L + 0.05 + 0.4 * rng.random((n, d))
    mask = rng.random(n) < 0.6
    scale = np.array([0.25, 0.3, 20.0, 35.0, 18.0])  # two short, three long length scales
    return L, U, mask, scale


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("split_ratio", [0.0, 1.0 / 3.0, 1.0])
def test_children_tile_parent(backend, boxes, split_ratio):
    L, U, mask, scale = boxes
    d = L.shape[1]
    cL, cU, counts = _split(backend, L, U, mask, scale, keep_inactive=False, split_ratio=split_ratio,
                            min_width=np.full(d, 1e-5))
    assert counts.shape == (int(mask.sum()),)
    assert np.all(counts % 2 == 1) and counts.sum() == cL.shape[0]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    rng = np.random.default_rng(1)
    for k, p in enumerate(np.flatnonzero(mask)):
        ch_L, ch_U = cL[offsets[k]:offsets[k + 1]], cU[offsets[k]:offsets[k + 1]]
        assert np.all(ch_L >= L[p] - 1e-15) and np.all(ch_U <= U[p] + 1e-15)
        np.testing.assert_allclose(np.prod(ch_U - ch_L, axis=1).sum(), np.prod(U[p] - L[p]), rtol=1e-12)
        pts = L[p] + rng.random((300, d)) * (U[p] - L[p])
        inside = np.all((pts[:, None, :] >= ch_L) & (pts[:, None, :] <= ch_U), axis=-1).sum(axis=1)
        assert np.all(inside == 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ratio_zero_splits_every_dimension(backend, boxes):
    L, U, mask, scale = boxes
    d = L.shape[1]
    _, _, counts = _split(backend, L, U, mask, scale, keep_inactive=False)
    assert np.all(counts == 2 * d + 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_longest_side_rule_in_scaled_units(backend):
    # Unit cube with length scales (1, 1, 10): dims 0 and 1 tie for widest in
    # scaled units, so ratio 1 splits exactly those two (5 children).
    L, U, scale = np.zeros((1, 3)), np.ones((1, 3)), np.array([1.0, 1.0, 10.0])
    cL, cU, counts = _split(backend, L, U, np.array([True]), scale, keep_inactive=False, split_ratio=1.0)
    assert counts.tolist() == [5]
    widths = cU - cL
    assert np.all(widths[:, 2] == 1.0)  # the flat dimension is never split
    np.testing.assert_allclose(widths[-1], [1 / 3, 1 / 3, 1.0])  # center box comes last


@pytest.mark.parametrize("backend", BACKENDS)
def test_min_width_stops_refinement(backend):
    # Dimension 0 is widest in scaled units but already narrower than min_width,
    # so dimension 1 is split instead; once every dimension is narrower, all may split.
    L = np.zeros((2, 2))
    U = np.array([[1e-6, 0.5], [1e-6, 1e-6]])
    scale = np.array([1e-9, 1.0])
    cL, cU, counts = _split(backend, L, U, np.array([True, True]), scale, keep_inactive=False,
                            split_ratio=1.0, min_width=np.full(2, 1e-5))
    assert counts.tolist() == [3, 3]
    np.testing.assert_allclose((cU - cL)[:3, 1], [0.5 / 3] * 3)  # first box: dimension 1 trisected
    np.testing.assert_allclose((cU - cL)[3:, 0], [1e-6 / 3] * 3)  # second box: widest scaled (dim 0)


def _cupy_layout(L_out, U_out, counts, mask):
    """Reorder NumPy keep_inactive output (inactive boxes left in place) as CuPy's (children first)."""
    children, inactive, pos, k = [], [], 0, 0
    for active in mask:
        if active:
            children.extend(range(pos, pos + counts[k]))
            pos, k = pos + counts[k], k + 1
        else:
            inactive.append(pos)
            pos += 1
    order = np.array(children + inactive, dtype=np.int64)
    return L_out[order], U_out[order]


@pytest.mark.parametrize("split_ratio", [0.0, 1.0 / 3.0, 1.0])
@pytest.mark.parametrize("keep", [False, True])
def test_numpy_and_cupy_agree(boxes, split_ratio, keep):
    if not has_cupy():
        pytest.skip("needs a GPU")
    L, U, mask, scale = boxes
    d = L.shape[1]
    kwargs = dict(keep_inactive=keep, split_ratio=split_ratio, min_width=np.full(d, 1e-5))
    L_np, U_np, counts_np = _split("numpy", L, U, mask, scale, **kwargs)
    L_cp, U_cp, counts_cp = _split("cupy", L, U, mask, scale, **kwargs)
    np.testing.assert_array_equal(counts_np, counts_cp)
    if keep:
        L_np, U_np = _cupy_layout(L_np, U_np, counts_np, mask)
    np.testing.assert_array_equal(L_np, L_cp)
    np.testing.assert_array_equal(U_np, U_cp)


@pytest.mark.parametrize("split_ratio", [0.0, 1.0 / 3.0])
@pytest.mark.parametrize("keep", [False, True])
def test_chunked_split_matches_single_chunk(boxes, monkeypatch, split_ratio, keep):
    # The GPU path builds children in chunks of parents; force ~7 parents per chunk.
    if not has_cupy():
        pytest.skip("needs a GPU")
    L, U, mask, scale = boxes
    d = L.shape[1]
    kwargs = dict(keep_inactive=keep, split_ratio=split_ratio, min_width=np.full(d, 1e-5))
    whole = _split("cupy", L, U, mask, scale, **kwargs)
    monkeypatch.setattr(partition, "_SPLIT_CHUNK_BYTES", 2 * (2 * d + 1) * d * 8 * 7)
    chunked = _split("cupy", L, U, mask, scale, **kwargs)
    for a, b in zip(whole, chunked):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("split_ratio", [0.0, 1.0 / 3.0, 1.0])
@pytest.mark.parametrize("chunked", [False, True])
def test_child_counts_match_split(backend, boxes, split_ratio, chunked, monkeypatch):
    L, U, mask, scale = boxes
    d = L.shape[1]
    if chunked:  # ~7 boxes per chunk
        monkeypatch.setattr(partition, "_SPLIT_CHUNK_BYTES", 4 * d * 8 * 7)
    _, _, counts = _split(backend, L, U, mask, scale, keep_inactive=False, split_ratio=split_ratio,
                          min_width=np.full(d, 1e-5))
    xp = _xp(backend)
    only = split_child_counts(xp.asarray(L[mask]), xp.asarray(U[mask]), xp.asarray(scale), split_ratio=split_ratio,
                              min_width=xp.asarray(np.full(d, 1e-5)), backend=backend)
    np.testing.assert_array_equal(to_numpy(only), counts)


@pytest.mark.parametrize("split_ratio", [0.0, 1.0 / 3.0])
def test_partition_loop_independent_of_frontier_chunks(monkeypatch, split_ratio):
    # 97 boxes per chunk (prime, so parents' children straddle chunk boundaries)
    # against one chunk per partition: same partitions and targets, same result up
    # to rounding (sampling fewer boxes per call changes GEMM shapes, so EI values
    # can differ in the last bits).
    if not has_cupy():
        pytest.skip("needs a GPU")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment3"))
    from common import make_gp

    gp, X, bounds = make_gp("problem5d", 32, 0)
    d = X.shape[1]

    def search():
        return run.exactbo_partitioning(X, bounds, np.full(d, 1e-5), 0.1, gp, 6 if split_ratio == 0.0 else 1000,
                                        backend="cupy", n_gpus=1, split_ratio=split_ratio, validation=False,
                                        logMask=True)

    whole = search()
    monkeypatch.setattr(run, "_FRONTIER_CHUNK_BOXES", 97)
    chunked = search()
    assert max(p["n_boxes"] for p in whole.log["partitions"]) > 10 * 97
    for a, b in zip(whole.log["partitions"], chunked.log["partitions"], strict=True):
        assert [a[k] for k in ("n_boxes", "n_analyze", "n_active", "n_target")] == \
            [b[k] for k in ("n_boxes", "n_analyze", "n_active", "n_target")]
        np.testing.assert_allclose([a["max_ei_hi"], a["best"]], [b["max_ei_hi"], b["best"]], rtol=1e-12)
    np.testing.assert_allclose(whole.X, chunked.X, rtol=0, atol=1e-12)


def test_partition_loop_independent_of_gpu_count(monkeypatch):
    # The frontier's parents are sharded over the GPUs; force sharding from the
    # first partition and compare against one GPU (same partitions and targets,
    # same result up to rounding).
    if not has_cupy():
        pytest.skip("needs a GPU")
    import cupy

    if cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs 2+ GPUs")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment3"))
    from common import make_gp

    gp, X, bounds = make_gp("problem5d", 32, 0)
    d = X.shape[1]

    def search(n_gpus):
        return run.exactbo_partitioning(X, bounds, np.full(d, 1e-5), 0.1, gp, 1000, backend="cupy", n_gpus=n_gpus,
                                        validation=False, logMask=True)

    one = search(1)
    monkeypatch.setattr(run, "_MIN_WORK_PER_GPU", 1.0)
    monkeypatch.setattr(run, "_FRONTIER_CHUNK_BOXES", 97)
    many = search(None)
    assert many.log["converged"] and one.log["converged"]
    for a, b in zip(one.log["partitions"], many.log["partitions"], strict=True):
        assert [a[k] for k in ("n_boxes", "n_analyze", "n_active", "n_target")] == \
            [b[k] for k in ("n_boxes", "n_analyze", "n_active", "n_target")]
        np.testing.assert_allclose([a["max_ei_hi"], a["best"]], [b["max_ei_hi"], b["best"]], rtol=1e-12)
    np.testing.assert_allclose(one.X, many.X, rtol=0, atol=1e-12)
