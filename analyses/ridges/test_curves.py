"""Strokes, pruning, bridges and resampling on small networks."""

import numpy as np

import curves


def network(paths):
    """Edges and vertex coordinates (25 m grid) of polylines given as lists of (col, row) corners."""
    ids, edges = {}, []
    for p in paths:
        for u, v in zip(p[:-1], p[1:]):
            edges.append([ids.setdefault(u, len(ids)), ids.setdefault(v, len(ids))])
    xy = np.array(sorted(ids, key=ids.get), float) * 25.0
    return np.array(edges), xy


def lengths(v, off):
    return sorted(int(b - a) - 1 for a, b in zip(off[:-1], off[1:]))


def test_t_junction_gives_two_strokes():
    main = [(i, 0) for i in range(61)]
    branch = [(30, j) for j in range(21)]
    e, xy = network([main, branch])
    v, off = curves.strokes(e[:, 0], e[:, 1], xy)
    assert lengths(v, off) == [20, 60]


def test_short_spur_is_pruned():
    main = [(i, 0) for i in range(61)]
    spur = [(30, j) for j in range(3)]
    e, xy = network([main, spur])
    v, off = curves.strokes(e[:, 0], e[:, 1], xy)
    assert lengths(v, off) == [60]


def test_deflection_limit_decides_the_pairing():
    a = [(i, 0) for i in range(21)]
    b = [(-i, i) for i in range(21)]
    c = [(-i, -i) for i in range(21)]
    e, xy = network([a, b, c])
    assert len(curves.strokes(e[:, 0], e[:, 1], xy, max_deflection=30.0)[1]) - 1 == 3
    assert len(curves.strokes(e[:, 0], e[:, 1], xy)[1]) - 1 == 2


def test_bridge_adds_flat_stretch_between_two_pieces():
    left = np.array([[0, 1], [1, 2]], np.int32)
    right = np.array([[10, 11], [11, 12]], np.int32)
    flat = np.array([[2, 3], [3, 10]], np.int32)
    out = curves.bridge(np.r_[left, right], flat)
    assert len(out) == 6
    assert len(curves.bridge(np.r_[left, right], flat, max_edges=1)) == 4
    spur = np.array([[2, 3], [3, 4]], np.int32)  # touches the network once: not added
    assert len(curves.bridge(np.r_[left, right], spur)) == 4


def test_resample_straight_line_every_5_m():
    p = np.c_[np.arange(0, 1001, 25.0), np.zeros(41)]
    q = curves.smooth_resample(p)
    assert np.allclose(np.diff(q[:, 0]), 5.0, atol=1e-6) and abs(q[-1, 0] - 1000) < 5
    assert np.abs(q[:, 1]).max() < 1e-6
