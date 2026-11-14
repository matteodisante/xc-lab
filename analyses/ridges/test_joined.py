"""Joins, trees and paths on small synthetic networks."""

import numpy as np

import joined

X0, Y0 = -1000.0, 2000.0


def line(x0, y0, x1, y1):
    n = int(np.hypot(x1 - x0, y1 - y0) // 5)
    t = np.linspace(0, 1, n + 1)[:, None]
    return np.array([x0, y0]) + t * np.array([x1 - x0, y1 - y0])


def pack(curves):
    off = np.r_[0, np.cumsum([len(c) for c in curves])]
    return np.concatenate(curves), off


def raster(valley=None):
    h = np.full((200, 300), 100, np.int16)
    if valley:  # columns of the raster where the ground is at the stream level
        h[:, valley[0]:valley[1]] = 0
    return h


def paths(net):
    emap = joined.edge_map(net)
    return [(tr["id"], *joined.path_polyline(net, tr, j, emap)) for tr in joined.trees(net) for j in tr["tips"]]


def test_gap_is_joined_and_gives_one_path():
    xy, off = pack([line(0, 0, 2000, 0), line(2300, 0, 4500, 0)])
    net = joined.Net(xy, off, np.zeros((0, 2), int), np.zeros((0, 2)))
    assert joined.make_joins(net, raster(), X0, Y0) == 1
    p = paths(net)
    assert len(p) == 1 and abs(len(p[0][1]) * 5 - 4500) < 30
    assert np.allclose(np.abs(np.diff(p[0][1][:, 0])), 5, atol=0.5)


def test_valley_blocks_the_join():
    xy, off = pack([line(0, 0, 2000, 0), line(2300, 0, 4500, 0)])
    net = joined.Net(xy, off, np.zeros((0, 2), int), np.zeros((0, 2)))
    assert joined.make_joins(net, raster(valley=(121, 131)), X0, Y0) == 0  # x = 2025 .. 2275 m
    assert len(paths(net)) == 0 or len({t for t, _, _ in paths(net)}) == 2


def test_narrow_stream_between_two_crests_blocks_the_join():
    xy, off = pack([line(0, 0, 2000, 0), line(2300, 0, 4500, 0)])
    net = joined.Net(xy, off, np.zeros((0, 2), int), np.zeros((0, 2)))
    h = raster()
    h[:, 130] = 0  # a one-cell stream at x = 2250 m; the cells beside it stay high
    assert joined.make_joins(net, h, X0, Y0) == 0


def test_flat_col_between_two_crests_is_crossed():
    xy, off = pack([line(0, 0, 2000, 0), line(2300, 0, 4500, 0)])
    net = joined.Net(xy, off, np.zeros((0, 2), int), np.zeros((0, 2)))
    h = raster()
    h[:, 121:131] = 25  # a col 250 m wide, 25 m above the streams
    assert joined.make_joins(net, h, X0, Y0) == 1


def test_join_must_be_close_and_aligned():
    xy, off = pack([line(0, 0, 2000, 0), line(2900, 700, 2900, 1500)])  # too far, and off the cone of both ends
    net = joined.Net(xy, off, np.zeros((0, 2), int), np.zeros((0, 2)))
    assert joined.make_joins(net, raster(), X0, Y0) == 0


def test_junction_gives_two_paths_from_the_root():
    xy, off = pack([line(0, 0, 2000, 0), line(1000, 0, 1000, 1500)])
    net = joined.Net(xy, off, np.array([[0, 1]]), np.array([[1000.0, 0.0]]))
    p = paths(net)
    assert len(p) == 2 and len({t for t, _, _ in p}) == 1
    lengths = [len(q) * 5 for _, q, _ in p]
    assert max(lengths) > 2450 and all(1950 < x < 2550 for x in lengths)


def test_join_snaps_to_a_dead_end():
    a = line(0, 0, 2000, 0)
    b = line(2150, 60, 4000, 60)  # its dead end is within 200 m of the point of a nearest to the join
    xy, off = pack([a, b])
    net = joined.Net(xy, off, np.zeros((0, 2), int), np.zeros((0, 2)))
    assert joined.make_joins(net, raster(), X0, Y0) == 1
    assert any(e[2] == 2 for e in net.edges)


def test_cells_crossed_includes_every_touched_cell():
    r, c = joined.cells_crossed(np.array([0.0, 0.0]), np.array([100.0, 0.0]), 0.0, 0.0)
    assert set(c) == {0, 1, 2, 3, 4} and set(r) == {0}
    # a diagonal through cell corners touches the cells on both sides of each corner
    r, c = joined.cells_crossed(np.array([0.0, 0.0]), np.array([75.0, -75.0]), 0.0, 0.0)
    assert {(0, 1), (1, 0)} <= set(zip(r.tolist(), c.tolist()))


def test_diagonal_stream_cannot_be_slipped_through():
    xy, off = pack([line(0, 0, 2000, 0), line(2300, 0, 4500, 0)])
    net = joined.Net(xy, off, np.zeros((0, 2), int), np.zeros((0, 2)))
    h = raster()
    for i in range(30):  # a stream of diagonal cells across the line at row 80 (y = 0)
        h[70 + i, 140 - i + 0] = 0
    assert joined.make_joins(net, h, X0, Y0) == 0
