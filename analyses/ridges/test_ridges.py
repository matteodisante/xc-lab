"""Synthetic DEMs: a ridge between two valleys gives one crest; a pit is filled; a tilted plane gives none."""

import numpy as np

import ridges


def two_valleys(h=120, w=201, slope=4.0, fall=1.0):
    """Valleys at columns 50 and 150 falling south, a ridge at column 100 (and the sides at 0 and 200)."""
    x = np.arange(w)
    dist = np.minimum(np.abs(x - 50), np.abs(x - 150))
    z = 100.0 + slope * dist[None, :] + fall * (h - np.arange(h))[:, None]
    return z.astype(np.float32)


def run(z, valley_km2=0.01):
    lid, sz, tin, tout = ridges.drainage(z, valley_km2=valley_km2)
    return lid.reshape(z.shape), ridges.crest_edges(z, lid, sz, tin, tout)


def test_ridge_between_two_valleys():
    z = two_valleys()
    lid, (acc, flat) = run(z)
    w = z.shape[1]
    cols = (acc % (w + 1))[:, 0]
    rows = acc[:, 0] // (w + 1)
    # all crest sides on the corner line between columns 99 and 100 (corner column 100) except near the borders
    inner = (rows > 10) & (rows < 100)
    assert inner.sum() > 80
    assert set(cols[inner]) <= {100, 101}
    # the vertical sides of this line are consecutive
    assert len(np.unique(acc[inner])) >= 80


def test_valleys_are_two_links_not_related():
    z = two_valleys()
    lid, _ = run(z)
    assert lid[100, 60] != lid[100, 140] and lid[100, 60] > 0 and lid[100, 140] > 0
    assert lid[100, 90] != lid[100, 110]  # either side of the ridge


def test_tilted_plane_has_no_crest():
    z = (1000 - 3.0 * np.add.outer(np.arange(100), np.arange(100))).astype(np.float32)
    _, (acc, _) = run(z)
    r, c = acc[:, 0] // 101, acc[:, 0] % 101
    # the border passes the cross-section test (the ground beyond it is unknown); the interior does not
    assert not ((r > 7) & (r < 93) & (c > 7) & (c < 93)).any()


def test_pit_is_filled_and_drains():
    z = two_valleys()
    z[60:64, 40:44] -= 30  # a pit in the left valley floor
    filled, d, order = ridges.flood(z)
    f = filled.reshape(z.shape)
    assert f[61, 41] > z[61, 41] + 1
    assert order.size == z.size and len(set(order.tolist())) == z.size
    # every cell reaches an outlet following the directions
    h, w = z.shape
    for start in (61 * w + 41, 10 * w + 10, 110 * w + 190):
        i, steps = start, 0
        while d[i] >= 0 and steps < 10**5:
            i += ridges.DR[d[i]] * w + ridges.DC[d[i]]
            steps += 1
        assert d[i] == ridges.OUTLET


def test_low_ridge_fails_cross_section():
    z = two_valleys(slope=0.5)  # 3 m below the crest 6 cells away
    _, (acc, flat) = run(z)
    assert len(acc) == 0 and len(flat) > 50


# flats resolved by the gradients of Barnes, Lehman and Mulla (2014)

def follow(d, w, r, c, limit=10**5):
    """Cells visited from (r, c) following the directions until an outlet."""
    path, i = [], r * w + c
    while d[i] >= 0 and len(path) < limit:
        path.append(i)
        i += ridges.DR[d[i]] * w + ridges.DC[d[i]]
    assert d[i] == ridges.OUTLET
    return path + [i]


def blm(z):
    filled, d, order = ridges.route(z, "blm")
    return filled, d, order


def test_flat_valley_floor_drains_to_its_outlet():
    h, w = 40, 41
    z = np.full((h, w), 300.0)
    z[3:30, 10:31] = 100.0  # floor
    z[30:, 20] = 99.0 - np.arange(h - 30)  # one gully south of the floor
    z = z.astype(np.float32)
    filled, d, order = blm(z)
    assert (filled == z.ravel()).all() and order.size == z.size
    for r in range(3, 30):
        for c in range(10, 31):
            path = follow(d, w, r, c)
            assert (30 * w + 20) in path and path[-1] // w == h - 1


def test_flat_col_splits_at_its_midline():
    h, w = 41, 91
    r, c = np.mgrid[:h, :w]
    z = 100.0 + 20.0 * np.maximum(np.abs(r - 20) - 2, 0)  # band of rows 18-22 between walls
    z = z - 2.0 * np.maximum(30 - c, 0) - 2.0 * np.maximum(c - 59, 0)  # falls away at both ends
    z[:3], z[-3:] = 500.0, 500.0
    z = z.astype(np.float32)
    _, d, _ = blm(z)
    west = [follow(d, w, 20, cc)[-1] % w == 0 for cc in range(31, 59)]
    divide = 30.5 + sum(west)  # between the last column draining west and the first draining east
    assert west == sorted(west, reverse=True)
    assert abs(divide - 44.5) <= 1.0  # the middle of the low edges, columns 30 and 59


def test_filled_depression_drains_to_its_spill_point():
    h, w = 60, 80
    r, c = np.mgrid[:h, :w]
    z = 200.0 - 2.0 * c + 0.3 * r
    z[20:30, 20:30] -= 20.0
    z = z.astype(np.float32)
    filled, d, _ = blm(z)
    f = filled.reshape(h, w)
    spill = 19 * w + 30  # lowest cell of the rim of the pit
    assert f[25, 25] == z[19, 30]
    for rr in range(20, 30):
        for cc in range(20, 30):
            path = follow(d, w, rr, cc)
            assert spill in path
            k = path.index(spill)
            assert (filled[path[max(k - 1, 0):k + 1]] == z[19, 30]).all() and filled[path[k + 1]] < z[19, 30]


def increments(z):
    filled = ridges.flood(z)[0]
    d = ridges._exits(z, filled)
    return filled, d, ridges._increments(filled, d, z.shape[1])


def several_flats(seed=3):
    """A rough surface with two flat valley floors, a flat col, a plateau and a filled pit; no exact slope ties
    outside the flats because of the random relief."""
    rng = np.random.default_rng(seed)
    h, w = 90, 120
    r, c = np.mgrid[:h, :w]
    z = 300.0 + 0.7 * r - 0.9 * c + rng.random((h, w))
    z[10:25, 10:50] = 150.0 + 0.0 * z[10:25, 10:50]  # plateau-like floor
    z[40:48, 30:90] = 120.0  # long floor
    z[60:70, 60:70] -= 80.0  # pit
    z[75:85, 90:110] = z[75:85, 90:110].min()  # another flat
    return z.astype(np.float32)


def test_directions_do_not_depend_on_the_orientation():
    z = several_flats()
    filled0, d0, inc0 = increments(z)
    assert (d0 == ridges.UNSEEN).sum() > 300 and (filled0.reshape(z.shape) != z).any()
    d0 = ridges.blm_directions(z, filled0)[0].reshape(z.shape)
    # direction k becomes k' under the map of the offsets
    off = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ridges.DR, ridges.DC))}
    for name, f, g in (("transpose", lambda a: a.T, lambda dr, dc: (dc, dr)),
                       ("flip_rows", lambda a: a[::-1], lambda dr, dc: (-dr, dc)),
                       ("flip_cols", lambda a: a[:, ::-1], lambda dr, dc: (dr, -dc))):
        zt = np.ascontiguousarray(f(z))
        filledt, _, inct = increments(zt)
        back = lambda a: f(a.reshape(zt.shape))
        assert (back(filledt) == filled0.reshape(z.shape)).all(), name
        assert (back(inct) == inc0.reshape(z.shape)).all(), name  # the increments are covariant
        dt = back(ridges.blm_directions(zt, filledt)[0])
        # the transformed direction of every cell; where several neighbours tie it may differ, never in slope
        perm = np.array([off[g(int(a), int(b))] for a, b in zip(ridges.DR, ridges.DC)])
        dt = np.where(dt >= 0, perm[np.maximum(dt, 0)], dt)  # codes back in the original frame
        diff = d0 != dt
        fl, inc = filled0.reshape(z.shape), inc0.reshape(z.shape)
        for rr, cc in zip(*np.nonzero(diff)):
            def val(k, rr=rr, cc=cc):  # slope of direction k on (filled, increments)
                r2, c2 = rr + ridges.DR[k], cc + ridges.DC[k]
                s = (fl[rr, cc] - fl[r2, c2]) or (inc[rr, cc] - inc[r2, c2])
                return s / (1.0 if k % 2 == 0 else 1.41421356)
            assert abs(val(d0[rr, cc]) - val(dt[rr, cc])) < 1e-6, (name, rr, cc)
        assert diff.mean() < 0.01


def test_flats_directions_agree_with_the_flood_outside_the_flats():
    z = several_flats()
    filled, d_flood, _ = ridges.route(z, "flood")
    _, d_blm, _ = ridges.route(z, "blm")
    _, d, inc = increments(z)
    assert (d_flood == d_blm)[inc == 0].mean() > 0.999
