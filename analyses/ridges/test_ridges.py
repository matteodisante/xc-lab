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
