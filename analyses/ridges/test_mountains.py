"""Mountain area on synthetic elevation grids built from 1 km cells."""

import numpy as np

import mountains


def dem(h):
    """25 m DEM whose 1 km mean elevations are h."""
    return np.kron(h, np.ones((mountains.BLOCK, mountains.BLOCK))).astype(np.float32)


def test_lowland_is_not_mountain_and_high_ground_is():
    h = np.full((40, 40), 200.0)
    h[10:30, 10:30] = 3000.0
    m = mountains.mountain_area(dem(h))
    assert m[20, 20] and not m[2, 2]
    assert m.sum() >= 400


def test_enclosed_basin_is_added_and_small_massif_dropped():
    h = np.full((60, 60), 200.0)
    h[5:35, 5:35] = 3000.0
    h[18:21, 18:21] = 200.0  # 9 km2 basin inside the massif
    h[45:50, 45:50] = 3000.0  # a second, smaller massif
    m = mountains.mountain_area(dem(h))
    assert m[19, 19] and m[10, 10]
    assert not m[47, 47]


def test_open_valley_is_not_filled():
    h = np.full((60, 60), 200.0)
    h[5:35, 5:35] = 3000.0
    h[18:21, 5:20] = 200.0  # a valley open to the outside
    m = mountains.mountain_area(dem(h))
    assert not m[19, 8]


def test_local_range_rule_picks_hills_of_relief_over_300_m():
    h = np.full((40, 40), 400.0)
    h[::2, ::2] = 800.0  # relief 400 m within a few km, elevation 400-800 m: class 6
    assert mountains.classes(h)[20, 20]
    assert not mountains.classes(np.full((40, 40), 400.0))[20, 20]
