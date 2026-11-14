import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import terrain  # noqa: E402


def dem(z, cell=200):
    return {"z": np.asarray(z, np.float32), "x0": 1000.0, "y0": 5000.0, "cell": float(cell)}


def test_constants_are_the_users_rule():
    assert terrain.EDGES_M == (300.0, 800.0, 1500.0) and terrain.MIN_ON_DEM == 0.5
    assert terrain.CLASSES == ("plains", "hills", "low mountains", "high mountains")


def test_blocks_average_the_cells_that_hold_their_centres():
    z = np.arange(16 * 16, dtype=float).reshape(16, 16)  # two blocks by two, 25 m cells
    bx0, by_top, s, c = terrain.to_blocks(z, ix_left=0, iy_top=15)
    assert (bx0, by_top) == (0, 1) and s.shape == (2, 2) and (c == 64).all()
    assert terrain.block_means(s, c)[0, 0] == pytest.approx(z[:8, :8].mean())
    assert terrain.block_means(s, c)[1, 1] == pytest.approx(z[8:, 8:].mean())


def test_blocks_with_more_than_half_nodata_or_all_zero_have_no_ground():
    z = np.full((8, 8), 100.0)
    z[:4] = np.nan  # exactly half valid: kept
    _, _, s, c = terrain.to_blocks(z, 0, 7)
    assert terrain.block_means(s, c)[0, 0] == pytest.approx(100)
    z[4, :] = np.nan  # fewer than half
    _, _, s, c = terrain.to_blocks(z, 0, 7)
    assert np.isnan(terrain.block_means(s, c)[0, 0])
    _, _, s, c = terrain.to_blocks(np.zeros((8, 8)), 0, 7)  # sea delivered as 0 m
    assert np.isnan(terrain.block_means(s, c)[0, 0])
    _, _, s, c = terrain.to_blocks(np.where(np.arange(64).reshape(8, 8) == 0, 5.0, 0.0), 0, 7)
    assert terrain.block_means(s, c)[0, 0] == pytest.approx(5 / 64)  # a block with relief is not sea


def test_ground_reads_the_block_under_a_point_and_nan_outside():
    d = dem([[100, 200, np.nan], [300, 400, 500]])  # rows go south
    x = np.array([1050.0, 1250.0, 1450.0, 1050.0, 999.0, 1650.0, 1050.0])
    y = np.array([4950.0, 4950.0, 4950.0, 4750.0, 4950.0, 4950.0, 5001.0])
    got = terrain.ground(d, x, y)
    assert got[[0, 1, 3]].tolist() == [100, 200, 300]
    assert np.isnan(got[2]) and np.isnan(got[4:]).all()  # NaN block, west, east and north of the map


def test_median_is_over_the_points_on_the_dem():
    m, n, on = terrain.median_ground(np.array([100, 200, np.nan, 1000, np.nan]))
    assert (m, n, on) == (200.0, 5, 3)
    assert np.isnan(terrain.median_ground(np.full(4, np.nan))[0])


def test_classes_have_the_edges_of_the_rule():
    med = np.array([0.0, 299.9, 300.0, 799.9, 800.0, 1499.9, 1500.0, 3000.0])
    got = terrain.classify(med, np.full(8, 100), np.full(8, 100))
    assert got.tolist() == ["plains", "plains", "hills", "hills", "low mountains", "low mountains", "high mountains",
                            "high mountains"]


def test_a_flight_with_less_than_half_of_its_points_on_the_dem_is_unknown():
    got = terrain.classify([500, 500, 500, 500, np.nan], [10, 10, 10, 0, 10], [4, 5, 10, 0, 0])
    assert got.tolist() == ["unknown", "hills", "hills", "unknown", "unknown"]


def test_class_of_a_synthetic_track_with_a_known_median():
    # ground rising by 200 m per block from 100 m; the last point is off the map
    d = dem(np.tile(np.arange(10) * 200.0 + 100, (2, 1)))
    x = 1000.0 + np.array([50, 450, 850, 1250, 1450, 1650, 1850, 1250, 1450, 1650, 5000.0])
    e = terrain.ground(d, x, np.full(len(x), 4900.0))
    median, n, on = terrain.median_ground(e)
    assert (n, on) == (11, 10) and median == 1400.0
    assert terrain.classify([median], [n], [on])[0] == "low mountains"
    assert terrain.classify([median + 100], [n], [on])[0] == "high mountains"


def test_grid_points_are_per_segment_on_the_10_s_grid():
    fixes = pd.DataFrame({"segment_id": [0, 0, 0, 1, 1], "t": [3.0, 12.0, 31.0, 100.0, 125.0],
                          "E": [0.0, 9.0, 28.0, 0.0, 25.0], "N": 0.0, "z": 0.0})
    (k0, e0, _, _), (k1, e1, _, _) = list(terrain.grid_points(fixes))
    assert k0.tolist() == [1, 2, 3] and e0.tolist() == pytest.approx([7.0, 17.0, 27.0])
    assert k1.tolist() == [10, 11, 12] and e1.tolist() == pytest.approx([0.0, 10.0, 20.0])
