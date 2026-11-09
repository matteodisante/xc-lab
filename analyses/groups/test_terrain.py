import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import terrain  # noqa: E402

N = 130  # cells: the window of 49 fits twice
CENTRE = N // 2
CELL = terrain.CELL_M


def at_centre(z):
    cls, relief, gentle, cover = terrain.classify(z.astype(np.float32), chunk=40)
    return int(cls[CENTRE, CENTRE]), relief[CENTRE, CENTRE], gentle[CENTRE, CENTRE], cover[CENTRE, CENTRE]


def test_flat_plane_is_plains():
    cls, relief, gentle, cover = at_centre(np.full((N, N), 120.0))
    assert (cls, relief, gentle, cover) == (1, 0, 1, pytest.approx(1))


def test_gentle_plane_with_relief_is_hills_then_low_mountains():
    column = np.arange(N) * CELL
    cls, relief, gentle, _ = at_centre(np.tile(0.015 * column, (N, 1)))  # 1.5%: 144 m over the window
    assert (cls, gentle) == (2, 1) and relief == pytest.approx(0.015 * 48 * CELL, rel=1e-4)
    cls, relief, gentle, _ = at_centre(np.tile(0.05 * column, (N, 1)))  # 5%: still gentle, 480 m
    assert (cls, gentle) == (3, 1)


def test_plane_of_ten_percent_is_not_plains():
    cls, relief, gentle, _ = at_centre(np.tile(0.10 * np.arange(N) * CELL, (N, 1)))
    assert cls >= 2 and gentle == 0 and relief == pytest.approx(0.10 * 48 * CELL, rel=1e-4)
    assert cls == 4  # 960 m over the window


def test_cone_of_1000_m_is_high_mountains_at_its_centre():
    y, x = np.mgrid[:N, :N]
    r = np.hypot(y - CENTRE, x - CENTRE) * CELL
    cls, relief, _, _ = at_centre(np.maximum(0, 1000 - r / 5))  # 1000 m at the top, 0 from 5 km
    assert cls == 4 and relief == pytest.approx(1000, abs=1)


def test_relief_thresholds():
    for height, expected in ((90, 1), (92, 2), (304, 2), (306, 3), (914, 3), (916, 4)):
        z = np.zeros((N, N))
        z[CENTRE, CENTRE] = height  # one peak: relief = height, steep on 8 cells only
        assert at_centre(z)[0] == expected


def test_coverage_below_half_is_unknown():
    z = np.full((N, N), 100.0, np.float32)
    z[:, CENTRE + 5:] = np.nan  # the last valid column has no slope: 28 columns of 49, 0.57
    cls, _, _, cover = at_centre(z)
    assert cover == pytest.approx(28 / 49, abs=0.01) and cls == 1
    z[:, CENTRE - 2:] = np.nan  # 22 columns of 49: 0.45
    cls, _, _, cover = at_centre(z)
    assert cover < 0.5 and cls == 0


def test_class_is_independent_of_the_stripes():
    rng = np.random.default_rng(1)
    z = np.cumsum(rng.normal(0, 8, (N, N)), axis=1).astype(np.float32)
    z[40:50, 60:70] = np.nan
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(terrain.classify(z, chunk=17), terrain.classify(z, chunk=500)))


def test_blocks_of_a_tile():
    z = np.arange(30 * 24, dtype=np.float64).reshape(30, 24)  # a tile that starts inside a block in both axes
    bx0, by_top, sums, counts = terrain.to_blocks(z, ix_left=1003, iy_top=2005)
    assert (bx0, by_top) == (1003 // 8, 2005 // 8)
    # the block of the first cell holds the columns 1003..1007 and the rows 2005..2000+... : 5 columns, 6 rows
    assert counts[0, 0] == 5 * 6 and sums[0, 0] == z[:6, :5].sum()
    assert counts[1, 1] == 64 and sums[1, 1] == z[6:14, 5:13].sum()
    assert counts.sum() == z.size


def test_block_with_more_than_half_nodata_is_nodata():
    z = np.full((16, 8), 10.0)
    z[:4, :] = np.nan  # the first block of rows: 32 of 64 missing, valid
    z[8:13, :] = np.nan  # the second: 40 missing
    sums, counts = terrain.to_blocks(z, 800, 799)[2:]
    mean = terrain.mean_of_blocks(sums, counts)
    assert mean[0, 0] == 10 and np.isnan(mean[1, 0])


def test_sea_blocks_are_the_blocks_all_at_zero():
    z = np.zeros((16, 16))
    z[0, 0] = 0.4  # one cell above 0 m in the block (0, 0)
    z[8:, 8:] = np.nan
    z[8:, 8:][:7] = 0  # block (1, 1): 56 cells at 0 m and 8 NoData
    z[0:8, 8:] = -0.4  # block (0, 1): below 0 m, not 0
    sums, counts = terrain.to_blocks(z, 800, 799)[2:]
    sea = terrain.sea_blocks(sums, counts)
    assert sea.tolist() == [[False, False], [True, False]]
    mean = terrain.mean_of_blocks(sums, counts)
    assert mean[1, 0] == 0 and mean[1, 1] == 0  # 56 valid zeros is a land block at 0 m, not the sea


def test_lookup_of_a_point():
    terrain_map = {"cls": np.array([[1, 2], [3, 4]], np.uint8), "valid": np.array([[True, True], [False, True]]),
                   "x0": 1000.0, "y0": 5000.0, "cell": 200.0}
    code = terrain.cell_class(terrain_map, [1050, 1250, 1050, 1250, 900, 1050], [4950, 4950, 4750, 4750, 4950, 5100])
    assert code.tolist() == [1, 2, terrain.OUTSIDE, 4, terrain.OUTSIDE, terrain.OUTSIDE]


def runs_table():
    # flight 7: class 1 on k 0..79, class 2 on k 80..99, a gap, class 4 on k 120..139; flight 8: class 5 on 0..9
    return pd.DataFrame({"flight": [7, 7, 7, 8], "k_start": [0, 80, 120, 0], "k_end": [79, 99, 139, 9], "cls": [1, 2, 4, 5]})


def test_counts_by_overlap_ignore_the_gap():
    counts = terrain.class_counts(runs_table(), [7, 7, 7, 8], [0, 70, 100, 0], [99, 130, 119, 9])
    assert counts[0].tolist() == [0, 80, 20, 0, 0, 0]
    assert counts[1].tolist() == [0, 10, 20, 0, 11, 0]
    assert counts[2].sum() == 0
    assert counts[3].tolist() == [0, 0, 0, 0, 0, 10]


def test_eighty_percent_rule():
    flight, k0, k1 = [7, 7, 7, 7, 8], [0, 0, 60, 1, 0], [99, 139, 139, 99, 9]
    counts = terrain.class_counts(runs_table(), flight, k0, k1)
    lab, binary = terrain.label(counts), terrain.label_binary(counts)
    # 80 of 100 (exactly the rule); 80 of 120; 20 + 20 + 20; 79 of 99; only unknown points
    assert lab.tolist() == [1, terrain.MIXED, terrain.MIXED, terrain.MIXED, terrain.UNKNOWN]
    # lowland is classes 1 and 2 together: 100 of 120 points, 40 of 60, 99 of 99
    assert binary.tolist() == [1, 1, terrain.MIXED, 1, terrain.UNKNOWN]


def test_plains_merged_into_hills_is_the_lowland_class():
    counts = terrain.class_counts(runs_table(), [7, 7], [0, 0], [79, 99])  # 80 plains | 80 plains and 20 hills
    assert terrain.label(counts).tolist() == [1, 1]
    assert terrain.label_merged(counts).tolist() == [2, 2]
    mixed = np.array([[0, 40, 40, 20, 0, 0]])  # 40 plains, 40 hills: neither alone has 80%, together 80%
    assert terrain.label(mixed)[0] == terrain.MIXED and terrain.label_merged(mixed)[0] == 2
    assert terrain.label_binary(mixed)[0] == 1


def test_piece_counts_take_points_not_time():
    counts = terrain.piece_counts(runs_table(), [7], [790.0], [21])  # grid points k = 79 .. 99
    assert counts[0].tolist() == [0, 1, 20, 0, 0, 0]
