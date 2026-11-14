import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import twoway  # noqa: E402


def test_sites_chain_takeoffs_of_the_same_day_only():
    lat = [45.0, 45.02, 45.04, 45.0, 46.5]  # 2.2 km steps: the first three chain, the fourth is the same spot on day 2
    lon = [6.0] * 5
    date = ["a", "a", "a", "b", "a"]
    b = twoway.sites(lat, lon, date)
    assert b[0] == b[1] == b[2] and len({b[0], b[3], b[4]}) == 3


def test_cell_codes_are_pilot_by_block_pairs():
    cell, pilot, block = twoway.cell_codes(np.array([0, 0, 1, 1, 0]), np.array([0, 1, 1, 1, 0]))
    assert cell.tolist() == [0, 1, 2, 2, 0] and pilot.tolist() == [0, 0, 1] and block.tolist() == [0, 1, 1]


def test_counts_stack_pilots_blocks_cells_and_keep_the_total():
    pilot, block = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2])
    c = twoway.counts(pilot, block, np.random.default_rng(0), size=60)
    assert c.shape == (180, 4)
    assert (c[60:120][:, 1] == c[60:120][:, 2]).all()  # cells 1 and 2 share a block
    assert (c[:60][:, 0] == c[:60][:, 1]).all()  # cells 0, 1 share a pilot
    assert (c[120:].sum(axis=1) == 4).all()  # the cell draws are 4 draws of 4 cells


def test_mean_draws_point_is_the_equal_weight_mean_and_all_ones_reproduce_it():
    values = np.array([[1.0, 5.0], [3.0, np.nan], [5.0, 7.0]])
    cell = np.array([0, 0, 1])
    point, draws = twoway.mean_draws(values, cell, np.ones((6, 2)))
    assert point.tolist() == [3.0, 6.0] and np.allclose(draws, point)
    point, draws = twoway.mean_draws(values, cell, np.array([[0, 2], [1, 0]]))
    assert draws[0] == pytest.approx([5.0, 7.0]) and draws[1] == pytest.approx([2.0, 5.0])


def test_variance_and_band_follow_the_formula_and_a_constant_has_no_band():
    draws = np.vstack([np.full((10, 1), 1.0) + np.arange(10)[:, None], np.zeros((10, 1)), np.zeros((10, 1))])
    vp = np.var(np.arange(10))
    assert twoway.variance(draws)[0] == pytest.approx(vp)  # V_p + V_b - V_c = vp, and max(., V_p, V_b)
    lo, hi = twoway.band(np.array([2.0]), draws)
    assert hi[0] - lo[0] == pytest.approx(2 * twoway.Z * np.sqrt(vp))
    assert twoway.band(np.array([1.0]), np.ones((30, 1)))[0] == [1.0]


def test_the_paired_difference_of_a_population_with_itself_has_no_band():
    rng = np.random.default_rng(3)
    stacked = twoway.counts(np.arange(20), np.arange(20) // 2, rng, size=40)
    values = rng.normal(size=(20, 3))
    point, draws = twoway.mean_draws(values, np.arange(20), stacked)
    lo, hi = twoway.band(point - point, draws - draws)
    assert lo == [0.0] * 3 and hi == [0.0] * 3
