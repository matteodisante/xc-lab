"""Tile layout of the IGN DEM (skipped when the disk is not mounted)."""

import numpy as np
import pytest
import rasterio

import geo

pytestmark = pytest.mark.skipif(not geo.DEM.exists(), reason="IGN DEM not mounted")


def test_region_boxes_cover_the_departments():
    assert geo.region_box("pyrenees") == (300, 725, 6125, 6325)
    x0, x1, y0, y1 = geo.region_box("alps")
    assert (x0, x1, y0, y1) == (825, 1100, 6200, 6600)


def test_cell_centres_sit_on_multiples_of_25_m():
    f = geo.tile_index({"D065"})
    (xk, yk), paths = next(iter(f.items()))
    with rasterio.open(paths[0]) as src:
        assert (src.bounds.left + 12.5) % 25 == 0 and (src.bounds.top - 12.5) % 25 == 0
        assert abs(src.bounds.left + 12.5 - xk * 1000) < 1e-6 and abs(src.bounds.top - 12.5 - yk * 1000) < 1e-6


def test_own_cells_marks_department_tiles_only():
    own = geo.own_cells("alps")
    assert own.shape == (400, 275)
    assert 0.3 < own.mean() < 1.0
