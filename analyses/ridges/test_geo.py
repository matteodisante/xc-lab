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


def test_bd_alti_is_the_5m_dem_every_fifth_cell():
    import dem_resolution as dr

    box = (561000, 6189000, 566000, 6194000)  # one corner of the window pyr1
    z25, o25 = geo.window(*box)
    z5, o5 = dr.read5(box)
    assert o25 == o5
    sub = z5[::5, ::5][:z25.shape[0], :z25.shape[1]]
    both = np.isfinite(z25) & np.isfinite(sub)
    assert both.mean() > 0.5 and np.array_equal(z25[both], sub[both])
