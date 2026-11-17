"""The metre-to-cell conversion and the pipeline on a synthetic ridge sampled at 25 and 5 m."""

import numpy as np
import pytest

import curves
import dem_resolution as dr
import ridges


def test_conversion_gives_the_25_m_constants_back():
    assert dr.cells(dr.XSEC_M, 25.0) == ridges.XSEC_CELLS
    assert dr.cells(dr.BRIDGE_M, 25.0) == curves.BRIDGE_EDGES
    assert dr.cells(dr.PRUNE_M, 25.0) == curves.PRUNE_EDGES
    assert ridges.valley_cells(0.5, 25.0) == ridges.valley_cells(0.5) == 800
    assert dr.cells(dr.XSEC_M, 5.0) == 30 and dr.cells(dr.BRIDGE_M, 5.0) == 200 and dr.cells(dr.PRUNE_M, 5.0) == 20
    assert ridges.valley_cells(0.5, 5.0) == 20000


def ridge(cell, size=12000.0):
    """A wavy ridge (x = 6000 + 300 sin(2 pi y / 6000)) between two valleys 4 km on each side falling to the south,
    on a grid of `cell` m; cell (0, 0) has its centre at (0, size)."""
    n = int(size / cell) + 1
    x, d = np.meshgrid(np.arange(n) * cell, np.arange(n) * cell)  # d: distance from the top row
    crest = 6000.0 + 300.0 * np.sin(2 * np.pi * (size - d) / 6000.0)
    z = 500.0 + 0.3 * np.minimum(np.abs(x - (crest - 4000.0)), np.abs(x - (crest + 4000.0))) + 0.05 * (size - d)
    return z.astype(np.float32), (0.0, size)


def test_ridge_crest_found_at_both_resolutions_within_one_coarse_cell():
    box = (2000.0, 2000.0, 10000.0, 10000.0)
    found = {}
    for cell in (25.0, 5.0):
        z, org = ridge(cell)
        cs, ends = dr.crest_curves(z, org, cell, curves.SMOOTH_M, box)
        assert len(cs) >= 1
        found[cell] = np.concatenate(cs)
    d = dr.nearest(found[5.0], found[25.0])
    assert d.max() <= 25.0
    assert dr.length_km([found[25.0]]) > 5 and dr.length_km([found[5.0]]) > 5
    x, y = found[5.0].T
    assert np.abs(x - (6000.0 + 300.0 * np.sin(2 * np.pi * y / 6000.0))).max() < 25.0
