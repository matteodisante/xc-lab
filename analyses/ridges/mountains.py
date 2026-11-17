"""Mountain area (UNEP-WCMC, Kapos et al. 2000) on 1 km cells computed from the 25 m DEM.

Classes on the 1 km mean elevation z, its slope and its local elevation range (LER, max - min within RADIUS
cells): 1-3 z >= 2500 m; 4 z >= 1500 m and slope >= 2 deg; 5 z >= 1000 m and (slope >= 5 deg or LER > 300 m);
6 z >= 300 m and LER > 300 m. Enclosed non-mountain pieces of at most BASIN_KM2 are added and the largest
connected piece is kept.
"""

import numpy as np
from scipy import ndimage as ndi

BLOCK = 40  # 25 m cells per 1 km cell
MIN_VALID = 0.5  # share of valid 25 m cells for a 1 km cell to have data
RADIUS = 7  # km, local elevation range
LER = 300.0
Z_HIGH, Z_MID, Z_LOW = 2500.0, 1500.0, 1000.0
Z_FOOT = 300.0
SLOPE_MID, SLOPE_LOW = 2.0, 5.0  # degrees
BASIN_KM2 = 25
CONNECTIVITY = 2  # 1 = sides, 2 = sides and corners


def block_mean(z, block=BLOCK):
    """1 km mean elevation (NaN where less than MIN_VALID of the cells have data); z is cropped to whole blocks
    of `block` cells."""
    nby, nbx = z.shape[0] // block, z.shape[1] // block
    out = np.full((nby, nbx), np.nan, np.float32)
    for i in range(nby):
        a = z[i * block:(i + 1) * block, :nbx * block].reshape(block, nbx, block)
        ok = np.isfinite(a)
        n = ok.sum(axis=(0, 2))
        s = np.where(ok, a, 0).sum(axis=(0, 2), dtype=np.float64)
        out[i] = np.where(n >= MIN_VALID * block * block, s / np.maximum(n, 1), np.nan)
    return out


def local_range(h, radius=RADIUS):
    """max - min of h within a disk of `radius` cells (NaN cells ignored)."""
    r = np.arange(-radius, radius + 1)
    disk = (r[:, None] ** 2 + r[None, :] ** 2) <= radius**2
    hi = ndi.maximum_filter(np.where(np.isnan(h), -np.inf, h), footprint=disk, mode="constant", cval=-np.inf)
    lo = ndi.minimum_filter(np.where(np.isnan(h), np.inf, h), footprint=disk, mode="constant", cval=np.inf)
    return np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, np.nan)


def slope_deg(h):
    gy, gx = np.gradient(np.nan_to_num(h, nan=0.0), 1000.0)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def classes(h):
    """Boolean mountain classes 1-6 of the 1 km elevation grid h."""
    s, r = slope_deg(h), local_range(h)
    with np.errstate(invalid="ignore"):
        return ((h >= Z_HIGH) | ((h >= Z_MID) & (s >= SLOPE_MID))
                | ((h >= Z_LOW) & ((s >= SLOPE_LOW) | (r > LER))) | ((h >= Z_FOOT) & (r > LER)))


def mountain_area(z, own=None, block=BLOCK):
    """Boolean mask on 1 km cells (rows south from the first row of z); cells outside `own` (1 km cells,
    the tiles of the region's departments) count as no data."""
    return mountain_area_h(block_mean(z, block), own)


def mountain_area_h(h, own=None):
    """mountain_area from the 1 km mean elevations h."""
    if own is not None:
        h[~own] = np.nan
    nodata = np.isnan(h)
    m = classes(h) & ~nodata
    # non-mountain pieces that do not touch missing data or the border and are small
    lab, n = ndi.label(~m)
    touch = np.zeros(n + 1, bool)
    bad = ndi.binary_dilation(nodata, structure=ndi.generate_binary_structure(2, 1))
    touch[np.unique(lab[bad])] = True
    touch[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
    size = np.bincount(lab.ravel(), minlength=n + 1)
    small = ~touch & (size <= BASIN_KM2)
    small[0] = False
    m |= small[lab] & ~nodata
    lab, n = ndi.label(m, structure=ndi.generate_binary_structure(2, CONNECTIVITY))
    if n == 0:
        return m
    size = np.bincount(lab.ravel())
    size[0] = 0
    return lab == size.argmax()
