"""IGN BD ALTI 25 m tiles of a region, mosaicked over the bounding box of its departments (Lambert-93).

Tile files are named elevation_25m_FXX_<x>_<y>_...tif with the west edge x and the north edge y in km;
cell centres sit on multiples of 25 m. A mosaic has its first cell centre at (x0, y0), rows go south.
"""

import os
import re
from pathlib import Path

import numpy as np
import rasterio

DEM = Path("/Volumes/SSD_DISANTE/IGN_France_Elevation_25m/mainland_france/geotiff")
CACHE = Path(os.environ.get("RIDGES_CACHE", "/Volumes/SSD_DISANTE/xc-lab/ridges"))
DEM_M = float(os.environ.get("RIDGES_DEM", "25"))  # cell size (m) of the run: 25 (BD ALTI) or 5 (RGE ALTI)
CELL = 25.0  # cell of the BD ALTI tiles read here
TILE = 1000  # cells per tile side
REGIONS = {
    "alps": ["04", "05", "06", "26", "38", "73", "74", "83", "84"],
    "pyrenees": ["09", "11", "31", "64", "65", "66"],
}
_NAME = re.compile(r"elevation_25m_FXX_(\d+)_(\d+)_")


def tile_index(folders=None):
    """{(x_km, y_km): [paths]} of the tiles in the department folders (all folders when None)."""
    out = {}
    for d in sorted(DEM.glob("D*")):
        if folders is not None and d.name not in folders:
            continue
        for f in d.glob("elevation_25m_*.tif"):
            m = _NAME.match(f.name)
            out.setdefault((int(m[1]), int(m[2])), []).append(f)
    return out


def region_box(region):
    """(x_min_km, x_max_km, y_min_km, y_max_km) of the tiles of the departments of a region (tile corners)."""
    keys = tile_index({f"D{d:0>3}" for d in REGIONS[region]})
    xs, ys = zip(*keys)
    return min(xs), max(xs) + 25, min(ys) - 25, max(ys)


def mosaic(region):
    """Elevations (float32, NaN = no data) over the bounding box, and (x0, y0) of cell (0, 0)."""
    x0k, x1k, y0k, y1k = region_box(region)
    n_x, n_y = (x1k - x0k) // 25, (y1k - y0k) // 25
    z = np.full((n_y * TILE, n_x * TILE), np.nan, np.float32)
    tiles = tile_index()
    for i in range(n_y):
        for j in range(n_x):
            for f in tiles.get((x0k + 25 * j, y1k - 25 * i), []):
                with rasterio.open(f) as src:
                    a = src.read(1)
                a = np.where(a == src.nodata, np.nan, a).astype(np.float32)
                blk = z[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]
                np.copyto(blk, a, where=np.isnan(blk))
    return z, (x0k * 1000.0, y1k * 1000.0)


def own_cells(region):
    """Boolean grid on 1 km cells of the mosaic: True inside the tiles of the region's departments."""
    x0, x1, y0, y1 = region_box(region)
    own = np.zeros(((y1 - y0) // 25, (x1 - x0) // 25), bool)
    for x, y in tile_index({f"D{d:0>3}" for d in REGIONS[region]}):
        own[(y1 - y) // 25, (x - x0) // 25] = True
    return np.kron(own, np.ones((25, 25), bool))


def window(x0, y0, x1, y1):
    """Elevations (float32, NaN = no data) of the cells with centres in [x0, x1] x [y0, y1] (Lambert-93 m),
    and the centre (x, y) of cell (0, 0)."""
    c0, c1 = int(np.ceil(x0 / CELL)), int(np.floor(x1 / CELL))
    r0, r1 = int(np.ceil(-y1 / CELL)), int(np.floor(-y0 / CELL))  # row r has y = -r * CELL
    z = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan, np.float32)
    tiles = tile_index()
    for kx in range(c0 // TILE, c1 // TILE + 1):
        for ky in range(-r1 // TILE, -r0 // TILE + 2):
            for f in tiles.get((kx * 25, ky * 25), [])[:1]:
                with rasterio.open(f) as src:
                    a = src.read(1)
                a = np.where(a == src.nodata, np.nan, a).astype(np.float32)
                tr0, tc0 = -ky * TILE, kx * TILE  # tile row 0 has y = ky * 25000: row index -ky * 1000
                ra, rb = max(r0, tr0), min(r1 + 1, tr0 + TILE)
                ca, cb = max(c0, tc0), min(c1 + 1, tc0 + TILE)
                if ra < rb and ca < cb:
                    z[ra - r0:rb - r0, ca - c0:cb - c0] = a[ra - tr0:rb - tr0, ca - tc0:cb - tc0]
    return z, (c0 * CELL, -r0 * CELL)
