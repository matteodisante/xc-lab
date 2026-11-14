"""Terrain class of a flight: the median ground elevation under its points.

Ground: the IGN BD ALTI v2 DEM, 25 m, Lambert-93, averaged over blocks of BLOCK x BLOCK cells on a CELL_M grid
aligned on multiples of CELL_M (a 25 m cell belongs to the block that holds its centre). A block with more than
half NoData is NoData; a block whose cells are all exactly 0 m is NoData too (IGN delivers some sea as 0 m, not
as NoData). Tiles are read one at a time and added into the blocks, never the whole of France at 25 m.

Flight: every point of the flight on the 10-s grid (each segment of the cleaned track, whole flight from takeoff
to landing), taken to Lambert-93 with the frame of its fixes. The ground elevation under a point is the mean of
the block that holds it; points outside the DEM (NaN block or off the map) are dropped. The class of the flight
is read on the median of the remaining ground elevations, with the edges EDGES_M:

    plains < 300 m <= hills < 800 m <= low mountains < 1500 m <= high mountains.

A flight with less than MIN_ON_DEM of its points on the DEM is unknown.

    uv run python terrain.py --build    # the elevation grid on SSD_DISANTE, once (a few minutes)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRID_S = 10  # s, grid of the points of a flight
EDGES_M = (300.0, 800.0, 1500.0)  # plains | hills | low mountains | high mountains
CLASSES = ("plains", "hills", "low mountains", "high mountains")
UNKNOWN = "unknown"
MIN_ON_DEM = 0.5  # a flight needs at least this share of its points on the DEM

DEM = Path("/Volumes/SSD_DISANTE/IGN_France_Elevation_25m/mainland_france")
CACHE = Path("/Volumes/SSD_DISANTE/xc-lab/universality/dem_bdalti25_mean200m_sea0nodata")
ELEVATION, ORIGIN = CACHE / "elevation_200m.npy", CACHE / "grid.json"
CELL_M, BLOCK, SRC_M, TILE_KM = 200, 8, 25, 25
NODATA = -99999.0


# ---- DEM: 25 m tiles to the 200 m blocks (as analyses/groups/terrain.py) ---------------------------

def to_blocks(z, ix_left, iy_top):
    """Sums and counts of valid cells of the BLOCK x BLOCK blocks that hold the centres of the tile `z` (NaN =
    NoData), the cell (r, c) being centred on x = (ix_left + c) 25 m, y = (iy_top - r) 25 m. Returns the block
    column of the first block column, the block row of the first block row (rows go south), sums, counts."""
    h, w = z.shape
    bx0, by_top = ix_left // BLOCK, iy_top // BLOCK
    by_bot = (iy_top - h + 1) // BLOCK
    nbx = (ix_left + w - 1) // BLOCK - bx0 + 1
    nby = by_top - by_bot + 1
    full = np.full((nby * BLOCK, nbx * BLOCK), np.nan, np.float64)
    top, left = (by_top + 1) * BLOCK - 1 - iy_top, ix_left - bx0 * BLOCK
    full[top:top + h, left:left + w] = z
    cells = full.reshape(nby, BLOCK, nbx, BLOCK)
    ok = np.isfinite(cells)
    return bx0, by_top, np.where(ok, cells, 0.0).sum(axis=(1, 3)), ok.sum(axis=(1, 3))


def block_means(sums, counts):
    """The mean of each block, NaN where more than half of its cells are NoData or all are exactly 0 m."""
    n = BLOCK * BLOCK
    z = np.where(counts >= n / 2, sums / np.maximum(counts, 1), np.nan).astype(np.float32)
    z[(counts == n) & (sums == 0)] = np.nan
    return z


def tile_files():
    """{(x_km, y_km): paths, latest edition first} of the GeoTIFF tiles named by their top-left cell centre."""
    index = pd.read_csv(DEM / "geotiff_index.csv", usecols=["geotiff", "edition_date"])
    xy = index["geotiff"].str.extract(r"FXX_(\d+)_(\d+)_").astype(int)
    index["x"], index["y"] = xy[0], xy[1]
    index = index.sort_values("edition_date", ascending=False, kind="stable")
    return {k: [DEM / p for p in g["geotiff"]] for k, g in index.groupby(["x", "y"])}


def read_tile(paths):
    """One tile at 25 m as float32 with NaN for NoData, the value of the first path where it has one."""
    import rasterio

    z = None
    for path in paths:
        with rasterio.open(path) as src:
            a = src.read(1).astype(np.float32)
        a[a == NODATA] = np.nan
        z = a if z is None else np.where(np.isnan(z), a, z)
    return z


def mosaic(tiles):
    """The elevation of every 200 m block of the tiles, and the top-left corner (x0, y0) in Lambert-93 metres."""
    per = TILE_KM * 1000 // SRC_M
    xs, ys = np.array([k[0] for k in tiles]), np.array([k[1] for k in tiles])
    bx_min, bx_max = xs.min() * 1000 // SRC_M // BLOCK, (xs.max() * 1000 // SRC_M + per - 1) // BLOCK
    by_top, by_bot = ys.max() * 1000 // SRC_M // BLOCK, (ys.min() * 1000 // SRC_M - per + 1) // BLOCK
    nbx, nby = bx_max - bx_min + 1, by_top - by_bot + 1
    sums, counts = np.zeros((nby, nbx)), np.zeros((nby, nbx), np.uint8)
    for n, ((x, y), paths) in enumerate(sorted(tiles.items())):
        bx0, by0, s, c = to_blocks(read_tile(paths), x * 1000 // SRC_M, y * 1000 // SRC_M)
        r, q = by_top - by0, bx0 - bx_min
        sums[r:r + s.shape[0], q:q + s.shape[1]] += s
        counts[r:r + s.shape[0], q:q + s.shape[1]] += c.astype(np.uint8)
        if n % 200 == 0:
            print(f"tile {n}/{len(tiles)}", flush=True)
    return block_means(sums, counts), bx_min * CELL_M, (by_top + 1) * CELL_M


def parameters_text(n_valid):
    return f"""# Ground elevation on a {CELL_M} m grid

Written by `analyses/universality/terrain.py` of xc-lab (do not edit by hand).

- Source: IGN BD ALTI v2, 25 m, mainland France and Corsica, EPSG:2154, metres, GeoTIFF copies in
  `IGN_France_Elevation_25m/mainland_france/geotiff`. A tile delivered by several departments is merged: the
  value of the latest edition, NoData filled from the others.
- Resampling: mean of {BLOCK} x {BLOCK} cells of {SRC_M} m on a grid of {CELL_M} m aligned on multiples of {CELL_M} m;
  a 25 m cell belongs to the block that holds its centre. A block with more than half of its {BLOCK * BLOCK} cells
  NoData is NaN; a block whose {BLOCK * BLOCK} cells are all exactly 0 m is NaN too (sea delivered as 0 m).
- Files: `elevation_200m.npy` (float32, NaN = no ground; row r, column c is the square [x0 + c cell,
  x0 + (c + 1) cell] x [y0 - (r + 1) cell, y0 - r cell]) and `grid.json` (x0, y0, cell in Lambert-93 metres).
- {n_valid:,} blocks with an elevation.
"""


def build(tiles=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    z, x0, y0 = mosaic(tiles or tile_files())
    np.save(ELEVATION, z)
    ORIGIN.write_text(json.dumps({"x0": int(x0), "y0": int(y0), "cell": CELL_M}))
    (CACHE / "PARAMETERS.md").write_text(parameters_text(int(np.isfinite(z).sum())))


# ---- ground elevation under points, class of a flight -----------------------------------------------

def load(mmap=True):
    """The grid: {"z": elevation (memory-mapped), "x0", "y0", "cell"}."""
    return {"z": np.load(ELEVATION, mmap_mode="r" if mmap else None), **json.loads(ORIGIN.read_text())}


def ground(dem, x, y):
    """The ground elevation (m) of the block under the Lambert-93 positions x, y; NaN outside the DEM."""
    z = dem["z"]
    ix = np.floor((np.asarray(x, float) - dem["x0"]) / dem["cell"]).astype(np.int64)
    iy = np.floor((dem["y0"] - np.asarray(y, float)) / dem["cell"]).astype(np.int64)
    inside = (ix >= 0) & (ix < z.shape[1]) & (iy >= 0) & (iy < z.shape[0])
    out = np.full(ix.shape, np.nan)
    out[inside] = z[iy[inside], ix[inside]]
    return out


def median_ground(elevation):
    """(median of the finite ground elevations, points, points on the DEM) of a flight; NaN median when none."""
    on = np.isfinite(elevation)
    return (float(np.median(elevation[on])) if on.any() else float("nan")), int(len(elevation)), int(on.sum())


def classify(median, n_points, n_on):
    """The class of flights from their median ground elevation and their points: names of CLASSES, or UNKNOWN
    when fewer than MIN_ON_DEM of the points are on the DEM."""
    median, n_points, n_on = (np.asarray(a, float) for a in (median, n_points, n_on))
    names = np.array(CLASSES + (UNKNOWN,))
    k = np.digitize(median, EDGES_M)
    known = (n_points > 0) & (n_on >= MIN_ON_DEM * n_points) & np.isfinite(median)
    return names[np.where(known, k, len(CLASSES))]


def grid_points(fixes):
    """Per segment of a cleaned flight: the grid instants (t = k GRID_S) inside it and E, N, z there."""
    fixes = fixes.sort_values("t", kind="stable")
    t, seg = fixes["t"].to_numpy(float), fixes["segment_id"].to_numpy()
    for s in np.unique(seg):
        q = seg == s
        ts = t[q]
        k = np.arange(int(np.ceil(ts[0] / GRID_S - 1e-9)), int(np.floor(ts[-1] / GRID_S + 1e-9)) + 1)
        if len(k):
            yield k, *(np.interp(k * GRID_S, ts, fixes[c].to_numpy(float)[q]) for c in ("E", "N", "z"))


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
