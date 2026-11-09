"""Terrain classes of the paraglider flights (Hammond landforms) and the pieces counted per class.

Classes of the computer version of Hammond (1964) by Dikau, Brabb and Mark (1991, USGS OFR 91-634), on the
IGN BD ALTI v2 25 m DEM (Lambert-93):

1. the DEM is averaged over blocks of 8 x 8 cells on a 200 m grid aligned on multiples of 200 m (a 25 m cell
   belongs to the block that holds its centre); a block with more than half NoData is NoData; tiles are read
   one by one and added into the blocks, never the whole of France at 25 m;
2. slope by central differences on the 200 m grid; gentle = slope below GENTLE;
3. in a square window of WINDOW x WINDOW cells around each cell, over the valid cells (elevation and slope
   defined): relief = max - min, gentle share = mean of gentle, coverage = share of valid cells; coverage
   below COVERAGE: class 0;
4. 1 plains (relief < 91 m, gentle share > 0.5), 2 hills (relief < 305 m, not plains), 3 low mountains
   (305 <= relief < 915 m), 4 high mountains (relief >= 915 m). Lowland = 1 or 2, mountain = 3 or 4.

The class map, relief and gentle share are saved in TERRAIN (PARAMETERS.md says how). A flight is its
positions on the 10-s grid of tamsd.py, taken to latitude and longitude with the origin of its frame and to
Lambert-93; the class of the cell under each point is stored as a run-length table (flight, k_start, k_end,
class) in TERRAIN, k = t / 10 s on the flight clock, both ends included. Codes of a point: 0 the cell has a
window of low coverage, 1 to 4 the classes, 5 outside the DEM (the cell itself is NoData or off the map).
A piece, a window or a flight takes a class when at least MAJORITY of its grid points are in it; else it is
mixed (-1), or unknown (0) when that share is in codes 0 and 5.

    uv run --with rasterio --with pyproj python terrain.py --open [--sample N] [--recompute] [--rebuild-map]
        [--workers W]    # from this folder, needs SSD_DISANTE; after tamsd.py and prepost.py --open --window 2400
    uv run --with rasterio --with pyproj python terrain.py --open --compare [--only raw,route,flight]
        # the group-against-solo comparisons by binary class; after the above, raw.py, matched.py --stats, prepost.py
"""

import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

import tamsd
from blocks import with_takeoffs
from tamsd import CACHE, COLORS, FIXES, GRID_S, LABELS, OPEN, OUT, SUPPORT, TAU_MAX, stretches

DEM = Path("/Volumes/SSD_DISANTE/IGN_France_Elevation_25m/mainland_france")
TERRAIN = Path("/Volumes/SSD_DISANTE/xc-lab/terrain/hammond_200m")
MAP = TERRAIN / "hammond_200m.npz"
CELL_M, BLOCK, SRC_M = 200, 8, 25  # grid, cells per block side, source grid
GENTLE, WINDOW, COVERAGE = 0.08, 49, 0.5
PLAINS_RELIEF, HILLS_RELIEF, LOW_MOUNTAIN_RELIEF, PLAINS_GENTLE = 91.0, 305.0, 915.0, 0.5
MAJORITY = 0.8
NODATA = -99999.0
MIXED, UNKNOWN = -1, 0
CLASSES = {1: "plains", 2: "hills", 3: "low mountains", 4: "high mountains"}
BINARY = {1: "lowland", 2: "mountain"}
OUTSIDE = 5  # point code: outside the DEM
TILE_KM, DENSITY_M, RANK = 25, 2000, 10**7
POPS = {"solo": "S", "group_solo": "A", "group": "G"}
STAGES = ("pre", "start", "end", "post")
PREPOST_FILE = "prepost_pieces_2400_lag1840.npz"
BASEMAP = Path(__file__).resolve().parents[2].parent / "xc-viewer/src/xc_viewer/assets/basemap.json"
CLASS_COLORS = {0: "#c9c9c9", 1: "#ece6a6", 2: "#b3c97b", 3: "#a67c52", 4: "#5b3a63"}


# ---- DEM: 25 m tiles to the 200 m blocks -------------------------------------------------------

def to_blocks(z, ix_left, iy_top):
    """Sums and counts of valid cells of the 8 x 8 blocks that hold the centres of the tile `z` (NaN = NoData).
    The centre of cell (r, c) is at x = (ix_left + c) 25 m, y = (iy_top - r) 25 m. Returns the block column of
    the first block column, the block row of the first block row (rows go south), the sums and the counts."""
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


def sea_blocks(sums, counts):
    """The blocks whose 64 cells are all exactly 0 m: sea that IGN delivers as 0 instead of NoData. They are NoData."""
    return (counts == BLOCK * BLOCK) & (sums == 0)


def mean_of_blocks(sums, counts):
    """The mean of each block, NaN where more than half of its cells are NoData."""
    half = BLOCK * BLOCK / 2
    return np.where(counts >= half, sums / np.maximum(counts, 1), np.nan).astype(np.float32)


def tile_files():
    """{(x_km, y_km): paths, latest edition first} of the GeoTIFF tiles named by their top-left cell centre. A
    tile that several departments deliver is merged (the first value, from the latest edition)."""
    index = pd.read_csv(DEM / "geotiff_index.csv", usecols=["geotiff", "edition_date"])
    xy = index["geotiff"].str.extract(r"FXX_(\d+)_(\d+)_").astype(int)
    index["x"], index["y"] = xy[0], xy[1]
    index = index.sort_values("edition_date", ascending=False, kind="stable")
    return {k: [DEM / p for p in g["geotiff"]] for k, g in index.groupby(["x", "y"])}


def read_tile(paths, x_km, y_km):
    """One tile at 25 m as float32 with NaN for NoData, the first path's value where it has one."""
    import rasterio

    z = None
    for path in paths:
        with rasterio.open(path) as src:
            assert abs(src.transform.c + SRC_M / 2 - x_km * 1000) < 1e-3 and abs(src.transform.f - SRC_M / 2 - y_km * 1000) < 1e-3
            a = src.read(1).astype(np.float32)
        a[a == NODATA] = np.nan
        z = a if z is None else np.where(np.isnan(z), a, z)
    return z


def mosaic(tiles):
    """The 200 m elevation of every block of France with the sea blocks (sea_blocks) set to NaN: (grid of means,
    sea blocks, x0, y0) with (x0, y0) the top-left corner in Lambert-93 metres. Tiles are added one at a time
    into float64 sums."""
    per = TILE_KM * 1000 // SRC_M
    xs, ys = np.array([k[0] for k in tiles]), np.array([k[1] for k in tiles])
    bx_min, bx_max = xs.min() * 1000 // SRC_M // BLOCK, (xs.max() * 1000 // SRC_M + per - 1) // BLOCK
    by_top, by_bot = ys.max() * 1000 // SRC_M // BLOCK, (ys.min() * 1000 // SRC_M - per + 1) // BLOCK
    nbx, nby = bx_max - bx_min + 1, by_top - by_bot + 1
    sums, counts = np.zeros((nby, nbx)), np.zeros((nby, nbx), np.uint8)
    for n, ((x, y), paths) in enumerate(sorted(tiles.items())):
        bx0, by0, s, c = to_blocks(read_tile(paths, x, y), x * 1000 // SRC_M, y * 1000 // SRC_M)
        r, q = by_top - by0, bx0 - bx_min
        sums[r:r + s.shape[0], q:q + s.shape[1]] += s
        counts[r:r + s.shape[0], q:q + s.shape[1]] += c.astype(np.uint8)
        if n % 200 == 0:
            print(f"tile {n}/{len(tiles)}", flush=True)
    sea = sea_blocks(sums, counts)
    z = mean_of_blocks(sums, counts)
    z[sea] = np.nan
    return z, sea, bx_min * CELL_M, (by_top + 1) * CELL_M


# ---- classes ------------------------------------------------------------------------------------

def classify(z, cell=CELL_M, chunk=400):
    """Hammond classes of a grid `z` (metres, NaN = NoData): class (uint8), relief, gentle share, coverage.
    Stripes of rows with a halo keep the memory small; the window is zero-padded outside the grid."""
    n, m = z.shape
    half = WINDOW // 2
    halo = half + 1
    cls = np.zeros(z.shape, np.uint8)
    relief, gentle, cover = (np.full(z.shape, np.nan, np.float32) for _ in range(3))
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        blk = np.full((r1 - r0 + 2 * halo, m), np.nan, np.float32)
        lo, hi = max(0, r0 - halo), min(n, r1 + halo)
        blk[lo - (r0 - halo):hi - (r0 - halo)] = z[lo:hi]
        slope = np.full(blk.shape, np.nan, np.float32)
        gx = (blk[1:-1, 2:] - blk[1:-1, :-2]) / (2 * cell)
        gy = (blk[2:, 1:-1] - blk[:-2, 1:-1]) / (2 * cell)
        slope[1:-1, 1:-1] = np.hypot(gx, gy)
        valid = np.isfinite(blk) & np.isfinite(slope)
        z_hi = np.where(valid, blk, -np.inf).astype(np.float64)
        z_lo = np.where(valid, blk, np.inf).astype(np.float64)
        top = maximum_filter1d(maximum_filter1d(z_hi, WINDOW, axis=0, mode="constant", cval=-np.inf), WINDOW, axis=1,
                               mode="constant", cval=-np.inf)
        bottom = minimum_filter1d(minimum_filter1d(z_lo, WINDOW, axis=0, mode="constant", cval=np.inf), WINDOW, axis=1,
                                  mode="constant", cval=np.inf)
        box = lambda a: uniform_filter1d(uniform_filter1d(a.astype(np.float64), WINDOW, axis=0, mode="constant"),
                                         WINDOW, axis=1, mode="constant")
        cov, gen = box(valid), box(valid & (slope < GENTLE))
        s = slice(halo, halo + r1 - r0)
        cov, gen, top, bottom = cov[s], gen[s], top[s], bottom[s]
        has = cov > 1e-12
        rel = np.where(has, top - np.where(has, bottom, 0), np.nan)
        share = np.where(has, gen / np.where(has, cov, 1), np.nan)
        relief[r0:r1], gentle[r0:r1], cover[r0:r1] = rel, share, cov
        known = cov >= COVERAGE - 1e-9
        out = np.select([rel >= LOW_MOUNTAIN_RELIEF, rel >= HILLS_RELIEF, (rel < PLAINS_RELIEF) & (share > PLAINS_GENTLE)],
                        [4, 3, 1], 2)
        cls[r0:r1] = np.where(known, out, 0)
    return cls, relief, gentle, cover


def parameters_text():
    return f"""# Hammond landform classes on a 200 m grid

Written by `analyses/groups/terrain.py` of xc-lab (do not edit by hand).

- Source: IGN BD ALTI v2, 25 m, mainland France and Corsica, EPSG:2154, vertical IGN69 (Corsica IGN78C), metres,
  GeoTIFF copies in `IGN_France_Elevation_25m/mainland_france/geotiff`. A tile delivered by several departments
  is merged: the value of the latest edition, NoData filled from the others.
- Resampling: mean of 8 x 8 cells of 25 m on a grid of {CELL_M} m aligned on multiples of {CELL_M} m in Lambert-93;
  a 25 m cell (centres on multiples of 25 m) belongs to the block that holds its centre. A block with more than
  half of its 64 cells NoData is NoData. A block whose 64 cells are all exactly 0 m is NoData too: IGN delivers
  some sea as 0 m and not as NoData.
- Slope: central differences on the {CELL_M} m grid, gentle = slope below {GENTLE} (rise over run).
  A cell is valid when its elevation and its slope are defined.
- Window: square of {WINDOW} x {WINDOW} cells ({WINDOW * CELL_M / 1000:.1f} km) around each cell, over the valid cells:
  relief = max - min of elevation; gentle share = mean of gentle; coverage = valid cells over {WINDOW * WINDOW}
  (cells outside the grid count as invalid); coverage below {COVERAGE}: class 0, unknown.
- Classes: 1 plains = relief < {PLAINS_RELIEF:g} m and gentle share > {PLAINS_GENTLE}; 2 hills = relief < {HILLS_RELIEF:g} m and not
  plains; 3 low mountains = {HILLS_RELIEF:g} <= relief < {LOW_MOUNTAIN_RELIEF:g} m; 4 high mountains = relief >= {LOW_MOUNTAIN_RELIEF:g} m.
  Binary: lowland = 1 or 2, mountain = 3 or 4.
- Files: `hammond_200m.npz` (cls uint8, relief and gentle float32, coverage float32, valid bool = the cell itself
  has an elevation, sea bool = the block was masked as sea at 0 m (its `valid` is false), x0 and y0 = Lambert-93 coordinates of the top-left corner, cell = {CELL_M}; row r, column c
  is the square [x0 + c cell, x0 + (c + 1) cell] x [y0 - (r + 1) cell, y0 - r cell]). The class of a cell
  with `valid` false is defined by its window (it is not 0 when coverage allows), but the lookup of the flights calls
  it outside the DEM.
- `flight_runs_<run>_<open>.parquet`: per flight, runs of the class code of the 10-s grid points
  (flight, flight_id, k_start, k_end, cls; k = t / 10 s on the flight clock, both ends included;
  codes 0 low coverage, 1 to 4 the classes, 5 outside the DEM; a run stops where the grid has a gap).
- References: Hammond, E. H. (1964), Analysis of properties in land form geography, Annals of the AAG 54;
  Dikau, R., Brabb, E. E., Mark, R. M. (1991), Landform classification of New Mexico by computer,
  USGS Open-File Report 91-634.
"""


def leaves(x, path=()):
    """The numeric leaves of nested dicts as {path: value}."""
    if isinstance(x, dict):
        return {k: v for key, value in x.items() for k, v in leaves(value, path + (key,)).items()}
    return {" / ".join(path): x} if isinstance(x, (int, float)) and not isinstance(x, bool) else {}


def mask_effect(before, after):
    """Every numeric result of the report (not the map itself) that the sea mask changed: {path: [before, after]}."""
    skip = ("sea mask", "parameters", "run")
    b = leaves({k: v for k, v in before.items() if k not in skip})
    a = leaves({k: v for k, v in after.items() if k not in skip})
    return {k: [b[k], a[k]] for k in a if k in b and abs(a[k] - b[k]) > 1e-9}


def sea_section(result):
    """The part of PARAMETERS.md that says what the sea mask did to the counts of the reference run."""
    m = result["sea mask"]
    e = m.get("effect", {})
    lines = [f"\n## Sea mask\n", f"- {m['blocks masked']:,} blocks masked ({m['share of the blocks with an elevation before the mask (%)']:.1f}% "
             f"of the blocks with an elevation); {m['valid blocks left']:,} blocks left."]
    if e:
        lines.append(f"- Effect on the counts of the run `{result['run']}` (before -> after), open flights:")
        for pop in ("S", "A", "G"):
            row = lambda k: f"{e[k][0]:.2f} -> {e[k][1]:.2f}" if k in e else "unchanged"
            lines.append(f"  - {pop}: grid points unknown {row(f'grid points / {pop} / unknown')} %, plains "
                         f"{row(f'grid points / {pop} / plains')} %, lowland {row(f'grid points / {pop} / lowland')} %, mountain "
                         f"{row(f'grid points / {pop} / mountain')} %.")
        for pop in ("S", "A", "G"):
            row = lambda k: f"{int(e[k][0])} -> {int(e[k][1])}" if k in e else "unchanged"
            lines.append(f"  - {pop}, pieces of dataset 1000: lowland {row(f'pieces of dataset 1000 / {pop} / binary / lowland / n')}, "
                         f"mountain {row(f'pieces of dataset 1000 / {pop} / binary / mountain / n')}, "
                         f"unknown {row(f'pieces of dataset 1000 / {pop} / binary / unknown / n')}.")
        lines.append("  - events of prepost whose four windows share a binary class: " + ", ".join(
            f"{n} {int(e[k][0])} -> {int(e[k][1])}" if (k := f'prepost events, windows of 2400 s / binary / {n} / n') in e else f"{n} unchanged"
            for n in BINARY.values()) + ".")
        lines.append("  The full list of changed numbers is in `terrain.json` (sea mask / effect).")
    return "\n".join(lines) + "\n"


def build_map(path=MAP):
    """The DEM pass: mosaic, classes, and the npz of `path` with PARAMETERS.md beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    z, sea, x0, y0 = mosaic(tile_files())
    valid = np.isfinite(z)
    cls, relief, gentle, cover = classify(z)
    np.savez_compressed(path, cls=cls, relief=relief, gentle=gentle, coverage=cover, valid=valid, sea=sea, x0=x0,
                        y0=y0, cell=CELL_M)
    (path.parent / "PARAMETERS.md").write_text(parameters_text())


def load_map(path=MAP):
    m = np.load(path)
    return {k: m[k] for k in ("cls", "valid", "sea")} | {"x0": float(m["x0"]), "y0": float(m["y0"]), "cell": float(m["cell"])}


def cell_class(terrain, x, y):
    """The point code (0 to 5) of Lambert-93 positions x, y."""
    cls, valid = terrain["cls"], terrain["valid"]
    ix = np.floor((np.asarray(x) - terrain["x0"]) / terrain["cell"]).astype(np.int64)
    iy = np.floor((terrain["y0"] - np.asarray(y)) / terrain["cell"]).astype(np.int64)
    inside = (ix >= 0) & (ix < cls.shape[1]) & (iy >= 0) & (iy < cls.shape[0])
    ix, iy = np.where(inside, ix, 0), np.where(inside, iy, 0)
    return np.where(inside & valid[iy, ix], cls[iy, ix], OUTSIDE).astype(np.uint8)


# ---- flights: runs of classes -------------------------------------------------------------------

def grid_points(fixes):
    """Per segment of a cleaned flight: the grid instants k (t = k GRID_S) inside it and the E, N, z there."""
    fixes = fixes.sort_values("t", kind="stable")
    t, seg = fixes["t"].to_numpy(float), fixes["segment_id"].to_numpy()
    for s in np.unique(seg):
        q = seg == s
        ts = t[q]
        k = np.arange(int(np.ceil(ts[0] / GRID_S - 1e-9)), int(np.floor(ts[-1] / GRID_S + 1e-9)) + 1)
        if len(k):
            yield k, *(np.interp(k * GRID_S, ts, fixes[c].to_numpy(float)[q]) for c in ("E", "N", "z"))


def run_lengths(k, code):
    """Runs of equal `code` on consecutive instants k: arrays k_start, k_end (included) and class."""
    cut = np.r_[True, (np.diff(k) != 1) | (np.diff(code.astype(int)) != 0)]
    start = np.flatnonzero(cut)
    end = np.r_[start[1:] - 1, len(k) - 1]
    return k[start], k[end], code[start]


_STATE = {}


def _init(map_path):
    from pyproj import Transformer

    _STATE["terrain"] = load_map(map_path)
    _STATE["to_l93"] = Transformer.from_crs(4326, 2154, always_xy=True)


def _job(flight, fixes, frame, spans):
    """Runs of one flight, and its grid points per population on the density grid (2 km cells)."""
    from common.legacy import LocalFrame, enu_to_geodetic

    terrain, to_l93 = _STATE["terrain"], _STATE["to_l93"]
    frame = LocalFrame(*frame)
    ks, codes, xs, ys = [], [], [], []
    for k, e, n, z in grid_points(fixes):
        lat, lon = enu_to_geodetic(e, n, z, frame)
        x, y = to_l93.transform(lon, lat)
        ks.append(k), codes.append(cell_class(terrain, x, y)), xs.append(x), ys.append(y)
    if not ks:
        return flight, None, {}
    order = np.argsort([k[0] for k in ks])
    k, code, x, y = (np.concatenate([a[i] for i in order]) for a in (ks, codes, xs, ys))
    runs = run_lengths(k, code)
    spans = spans.sort_values("a")
    a, b, pop = spans["a"].to_numpy(), spans["b"].to_numpy(), spans["population"].to_numpy()
    t = k * float(GRID_S)
    i = np.searchsorted(a, t, "right") - 1
    inside = (i >= 0) & (t < b[np.maximum(i, 0)])
    nx = int(np.ceil(terrain["cls"].shape[1] * terrain["cell"] / DENSITY_M))
    cx = np.floor((x - terrain["x0"]) / DENSITY_M).astype(np.int64)
    cy = np.floor((terrain["y0"] - y) / DENSITY_M).astype(np.int64)
    on_map = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < terrain["cls"].shape[0] * terrain["cell"] / DENSITY_M)
    density = {}
    for name in POPS:
        m = inside & on_map & (pop[np.maximum(i, 0)] == name)
        if m.any():
            density[name] = np.unique(cy[m] * nx + cx[m], return_counts=True)
    return flight, runs, density


def flight_pass(flights, spans, frames, workers, limit=None, map_path=MAP):
    """The pass over the fixes: run table (DataFrame) and the density grids {population: {cell: points}}."""
    from common.legacy import stream_flights

    ids = flights.set_index("flight_id")["flight"]
    by_flight = dict(tuple(spans.groupby("flight")))
    rows, density, sent = [], {name: {} for name in POPS}, 0

    def store(future):
        flight, runs, dens = future.result()
        if runs is not None:
            rows.append(pd.DataFrame({"flight": flight, "k_start": runs[0], "k_end": runs[1], "cls": runs[2]}))
        for name, (cell, n) in dens.items():
            d = density[name]
            for c, v in zip(cell.tolist(), n.tolist()):
                d[c] = d.get(c, 0) + v

    with ProcessPoolExecutor(workers, initializer=_init, initargs=(map_path,)) as pool:
        pending = set()
        for fixes in stream_flights(FIXES["para"], ["segment_id", "t", "E", "N", "z"]):
            flight = ids.get(str(fixes["flight_id"].iat[0]))
            if flight is None or flight not in by_flight or flight not in frames.index:
                continue
            pending.add(pool.submit(_job, flight, fixes, tuple(frames.loc[flight]), by_flight[flight]))
            sent += 1
            if sent % 2000 == 0:
                print(f"{sent} flights sent", flush=True)
            if len(pending) >= 3 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    store(future)
            if limit and sent >= limit:
                break
        for future in pending:
            store(future)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["flight", "k_start", "k_end", "cls"])
    return table.sort_values(["flight", "k_start"], kind="stable").reset_index(drop=True), density


# ---- counts -------------------------------------------------------------------------------------

def class_counts(runs, flight, k0, k1):
    """Grid points per code (columns 0 to 5) of the intervals [k0, k1] (both included, k = t / GRID_S) of the
    flights `flight`, from the run table `runs` (sorted by flight and k_start)."""
    flight, k0, k1 = (np.asarray(a, np.int64) for a in (flight, k0, k1))
    rf, rs, re, rc = (runs[c].to_numpy(np.int64) for c in ("flight", "k_start", "k_end", "cls"))
    lo = np.searchsorted(rf * RANK + re, flight * RANK + k0, "left")
    hi = np.searchsorted(rf * RANK + rs, flight * RANK + k1, "right")
    n = np.maximum(hi - lo, 0)
    out = np.zeros((len(flight), 6), np.int64)
    q = np.repeat(np.arange(len(flight)), n)
    first = np.cumsum(n) - n
    r = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(first, n)
    np.add.at(out, (q, rc[r]), np.minimum(k1[q], re[r]) - np.maximum(k0[q], rs[r]) + 1)
    return out


def label(counts, share=MAJORITY):
    """The class (1 to 4) with at least `share` of the grid points; else UNKNOWN (0) when codes 0 and 5 have that
    share, else MIXED (-1). Intervals with no grid point are MIXED too (callers drop them)."""
    total = counts.sum(axis=1)
    need = share * total - 1e-9
    best = counts[:, 1:5].argmax(axis=1) + 1
    top = counts[np.arange(len(counts)), best]
    return np.where((total > 0) & (top >= need), best,
                    np.where((total > 0) & (counts[:, 0] + counts[:, 5] >= need), UNKNOWN, MIXED))


def label_binary(counts, share=MAJORITY):
    """The same for lowland (1) and mountain (2), from the codes 1 + 2 and 3 + 4."""
    low, high = counts[:, 1] + counts[:, 2], counts[:, 3] + counts[:, 4]
    return label(np.column_stack([counts[:, 0], low, high, np.zeros_like(low), np.zeros_like(low), counts[:, 5]]), share)


def shares(counts):
    """Share of grid points in each code, in %, with the split of the unknown ones."""
    total = counts.sum()
    pct = lambda v: float(100 * v / total) if total else float("nan")
    return {"points": int(total), **{name: pct(counts[c]) for c, name in CLASSES.items()},
            "unknown": pct(counts[0] + counts[5]), "outside the DEM": pct(counts[5]), "low coverage": pct(counts[0]),
            "lowland": pct(counts[1] + counts[2]), "mountain": pct(counts[3] + counts[4])}


def tally(lab, binary_lab, flight, pilot):
    """Counts per class and per binary class of items with labels `lab`, `binary_lab`, with the flights and pilots."""
    out = {"items": int(len(lab))}
    for key, values, names in (("classes", lab, CLASSES), ("binary", binary_lab, BINARY)):
        out[key] = {}
        for c, name in {**names, UNKNOWN: "unknown", MIXED: "mixed"}.items():
            m = values == c
            out[key][name] = {"n": int(m.sum()), "flights": int(np.unique(flight[m]).size), "pilots": int(np.unique(pilot[m]).size)}
    return out


def pilots_of(flights):
    """The pilot of each flight, a Series by flight number (a flight with no name is a pilot of its own)."""
    catalog = pd.read_csv(FIXES["para"].parents[1] / "catalog/catalog.csv", usecols=["flight_id", "pilot"], dtype=str)
    ids = flights["flight_id"].astype(str)
    pilot = ids.map(catalog.drop_duplicates("flight_id").set_index("flight_id")["pilot"])
    return pd.Series(pilot.where(pilot.notna() & (pilot.str.strip() != ""), "flight " + ids).to_numpy(str),
                     index=flights["flight"].to_numpy())


def busiest_takeoffs(flights, terrain, top=5):
    """The `top` takeoffs with most flights in each binary class of the cell under the takeoff, with names."""
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, 2154, always_xy=True).transform(flights["lon0"].to_numpy(), flights["lat0"].to_numpy())
    code = cell_class(terrain, x, y)
    catalog = pd.read_csv(FIXES["para"].parents[1] / "catalog/catalog.csv", usecols=["flight_id", "takeoff", "dept"], dtype=str)
    names = flights["flight_id"].astype(str).map(catalog.drop_duplicates("flight_id").set_index("flight_id")["takeoff"])
    table = flights.assign(code=code, name=names.fillna("?").to_numpy())
    out = {}
    for key, codes in (("lowland", (1, 2)), ("mountain", (3, 4))):
        sub = table[table["code"].isin(codes)]
        g = sub.groupby("name").agg(flights=("flight", "size"), lat=("lat0", "median"), lon=("lon0", "median"),
                                    cls=("code", lambda c: int(c.mode().iat[0])))
        out[key] = [{"takeoff": n, "flights": int(r.flights), "lat": round(r.lat, 4), "lon": round(r.lon, 4),
                     "class": CLASSES[r.cls]} for n, r in g.sort_values("flights", ascending=False).head(top).iterrows()]
    out["takeoff cells"] = {"unknown or outside": int((~table["code"].isin((1, 2, 3, 4))).sum()), "flights": int(len(table))}
    return out


def events_with_four(path, runs, pilot_of_flight):
    """The events (before/after, windows of prepost.py) with the four windows in `path`: how many, and those whose
    four windows share one class and one binary class, with their flights and pilots."""
    p = {k.split("/")[1]: v for k, v in np.load(path).items() if k.startswith("para/")}
    w = pd.DataFrame({k: p[k] for k in ("flight", "population", "episode", "start", "duration")})
    w = w[w["population"].isin(STAGES)]
    k0 = np.rint(w["start"].to_numpy() / GRID_S).astype(np.int64)
    counts = class_counts(runs, w["flight"], k0, k0 + np.rint(w["duration"].to_numpy() / GRID_S).astype(np.int64))
    w = w.assign(lab=label(counts), blab=label_binary(counts), n=counts.sum(axis=1))
    w = w[w["n"] > 0]
    g = w.groupby("episode")
    have = g["population"].nunique() == 4
    four = w[w["episode"].isin(have[have].index)]
    g = four.groupby("episode")
    ev = pd.DataFrame({"flight": g["flight"].first(), "lab": g["lab"].agg(lambda s: s.iat[0] if s.nunique() == 1 else MIXED),
                       "blab": g["blab"].agg(lambda s: s.iat[0] if s.nunique() == 1 else MIXED)})
    pilot = ev["flight"].map(pilot_of_flight).to_numpy(str)
    return {"events with four windows": int(len(ev)), **tally(ev["lab"].to_numpy(), ev["blab"].to_numpy(), ev["flight"].to_numpy(), pilot)}


def report(flights, spans, runs, terrain, pilot_of_flight):
    """Everything of terrain.json from the run table and the labels."""
    flights = flights[flights["flight"].isin(runs["flight"].unique())]
    spans = spans[spans["flight"].isin(flights["flight"])]
    k0 = np.ceil(spans["a"].to_numpy() / GRID_S - 1e-9).astype(np.int64)
    k1 = np.ceil(spans["b"].to_numpy() / GRID_S - 1e-9).astype(np.int64) - 1
    by_span = class_counts(runs, spans["flight"], k0, k1)
    result = {"open flights in the table": int(len(flights)), "grid points": {}}
    for name, short in POPS.items():
        result["grid points"][short] = shares(by_span[(spans["population"] == name).to_numpy()].sum(axis=0))
    per = pd.DataFrame(by_span).groupby(spans["flight"].to_numpy()).sum()
    per = per[per.sum(axis=1) > 0]
    lab, blab = label(per.to_numpy()), label_binary(per.to_numpy())
    pct = lambda v, c: float(100 * np.mean(v == c))
    result["flights in one class"] = {"flights": int(len(per)),
        "classes": {**{n: pct(lab, c) for c, n in CLASSES.items()}, "unknown": pct(lab, UNKNOWN), "mixed": pct(lab, MIXED)},
        "binary": {**{n: pct(blab, c) for c, n in BINARY.items()}, "unknown": pct(blab, UNKNOWN), "mixed": pct(blab, MIXED)},
        "rule": f"at least {MAJORITY:.0%} of the grid points of the flight (unknown points count in the total)"}
    pilot_flight = pilot_of_flight
    p = tamsd.read_pieces(CACHE / "pieces.npz")["para"]
    long = p["duration"] >= TAU_MAX / SUPPORT
    result["pieces of dataset 1000"] = {}
    for name, short in POPS.items():
        m = long & (p["population"] == name) & np.isin(p["flight"], flights["flight"])
        k = np.rint(p["start"][m] / GRID_S).astype(np.int64)
        c = class_counts(runs, p["flight"][m], k, k + np.rint(p["duration"][m] / GRID_S).astype(np.int64))
        keep = c.sum(axis=1) > 0
        f = p["flight"][m][keep]
        result["pieces of dataset 1000"][short] = tally(label(c[keep]), label_binary(c[keep]), f, pilot_flight.reindex(f).to_numpy(str))
    prepost = CACHE / PREPOST_FILE
    if prepost.exists():
        result["prepost events, windows of 2400 s"] = events_with_four(prepost, runs, pilot_flight)
    result["busiest takeoffs"] = busiest_takeoffs(flights, terrain)
    masked = int(terrain["sea"].sum())
    result["sea mask"] = {"blocks masked": masked, "valid blocks left": int(terrain["valid"].sum()),
                          "share of the blocks with an elevation before the mask (%)":
                          float(100 * masked / (masked + terrain["valid"].sum()))}
    return result


# ---- figure -------------------------------------------------------------------------------------

def figure(result, density, path, map_path=MAP):
    """Map of the classes with the density of group time, and bars of the shares of S, A, G."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    from pyproj import Transformer

    m = np.load(map_path)
    step = 5
    shown = np.where(m["valid"], m["cls"], 255)[::step, ::step].astype(float)
    shown[shown == 255] = np.nan
    x0, y0, cell = float(m["x0"]), float(m["y0"]), float(m["cell"])
    n, w = m["cls"].shape
    fig = plt.figure(figsize=(9, 10.5))
    ax = fig.add_axes([0.02, 0.36, 0.96, 0.62])
    ax.imshow(shown, extent=(x0, x0 + w * cell, y0 - n * cell, y0), cmap=ListedColormap([CLASS_COLORS[i] for i in range(5)]),
              vmin=-0.5, vmax=4.5, interpolation="nearest", rasterized=True)
    nx, ny = int(np.ceil(w * cell / DENSITY_M)), int(np.ceil(n * cell / DENSITY_M))
    grid = np.zeros(nx * ny)
    for c, v in density["group"].items():
        grid[c] += v * GRID_S / 3600
    grid = grid.reshape(ny, nx)
    hours = np.where(grid >= 0.5, grid, np.nan)
    image = ax.imshow(hours, extent=(x0, x0 + nx * DENSITY_M, y0 - ny * DENSITY_M, y0), cmap="autumn_r", alpha=0.85,
                      norm=LogNorm(vmin=0.5, vmax=np.nanmax(hours)), interpolation="nearest", rasterized=True)
    if BASEMAP.exists():
        to_l93 = Transformer.from_crs(4326, 2154, always_xy=True)
        panels = json.loads(BASEMAP.read_text())["panels"]
        for ring in panels["world"]["rings"] + panels["france"]["rings"]:
            r = np.asarray(ring)
            X, Y = to_l93.transform(r[:, 0], r[:, 1])
            ax.plot(X, Y, color="#555555", lw=0.3)
    valid = np.argwhere(m["valid"])
    ax.set(xlim=(x0 + valid[:, 1].min() * cell - 20000, x0 + valid[:, 1].max() * cell + 20000),
           ylim=(y0 - valid[:, 0].max() * cell - 20000, y0 - valid[:, 0].min() * cell + 20000), xticks=[], yticks=[])
    ax.set_aspect("equal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i]) for i in (1, 2, 3, 4, 0)]
    ax.legend(handles, [*CLASSES.values(), "unknown (low coverage)"], loc="lower left", fontsize=8, frameon=False)
    fig.colorbar(image, ax=ax, shrink=0.5, pad=0.01, label="group time (h per 4 km$^2$)")
    ax.set_title("Hammond classes (200 m) and group time of open paragliders", fontsize=10)
    for panel, (box, keys, names) in enumerate((([0.07, 0.05, 0.50, 0.24], [*CLASSES.values(), "unknown"],
                                                  [*CLASSES.values(), "unknown"]),
                                                 ([0.68, 0.05, 0.28, 0.24], ["lowland", "mountain", "unknown"],
                                                  ["lowland", "mountain", "unknown"]))):
        bx = fig.add_axes(box)
        for j, (pop, short) in enumerate(POPS.items()):
            vals = [result["grid points"][short][k] for k in keys]
            bx.bar(np.arange(len(keys)) + (j - 1) * 0.27, vals, 0.27, color=COLORS[pop], label=short)
        bx.set(xticks=range(len(keys)), xticklabels=names, ylabel="grid points (%)")
        bx.tick_params(labelsize=8)
        if panel == 0:
            bx.legend(frameon=False, fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


# ---- comparisons by terrain class (--compare) ---------------------------------------------------
# Three designs of the group-against-solo comparison, each restricted to the pieces (or events) of one terrain
# class, with the machinery of raw.py, matched.py and prepost.py imported, not copied.

BINARY_CODES = {"lowland": 1, "mountain": 2}
FOUR = {"hills": 2, "low mountains": 3, "high mountains": 4}  # plains merged into hills
QUOTED_LAG, TWO_PHASE = 1000, ("time in climb (%)", "searches that end in a climb (%)")
DESIGNS = ("raw", "route", "flight")
COMPARE_ROWS = {"raw": "raw, G - S", "route": "same route, net of selection", "flight": "same flight, g"}


def label_merged(counts):
    """label() with the plains counted as hills."""
    c = counts.copy()
    c[:, 2] += c[:, 1]
    c[:, 1] = 0
    return label(c)


def piece_counts(runs, flight, start, n_points):
    """Grid points per code of pieces (flight, first instant on the flight clock, number of grid points)."""
    k0 = np.rint(np.asarray(start) / GRID_S).astype(np.int64)
    return class_counts(runs, flight, k0, k0 + np.asarray(n_points, np.int64) - 1)


def hloc_at(curve, lag=QUOTED_LAG):
    """Value and band at `lag` of a quote with "lags" (twoway.quote over the lags)."""
    i = list(curve["lags"]).index(lag)
    return {"value": curve["value"][i], "band": [curve["band"][0][i], curve["band"][1][i]]}


def compare_raw(flights, summary, runs, pilot_of_flight):
    """Design 1, dataset 1000 of raw.py: G - S and A - S at every lag and the two phase statistics, S, A and G
    pieces each filtered by their own class; binary classes and, for the four-class version, hills (with the
    plains), low and high mountains."""
    import longlags
    import raw
    import isolated
    import twoway

    code, pilot, block = twoway.cells(flights, summary)
    counts = twoway.counts(pilot, block, np.random.default_rng(tamsd.SEED))
    p = dict(np.load(longlags.PIECES))
    pooled = {k: p["population"] == k for k in raw.POPS}
    use = np.flatnonzero(longlags.long_enough(p, np.isin(p["population"], list(raw.POPS)), TAU_MAX) & np.isin(p["flight"], runs["flight"].unique()))
    c = piece_counts(runs, p["flight"][use], p["start"][use], np.rint(p["duration"][use] / GRID_S) + 1)
    n = len(p["flight"])
    lab, blab, merged = (np.full(n, MIXED) for _ in range(3))
    lab[use], blab[use], merged[use] = label(c), label_binary(c), label_merged(c)
    plains_added = {POPS[k]: int(((merged == 2) & (lab != 2) & m).sum()) for k, m in pooled.items()}
    tables, defs = isolated.run_terms(flights), raw.stat_defs()
    pilot_of_row = pilot_of_flight.reindex(p["flight"]).to_numpy(str)
    j = {name: i for i, name in enumerate(raw.STATISTICS)}

    def one(chosen):
        got, draws, counted = {}, {}, {}
        for pop, short in raw.POPS.items():
            mask = pooled[pop] & chosen
            keep = longlags.long_enough(p, mask, TAU_MAX)
            got[pop] = longlags.curve(p, mask, code, None, TAU_MAX, counts, twoway.band, with_draws=True)
            got[pop]["rows"] = np.vstack([got[pop]["H_loc"], got[pop].pop("H_loc_draws")])
            pieces = (p["flight"][keep], p["start"][keep], p["start"][keep] + p["duration"][keep])
            draws[pop] = raw.phase_statistics(tables, defs, pieces, code, counts).pop("draws")
            counted[short] = {"pieces": int(keep.sum()), "flights": int(np.unique(p["flight"][keep]).size),
                              "pilots": int(np.unique(pilot_of_row[keep]).size)}
        gaps = {g: {"lags": got[a]["lags"], **twoway.quote(got[a]["rows"] - got[b]["rows"])} for g, (a, b) in raw.GAPS.items()}
        return {"counts": counted, "H_loc gaps": gaps, "H_loc(1000 s) gaps": {g: hloc_at(v) for g, v in gaps.items()},
                "phase gaps": {g: {s: twoway.quote(draws[a][:, j[s]] - draws[b][:, j[s]]) for s in TWO_PHASE}
                               for g, (a, b) in raw.GAPS.items()},
                "H_loc(1000 s)": {raw.POPS[pop]: hloc_at({"lags": c_["lags"], "value": c_["H_loc"], "band": c_["H_loc_band"]})
                                  for pop, c_ in got.items()}}

    out = {"binary": {name: one(blab == k) for name, k in BINARY_CODES.items()},
           "four classes": {name: one(merged == k) for name, k in FOUR.items()},
           "plains merged into hills": {"pieces added to hills": plains_added,
                                        "hills pieces before": {POPS[k]: int(((lab == 2) & m).sum()) for k, m in pooled.items()},
                                        "hills pieces after": {POPS[k]: int(((merged == 2) & m).sum()) for k, m in pooled.items()}}}
    return out


def compare_route(flights, summary, runs, pilot_of_flight):
    """Design 2, the same route (matched.py, the chosen caliper, cover MAIN_COVER): G - S|G, A - S|A and their
    difference at H_loc(1000 s) and on the two phase statistics, the query pieces of each side filtered by their
    own binary class."""
    import matched
    import route
    import twoway

    tracks = route.read_tracks(flights, None)
    table = pd.read_parquet(CACHE / f"comparisons_matches{matched.RHO_TAG}.parquet")
    chosen = json.loads((OUT / f"matched_counts{matched.RHO_TAG}.json").read_text())["chosen caliper"]["name"]
    line = {"caliper": chosen, "cover": matched.MAIN_COVER, "isolation": False}
    c = piece_counts(runs, tracks["flight"], tracks["start"], tracks["n"])
    blab = label_binary(c)
    ctx = matched.Context(tracks, flights, summary)
    M = matched.Measured(ctx, matched.select(table, "1000", **line))
    out = {"caliper": line}
    for name, k in BINARY_CODES.items():
        part = table[blab[table["piece"].to_numpy()] == k]
        est = {s: matched.estimate(ctx, M, part, s, line, ("all",), years=False)["all"] for s in ("G", "A")}
        if any(e["main"] is None for e in est.values()):
            out[name] = {"counts": {s: e["counts"] for s, e in est.items()}, "note": "no pieces with a control"}
            continue
        d = matched.contrasts(matched.draws(est["G"]), matched.draws(est["A"]))
        metrics = ("H_loc(1000 s)", *TWO_PHASE)
        both = est["G"]["main"]["boot"]["delta"]["H_loc"] - est["A"]["main"]["boot"]["delta"]["H_loc"]
        lags = est["G"]["main"]["lags"].tolist()
        out[name] = {
            "counts": {s: e["counts"] for s, e in est.items()},
            "deltas": {k: {m: matched.quoted(dm[m]) for m in metrics if m in dm} for k, dm in d.items()},
            "delta curves": {matched.CONTRASTS[0]: matched.delta_curve(est["G"]["main"]),
                             matched.CONTRASTS[1]: matched.delta_curve(est["A"]["main"]),
                             matched.CONTRASTS[2]: {"lags": lags, **twoway.quote(both)}},
            "curves": {label_: matched.curve_of(est[s]["main"], which) for label_, s, which in
                       (("G", "G", "group"), ("S|G", "G", "control"), ("A", "A", "group"), ("S|A", "A", "control"))}}
    return out


def compare_flight(flights, membership, summary, runs, pilot_of_flight, width=2400):
    """Design 3, the same flight (prepost.py --open, windows of `width` s): the events whose four windows share a
    binary class; g, d w, D_in and D_out at H_loc(1000 s) with their curves and the two phase statistics."""
    import decompose
    import isolated
    import prepost
    import raw
    import twoway
    from tamsd import read_pieces

    lags = prepost.window_lags(width)
    spans = stretches(flights, membership, summary)
    wins = prepost.windows(spans, np.random.default_rng(tamsd.SEED), width)
    p = read_pieces(CACHE / f"prepost_pieces_{width}_lag{lags.max()}.npz")["para"]
    info = decompose.flight_table(flights, summary)[0]
    block, of_pilot, of_block = twoway.cells(flights, summary)
    counts = twoway.counts(of_pilot, of_block, np.random.default_rng(tamsd.SEED))
    wins = wins.merge(pd.DataFrame({k: p[k] for k in ("flight", "population", "episode")}).drop_duplicates(),
                      on=["flight", "population", "episode"])
    tables, defs = isolated.run_terms(flights), raw.stat_defs()
    stage = np.isin(p["population"], STAGES) & np.isin(p["flight"], runs["flight"].unique())
    w = pd.DataFrame({k: p[k][stage] for k in ("flight", "population", "episode", "start", "duration")})
    w["row"] = np.flatnonzero(stage)
    c = piece_counts(runs, w["flight"], w["start"], np.rint(w["duration"] / GRID_S) + 1)
    w["blab"] = label_binary(c)
    g = w.groupby("episode")
    event = pd.DataFrame({"flight": g["flight"].first(), "n": g["population"].nunique(),
                          "same": g["blab"].agg(lambda s: s.iat[0] if s.nunique() == 1 else MIXED)})
    event = event[event["n"] == len(STAGES)]
    out = {"window_s": width, "events with four windows": int(len(event))}
    for name, k in BINARY_CODES.items():
        ep = event.index[event["same"] == k]
        rows = w.loc[w["episode"].isin(ep), "row"].to_numpy()
        pc = {key: v[rows] for key, v in p.items()}
        units = decompose.msd_units(pc, "episode", info, block)
        meta, r, q, stats = prepost.window_rates(tables, wins[wins["episode"].isin(ep) & wins["population"].isin(STAGES)],
                                                 block, defs)
        msd_on = lambda labels: prepost.curves_on(units, counts, labels, lags)
        phase = lambda labels: prepost.phase_on(meta, r, q, counts, labels, stats)
        m, ph = prepost.effects(msd_on, comparisons={}), prepost.effects(phase, comparisons={})
        flights_of = event.loc[ep, "flight"].to_numpy()
        entry = {"events": int(len(ep)), "flights": int(np.unique(flights_of).size),
                 "pilots": int(np.unique(pilot_of_flight.reindex(flights_of).to_numpy(str)).size), "effects": {}}
        for eff in ("g", "d w", "D_in", "D_out"):
            d = prepost.describe_msd(m[eff], lags)
            entry["effects"][eff] = {"events (MSD)": m[eff][1], "windows (MSD)": m[eff][2], "events (phase)": ph[eff][1],
                                     "H_loc(1000 s)": d[f"H_loc({prepost.QUOTED_LAG} s)"],
                                     "curve": {"lags": lags.tolist(), **d["dH_loc"]},
                                     **{s: twoway.quote(ph[eff][0][:, list(stats).index(s)]) for s in TWO_PHASE}}
        out[name] = entry
    return out


def forest_compare(result, path):
    """Delta H_loc(1000 s) of the three designs, lowland and mountain, with the 90% bands."""
    import matplotlib.pyplot as plt

    quote = {"raw": lambda d: d["H_loc(1000 s) gaps"]["G - S"],
             "route": lambda d: d["deltas"]["(G - S|G) - (A - S|A)"]["H_loc(1000 s)"],
             "flight": lambda d: d["effects"]["g"]["H_loc(1000 s)"]}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = {"lowland": CLASS_COLORS[2], "mountain": CLASS_COLORS[4]}
    for i, design in enumerate(DESIGNS):
        for j, name in enumerate(BINARY_CODES):
            d = result[design]["binary"][name] if design == "raw" else result[design][name]
            if "note" in d:
                continue
            q = quote[design](d)
            y = i + (j - 0.5) * 0.28
            ax.errorbar(q["value"], y, xerr=[[q["value"] - q["band"][0]], [q["band"][1] - q["value"]]], fmt="o", ms=5,
                        color=colors[name], elinewidth=1.8, capsize=2, label=name if i == 0 else None)
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set(yticks=range(len(DESIGNS)), yticklabels=[COMPARE_ROWS[d] for d in DESIGNS], xlabel=r"$\Delta H_{\rm loc}(1000\ {\rm s})$")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.set_title("Group effect by terrain class, open paragliders (90% bands)", fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return int(x) if isinstance(x, np.integer) else x


def compare(only=None):
    """Run the designs of `only` (default all) and write terrain_compare_<design>.json; the merged
    terrain_compare.json and the forest plot once the three are there."""
    flights = with_takeoffs(tamsd.read_flights())
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    runs = pd.read_parquet(TERRAIN / f"flight_runs_{LABELS.name}_{OPEN or 'all'}.parquet")
    pilot_of_flight = pilots_of(flights)
    for design in only or DESIGNS:
        if design == "raw":
            result = compare_raw(flights, summary, runs, pilot_of_flight)
        elif design == "route":
            result = compare_route(flights, summary, runs, pilot_of_flight)
        else:
            result = compare_flight(flights, membership, summary, runs, pilot_of_flight)
        (OUT / f"terrain_compare_{design}.json").write_text(json.dumps(jsonable(result), indent=1))
        print(design, "done", flush=True)
    parts = {d: OUT / f"terrain_compare_{d}.json" for d in DESIGNS}
    if all(f.exists() for f in parts.values()):
        merged = {d: json.loads(f.read_text()) for d, f in parts.items()}
        merged = {"run": LABELS.name, "rule": f"a piece or an event takes a class when at least {MAJORITY:.0%} of its grid points "
                  "are in it; events: the four windows share it", **merged}
        (OUT / "terrain_compare.json").write_text(json.dumps(merged, indent=1))
        forest_compare(merged, OUT / "terrain_compare.pdf")


# ---- driver -------------------------------------------------------------------------------------

def arg(name, default=None, cast=int):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


if __name__ == "__main__":
    tamsd.folders()
    TERRAIN.mkdir(parents=True, exist_ok=True)
    if "--compare" in sys.argv:
        compare(sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None)
        sys.exit()
    if "--rebuild-map" in sys.argv or not MAP.exists():
        build_map()
    terrain = load_map()
    sample = arg("--sample")
    tag = f"{LABELS.name}_{OPEN or 'all'}" + (f"_sample{sample}" if sample else "")
    runs_file, density_file = TERRAIN / f"flight_runs_{tag}.parquet", TERRAIN / f"density_2km_{tag}.npz"
    flights = with_takeoffs(tamsd.read_flights())
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    spans = stretches(flights, membership, summary)
    if "--recompute" in sys.argv or not runs_file.exists():
        meta = pd.read_parquet(FIXES["para"].parent / "flights_meta.parquet", columns=["flight_id", "lat0", "lon0", "alt0"])
        meta = meta.astype({"flight_id": str}).drop_duplicates("flight_id").set_index("flight_id")
        frames = flights.set_index("flight")[["flight_id"]].astype({"flight_id": str}).join(meta, on="flight_id")
        frames = frames[["lat0", "lon0", "alt0"]].fillna({"alt0": 0.0}).dropna()
        runs, density = flight_pass(flights, spans, frames, arg("--workers", 3), sample)
        runs.to_parquet(runs_file)
        np.savez(density_file, **{f"{name}/{k}": v for name, d in density.items()
                                  for k, v in (("cell", np.array(list(d))), ("points", np.array(list(d.values()))))})
    runs = pd.read_parquet(runs_file)
    dens = np.load(density_file)
    density = {name: dict(zip(dens[f"{name}/cell"].tolist(), dens[f"{name}/points"].tolist())) for name in POPS}
    pilot_of_flight = pilots_of(flights)
    result = report(flights, spans, runs, terrain, pilot_of_flight)
    result["run"] = LABELS.name
    result["parameters"] = {"cell_m": CELL_M, "gentle": GENTLE, "window_cells": WINDOW, "coverage": COVERAGE,
                            "majority": MAJORITY, "map": str(MAP)}
    suffix = f"_sample{sample}" if sample else ""
    unmasked = OUT / "terrain_unmasked.json"
    if unmasked.exists() and not sample:
        result["sea mask"]["effect"] = mask_effect(json.loads(unmasked.read_text()), result)
    if not sample:
        (TERRAIN / "PARAMETERS.md").write_text(parameters_text() + sea_section(result))
    (OUT / f"terrain{suffix}.json").write_text(json.dumps(result, indent=1))
    figure(result, density, OUT / f"terrain{suffix}.pdf")
    print(json.dumps({k: result[k] for k in result if k != "busiest takeoffs"}, indent=1))
    print(json.dumps(result["busiest takeoffs"], indent=1))
