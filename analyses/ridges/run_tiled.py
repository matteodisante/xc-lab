"""Tiled run of the routing and the crest edges of a region (tiled.py), resumable per stage and tile.

    RIDGES_FLATS=blm RIDGES_CACHE=/Volumes/SSD_DATA/xc-lab/ridges RIDGES_DEM=5 \\
        uv run --with numba python run_tiled.py alps --dem 5 --tile 2500 --stage all [--window alps1]

RIDGES_DEM must equal --dem: the scripts that follow (build.py, hgeo.py) find the parameter folder through it.
`run_massif.py` drives the whole chain.

Stages: dem (the DEM as one disk raster), mask (mountain area), fill, flats, dirs, acc, links, edges; `all` runs
them in this order; `clean` deletes the rasters and tables of the run (not the products or stats.json). The run folder is CACHE/tiled_<region or window>_<dem>m_T<T>/ (stage tables, per-cell rasters, stats.json);
mask_, edges_ and hand_<region> go to build.folder() (so that build.py --stage curves and hgeo.py read them).
The 5 m DEM is read from the VRT; what lies outside the mosaic, or has no tile, is no data. A window runs on its
own box with every 1 km cell in the mountain area.
"""

import argparse
import json
import resource
import shutil
import time

import numpy as np
import rasterio
from rasterio.windows import Window

import build
import curves
import dem_resolution
import geo
import mountains
import ridges
import tiled

TILE = {25: 400, 5: 2500}  # default tile sizes (cells)
STAGES = ["dem", "mask", "fill", "flats", "dirs", "acc", "links", "edges"]
STRIP = 500  # rows read at a time from the 5 m DEM
NODATA5 = -99999.0
GB = 2**30


def window_box(name):
    """(region, (x0, x1, y0, y1) in km) of a window: a trial window of dem_resolution.py with its routing margin, or a
    box `x0,x1,y0,y1` in km (region None). The cells have their centres at x0 + 5 j and y1 - 5 i, as in the region boxes."""
    if name in dem_resolution.WINDOWS:
        region, (x0, y0, x1, y1) = dem_resolution.WINDOWS[name]
        m = dem_resolution.MARGIN_M
        return region, tuple(int(v / 1000) for v in (x0 - m, x1 + m, y0 - m, y1 + m))
    box = tuple(int(v) for v in name.split(","))
    if len(box) != 4 or box[0] >= box[1] or box[2] >= box[3]:
        raise SystemExit(f"window {name!r}: a name of dem_resolution.WINDOWS or x0,x1,y0,y1 in km")
    return None, box


def window_tag(name):
    return name.replace(",", "-")


def box_shape(region, dem, window=None):
    """(rows, cols, (x, y) of the centre of cell (0, 0)) of the region box, or of the window (km box) inside it."""
    x0, x1, y0, y1 = window_box(window)[1] if window else geo.region_box(region)
    n = 1000 // dem
    return (y1 - y0) * n, (x1 - x0) * n, (x0 * 1000.0, y1 * 1000.0)


def read_dem5(src, r0, r1, c0, c1):
    """Elevations (float32, NaN = no data) of the rows r0:r1 and columns c0:c1 of the 5 m mosaic `src`; the part of
    the box outside the mosaic, and the cells without a tile, are NaN (rasterio clips a window that sticks out)."""
    out = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
    ra, rb, ca, cb = max(r0, 0), min(r1, src.height), max(c0, 0), min(c1, src.width)
    if ra < rb and ca < cb:
        v = src.read(1, window=Window(ca, ra, cb - ca, rb - ra)).astype(np.float32)
        v[v == NODATA5] = np.nan
        out[ra - r0:rb - r0, ca - c0:cb - c0] = v
    return out


def origin5(x0, y0):
    """(row, col) in the 5 m mosaic of the cell whose centre is (x0, y0)."""
    return (int(round((dem_resolution.ORIGIN5[1] - y0) / 5)), int(round((x0 - dem_resolution.ORIGIN5[0]) / 5)))


def load_dem(run, region, dem, window=None):
    """Writes the DEM of the region box (or window) to run.z; (x0, y0) is the centre of cell (0, 0) (Lambert-93 m)."""
    f = run.dir / "dem.json"
    if f.exists():
        return
    h, w, (x0, y0) = box_shape(region, dem, window)
    if dem == 25:
        z, _ = geo.mosaic(region)
        run.z.write(0, 0, z)
    else:
        r0, c0 = origin5(x0, y0)
        with rasterio.open(dem_resolution.DEM5) as src:
            for r in range(0, h, STRIP):
                run.z.write(r, 0, read_dem5(src, r0 + r, r0 + min(h, r + STRIP), c0, c0 + w))
    f.write_text(json.dumps(dict(x0=x0, y0=y0, shape=[h, w])))


def block_means(read, h, w, cell):
    """1 km mean elevations of a grid of cell size `cell` read by read(r0, r1, c0, c1), a block row at a time."""
    B = mountains.block_cells(cell)
    nbx = w // B
    return np.vstack([mountains.block_mean(read(i * B, (i + 1) * B, 0, nbx * B), B) for i in range(h // B)])


def mask_stage(run, region, dem, out, x0, y0, window=None):
    f = out / f"mask_{region}.npz"
    if f.exists():
        return
    h, w = run.shape
    if window:  # the window lies inside the mountain area: every 1 km cell counts
        mask = np.ones((h // (1000 // dem), w // (1000 // dem)), bool)
    else:
        hm = block_means(lambda *a: run.z.read(*a, np.nan), h, w, dem)
        mask = mountains.mountain_area_h(hm, geo.own_cells(region))
    tmp = f.with_name(f.stem + "_tmp.npz")
    np.savez_compressed(tmp, mask=mask, left=x0 - dem / 2, top=y0 + dem / 2, cell=1000.0)
    tmp.replace(f)
    run.log(f"  mask: mountain area {mask.sum()} km2")


def dir_size(path):
    return sum(p.stat().st_blocks * 512 for p in path.rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=list(geo.REGIONS))
    ap.add_argument("--dem", type=int, choices=[25, 5], default=25)
    ap.add_argument("--window", help="trial window of dem_resolution.py (35 x 35 km) or a box x0,x1,y0,y1 in km")
    ap.add_argument("--tile", type=int)
    ap.add_argument("--keep", action="store_true", help="keep filled, d and acc (deleted after links otherwise)")
    ap.add_argument("--stage", choices=STAGES + ["all", "clean"], default="all")
    a = ap.parse_args()
    if ridges.FLATS != "blm":
        raise SystemExit("set RIDGES_FLATS=blm")
    if geo.DEM_M != a.dem:
        raise SystemExit(f"set RIDGES_DEM={a.dem}")
    T = a.tile or TILE[a.dem]
    if a.window and window_box(a.window)[0] not in (None, a.region):
        raise SystemExit(f"window {a.window} lies in {window_box(a.window)[0]}")
    h, w, (x0, y0) = box_shape(a.region, a.dem, a.window)
    folder = geo.CACHE / f"tiled_{window_tag(a.window) if a.window else a.region}_{a.dem}m_T{T}"
    out = build.folder(a.dem)
    if a.stage == "clean":  # the rasters and tables of the run; stats.json stays (the products are in the parameter folder)
        for p in list(folder.iterdir()) if folder.exists() else []:  # (a volume may pair a file with a ._ file that goes with it)
            if p.is_dir():
                shutil.rmtree(p)
            elif p.name != "stats.json" and not p.name.startswith("._"):
                p.unlink(missing_ok=True)
        print(time.strftime("%H:%M:%S"), f"cleaned {folder}", flush=True)
        return  # before tiled.Run, which would create the DEM raster
    cell = float(a.dem)
    run = tiled.Run(folder, (h, w), T, cell=cell, xsec=ridges.cells(ridges.XSEC_M, cell), edge_cells=ridges.cells(curves.EDGE_M, cell),
                    out=out, name=a.region, keep=a.keep, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    run.log(f"{a.region} {a.dem} m, grid {h} x {w}, tile {T}, {len(list(run.tiles()))} tiles, folder {folder}")
    stats = folder / "stats.json"
    st = json.loads(stats.read_text()) if stats.exists() else {}
    todo = STAGES if a.stage == "all" else [a.stage]
    for s in todo:
        t = time.time()
        if s == "dem":
            load_dem(run, a.region, a.dem, a.window)
        elif s == "mask":
            mask_stage(run, a.region, a.dem, out, x0, y0, a.window)
        elif s == "edges":
            tiled.edges(run, x0, y0)
        else:
            getattr(tiled, s)(run)
        st[s] = dict(seconds=round(time.time() - t, 1),
                     peak_ram_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB, 2))
        if s == "flats" and (folder / "crossing.npz").exists():
            c = np.load(folder / "crossing.npz")
            st[s].update(crossing_flats=len(c["cells"]), largest_flat_cells=int(c["cells"].max(initial=0)),
                         largest_flat_box=[int(max(c["bbox"][:, 1] - c["bbox"][:, 0], default=0)),
                                           int(max(c["bbox"][:, 3] - c["bbox"][:, 2], default=0))])
        if s == "links":
            st[s]["valley_cells"] = int(np.load(folder / "valley.npz")["idx"].size)
            st[s]["links"] = int(np.load(folder / "valley.npz")["down"].size - 1)
        st["disk_gb"] = round(dir_size(folder) / 1e9, 2)
        stats.write_text(json.dumps(st, indent=1))
        run.log(f"{s}: {st[s]['seconds']} s, peak RAM {st[s]['peak_ram_gb']} GB, run folder {st['disk_gb']} GB")


if __name__ == "__main__":
    main()
