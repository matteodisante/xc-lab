"""Tiled run of the routing and the crest edges of a region (tiled.py), resumable per stage and tile.

    RIDGES_FLATS=blm RIDGES_CACHE=/Volumes/SSD_DATA/xc-lab/ridges/<cache> \\
        uv run --with numba python run_tiled.py alps --dem 25 --tile 400 --stage all

Stages: dem (the DEM as one disk raster), mask (mountain area), fill, flats, dirs, acc, links, edges; `all` runs
them in this order; `clean` deletes the rasters and tables of the run (not the products). The run folder is CACHE/tiled_<region>_<dem>m_T<T>/ (stage tables, per-cell rasters, stats.json);
mask_, edges_ and hand_<region> go to build.folder() at 25 m (so that build.py --stage curves and hgeo.py read them)
and to <run folder>/products at 5 m.
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
import dem_resolution
import geo
import mountains
import ridges
import tiled

TILE = {25: 400, 5: 2500}  # default tile sizes (cells)
STAGES = ["dem", "mask", "fill", "flats", "dirs", "acc", "links", "edges"]
STRIP = 500  # rows read at a time from the 5 m DEM
GB = 2**30


def box_shape(region, dem):
    x0, x1, y0, y1 = geo.region_box(region)
    n = 1000 // dem
    return (y1 - y0) * n, (x1 - x0) * n, (x0 * 1000.0, y1 * 1000.0)


def load_dem(run, region, dem):
    """Writes the DEM of the region box to run.z; (x0, y0) is the centre of cell (0, 0) (Lambert-93 m)."""
    f = run.dir / "dem.json"
    if f.exists():
        return
    h, w, (x0, y0) = box_shape(region, dem)
    if dem == 25:
        z, _ = geo.mosaic(region)
        run.z.write(0, 0, z)
    else:
        r0 = int(round((dem_resolution.ORIGIN5[1] - y0) / 5))
        c0 = int(round((x0 - dem_resolution.ORIGIN5[0]) / 5))
        with rasterio.open(dem_resolution.DEM5) as src:
            for r in range(0, h, STRIP):  # the box may reach beyond the mosaic: no data there
                n = min(STRIP, h - r)
                a = np.full((n, w), np.nan, np.float32)
                v = src.read(1, window=Window(c0, r0 + r, min(w, src.width - c0), n)).astype(np.float32)
                v[v == -99999.0] = np.nan
                a[:, :v.shape[1]] = v
                run.z.write(r, 0, a)
    f.write_text(json.dumps(dict(x0=x0, y0=y0, shape=[h, w])))


def mask_stage(run, region, dem, out, x0, y0):
    f = out / f"mask_{region}.npz"
    if f.exists():
        return
    B = 1000 // dem
    h, w = run.shape
    nby, nbx = h // B, w // B
    hm = np.vstack([mountains.block_mean(run.z.read(i * B, (i + 1) * B, 0, nbx * B, np.nan), B) for i in range(nby)])
    mask = mountains.mountain_area_h(hm, geo.own_cells(region))
    np.savez_compressed(f, mask=mask, left=x0 - dem / 2, top=y0 + dem / 2, cell=1000.0)
    run.log(f"  mask: mountain area {mask.sum()} km2")


def dir_size(path):
    return sum(p.stat().st_blocks * 512 for p in path.rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=list(geo.REGIONS))
    ap.add_argument("--dem", type=int, choices=[25, 5], default=25)
    ap.add_argument("--tile", type=int)
    ap.add_argument("--keep", action="store_true", help="keep filled, d and acc (deleted after links otherwise)")
    ap.add_argument("--stage", choices=STAGES + ["all", "clean"], default="all")
    a = ap.parse_args()
    if ridges.FLATS != "blm":
        raise SystemExit("set RIDGES_FLATS=blm")
    T = a.tile or TILE[a.dem]
    h, w, (x0, y0) = box_shape(a.region, a.dem)
    folder = geo.CACHE / f"tiled_{a.region}_{a.dem}m_T{T}"
    out = build.folder() if a.dem == 25 else folder / "products"
    cell = float(a.dem)
    run = tiled.Run(folder, (h, w), T, cell=cell, xsec=dem_resolution.cells(dem_resolution.XSEC_M, cell), out=out,
                    name=a.region, keep=a.keep)
    run.log(f"{a.region} {a.dem} m, grid {h} x {w}, tile {T}, {len(list(run.tiles()))} tiles, folder {folder}")
    stats = folder / "stats.json"
    st = json.loads(stats.read_text()) if stats.exists() else {}
    if a.stage == "clean":  # the rasters and tables of the run; the products and stats.json stay
        for f in folder.glob("*.npy"):
            f.unlink()
        for d in [p for p in folder.iterdir() if p.is_dir() and p.name != "products"]:
            shutil.rmtree(d)
        return
    todo = STAGES if a.stage == "all" else [a.stage]
    for s in todo:
        t = time.time()
        if s == "dem":
            load_dem(run, a.region, a.dem)
        elif s == "mask":
            mask_stage(run, a.region, a.dem, out, x0, y0)
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
