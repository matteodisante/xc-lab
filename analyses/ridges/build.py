"""From the DEM to the crest curves of a region; caches in CACHE/<parameter folder>/.

    uv run --with numba python build.py pyrenees [--stage drain|curves]

drain: mosaic, mountain area, flow routing, crest edges -> mask_<region>.npz, edges_<region>.npz, hand_<region>.npy
curves: edges -> strokes -> splines -> curves_<region>.npz (curves of the mountain area, points every STEP m)
"""

import argparse
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import curves
import geo
import joined
import mountains
import ridges
import tamsd

WORKERS = 6
HAND_NONE = -32768  # int16 height above the streams where the cell drains to no valley
MASK_SHARE = 0.5  # a curve belongs to the massif if this share of it lies in the mountain area


def folder():
    name = (f"bdalti{ridges.CELL:g}_valley{ridges.VALLEY_KM2:g}_xsec{ridges.XSEC_CELLS}-{ridges.XSEC_DROP:g}"
            f"_hand{ridges.HAND_MIN:g}_gap{curves.BRIDGE_EDGES}_prune{curves.PRUNE_EDGES}"
            f"_defl{curves.MAX_DEFLECTION:g}-{curves.DIRECTION_M:g}_min{curves.MIN_LENGTH:g}_step{curves.STEP:g}"
            f"_smooth{curves.SMOOTH_M:g}_join{joined.JOIN_M:g}-{joined.CONE_DEG:g}-{joined.SNAP_M:g}")
    d = geo.CACHE / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_peak(d, key):
    """Peak resident memory of this process (GB) of a stage, kept in peak_ram.json."""
    f = d / "peak_ram.json"
    peaks = json.loads(f.read_text()) if f.exists() else {}
    peaks[key] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    f.write_text(json.dumps(peaks, indent=1))


def write_parameters(d):
    (d / "PARAMETERS.md").write_text(f"""# Crest pipeline, {d.name}

Code: `xc-lab/analyses/ridges/` (`build.py`, `joined.py`, `hgeo.py`). DEM: IGN BD ALTI 25 m, Lambert-93.
Departments: {geo.REGIONS}.

| Parameter | Value |
|---|---|
| valley | cells draining at least {ridges.VALLEY_KM2} km2 |
| cross-section | ground {ridges.XSEC_CELLS} cells away on each side at least {ridges.XSEC_DROP:g} m lower |
| height above streams | at least {ridges.HAND_MIN:g} m on both sides |
| gaps bridged | flat stretches up to {curves.BRIDGE_EDGES} sides |
| pruning | dead ends shorter than {curves.PRUNE_EDGES} sides |
| strokes | deflection at most {curves.MAX_DEFLECTION:g} deg, direction measured {curves.DIRECTION_M:g} m out |
| curves | shorter than {curves.MIN_LENGTH:g} m dropped, spline cut-off {curves.SMOOTH_M:g} m, step {curves.STEP:g} m |
| mountain area | UNEP-WCMC on 1 km cells, see `mountains.py` |
| joins | straight, at most {joined.JOIN_M:g} m, within {joined.CONE_DEG:g} deg of the run-out direction ({joined.RUNOUT_M:g} m), height above streams at least {joined.HAND_MIN:g} m in every DEM cell crossed, snapped to a dead end within {joined.SNAP_M:g} m, {joined.N_COMPONENTS} components tried per dead end |
| TAMSD | {tamsd.N_LAGS} log-spaced lags to a quarter of each curve, at least {tamsd.MIN_CURVES} curves per lag, v = {tamsd.V_KMH:g} km/h, {tamsd.N_BOOT} bootstrap draws, seed {tamsd.SEED} |

Files, per region: `mask_*.npz` (1 km cells; `left`, `top` the west and north edges), `edges_*.npz`,
`hand_*.npy` (int16 m, {HAND_NONE} = none), `curves_*.npz`, `tamsd_rows_*_<kind>.npz`, `hgeo_*_<kind>.csv`,
`hgeo_windows.csv`, `paths_*.npz` (float32 offsets from `origin`; Lambert-93 = xy + origin).
""")


def drain(region, d):
    t = time.time()
    z, (x0, y0) = geo.mosaic(region)
    print(f"mosaic {z.shape} in {time.time() - t:.0f} s", flush=True)
    mask = mountains.mountain_area(z, geo.own_cells(region))
    left, top = x0 - ridges.CELL / 2, y0 + ridges.CELL / 2
    np.savez_compressed(d / f"mask_{region}.npz", mask=mask, left=left, top=top, cell=1000.0)
    print(f"mountain area {mask.sum()} km2", flush=True)
    lid, sz, tin, tout = ridges.drainage(z)
    print(f"drainage in {time.time() - t:.0f} s, {tin.size - 1} links", flush=True)
    acc, flat = ridges.crest_edges(z, lid, sz, tin, tout)
    hand = np.full(z.shape, HAND_NONE, np.int16)
    lid, sz = lid.reshape(z.shape), sz.reshape(z.shape)
    for r in range(0, z.shape[0], 500):
        s = slice(r, r + 500)
        hand[s] = np.where(lid[s] > 0, np.clip(np.rint(z[s] - sz[s]), -32767, 32767), HAND_NONE)
    np.save(d / f"hand_{region}.npy", hand)
    del lid, sz, hand
    w = z.shape[1]
    vid = np.unique(np.r_[acc.ravel(), flat.ravel()])
    near = curves.near_nodata(z, vid // (w + 1), vid % (w + 1), curves.EDGE_CELLS)
    np.savez_compressed(d / f"edges_{region}.npz", acc=acc, flat=flat, vid=vid, near=near, shape=z.shape,
                        x0=x0, y0=y0)
    print(f"{len(acc)} crest sides, {len(flat)} flat candidates, done in {time.time() - t:.0f} s", flush=True)


def junctions(verts, xy):
    """Pairs of curves that share a network vertex (rows of jc) and the vertex coordinates (rows of jxy)."""
    vv = np.concatenate(verts)
    ss = np.repeat(np.arange(len(verts)), [len(a) for a in verts])
    o = np.argsort(vv, kind="stable")
    vv, ss = vv[o], ss[o]
    cut = np.r_[0, np.nonzero(np.diff(vv))[0] + 1, len(vv)]
    jc, jxy = [], []
    for a, b in zip(cut[:-1], cut[1:]):
        if b - a < 2:
            continue
        first = ss[a]
        for other in np.unique(ss[a + 1:b]):
            if other != first:
                jc.append((first, other))
                jxy.append(xy[vv[a]])
    return np.array(jc, np.int64).reshape(-1, 2), np.array(jxy).reshape(-1, 2)


def _smooth(parts):
    return [curves.smooth_resample(p) for p in parts]


def build_curves(region, d):
    t = time.time()
    e = np.load(d / f"edges_{region}.npz")
    h, w = e["shape"]
    x0, y0 = float(e["x0"]), float(e["y0"])
    edges = curves.bridge(e["acc"], e["flat"])
    vid, inv = np.unique(edges.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)
    xy = np.c_[x0 - ridges.CELL / 2 + ridges.CELL * (vid % (w + 1)), y0 + ridges.CELL / 2 - ridges.CELL * (vid // (w + 1))]
    print(f"{len(edges)} sides after bridging, {len(vid)} vertices", flush=True)
    v, off = curves.strokes(inv[:, 0], inv[:, 1], xy)
    print(f"{len(off) - 1} strokes in {time.time() - t:.0f} s", flush=True)
    near = e["near"][np.searchsorted(e["vid"], vid)]
    parts = [xy[v[a:b]] for a, b in zip(off[:-1], off[1:])]
    edge = np.array([near[v[a:b]].any() for a, b in zip(off[:-1], off[1:])])
    # a stroke shorter than the minimum, even as a straight line, cannot give a long curve
    chord = np.array([np.hypot(*np.diff(p, axis=0).T).sum() for p in parts])
    todo = np.nonzero(chord >= curves.MIN_LENGTH * 0.7)[0]
    chunks = np.array_split(todo, 20 * WORKERS)
    with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("fork")) as ex:
        res = list(ex.map(_smooth, [[parts[i] for i in c] for c in chunks]))
    pts = [p for r in res for p in r]
    idx = np.array([i for c in chunks for i in c])
    length = np.array([(len(p) - 1) * curves.STEP for p in pts])
    m = np.load(d / f"mask_{region}.npz")
    mask, left, top, cell = m["mask"], float(m["left"]), float(m["top"]), float(m["cell"])
    keep, inm = [], []
    for k, p in enumerate(pts):
        j = np.clip(((p[:, 0] - left) // cell).astype(int), 0, mask.shape[1] - 1)
        i = np.clip(((top - p[:, 1]) // cell).astype(int), 0, mask.shape[0] - 1)
        share = mask[i, j].mean()
        inm.append(share)
        keep.append(length[k] >= curves.MIN_LENGTH and share >= MASK_SHARE)
    keep = np.array(keep)
    sel = np.nonzero(keep)[0]
    xyall = np.concatenate([pts[k] for k in sel])
    n = np.array([len(pts[k]) for k in sel])
    jc, jxy = junctions([v[off[i]:off[i + 1]] for i in idx[sel]], xy)
    np.savez_compressed(d / f"curves_{region}.npz", xy=xyall, off=np.r_[0, np.cumsum(n)], length=length[sel],
                        edge=edge[idx[sel]], inmask=np.array(inm)[sel], jc=jc, jxy=jxy,
                        x0=x0, y0=y0, shape=e["shape"])
    print(f"{len(sel)} curves, {length[sel].sum() / 1e3:.0f} km, median {np.median(length[sel]) / 1e3:.2f} km "
          f"({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=list(geo.REGIONS))
    ap.add_argument("--stage", choices=["drain", "curves"], default=None)
    a = ap.parse_args()
    d = folder()
    write_parameters(d)
    if a.stage in (None, "drain"):
        drain(a.region, d)
        record_peak(d, f"{a.region}_drain")
    if a.stage in (None, "curves"):
        build_curves(a.region, d)
        record_peak(d, f"{a.region}_curves")
