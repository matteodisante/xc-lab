"""Joined trees and paths of a region, their TAMSD and H_geo; writes the exports to CACHE/<parameter folder>/.

    uv run --with numba python hgeo.py pyrenees

Files: tamsd_rows_<region>_<kind>.npz, hgeo_<region>_<kind>.csv, hgeo_windows.csv, paths_<region>.npz
(kind = curves, paths). Coordinates are Lambert-93; stored points are float32 offsets from `origin`.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

import build
import geo
import joined
import tamsd

LONG_PATH_M = 10000.0  # the longest path of a tree is exported if at least this long
KINDS = ("curves", "paths")


def curves_data(d, region):
    c = np.load(d / f"curves_{region}.npz")
    return c, np.load(d / f"hand_{region}.npy", mmap_mode="r")


def rows_for(polys, ks):
    return np.array([tamsd.tamsd(p, ks) for p in polys])


def save_kind(d, region, kind, rows, ks, group, length, edge, tree):
    """Rows and fits of one kind; `group` is what the bootstrap resamples (curve or tree), `tree` the joined tree."""
    np.savez_compressed(d / f"tamsd_rows_{region}_{kind}.npz", rows=rows, k=ks, s_m=ks * tamsd.STEP, group=group,
                        tree=tree, length=length, edge=edge, step=tamsd.STEP)
    out = []
    for suffix, sel in (("", np.ones(len(rows), bool)), ("_noedge", ~edge)):
        tab, wins = tamsd.analyse(rows[sel], ks, group[sel])
        if suffix == "":
            pd.DataFrame(tab).to_csv(d / f"hgeo_{region}_{kind}.csv", index=False, float_format="%.6g")
        out += [dict(region=region, kind=kind + suffix, window=n, lo_km=lo, hi_km=hi, H=h, ci_lo=a, ci_hi=b,
                     n=int(sel.sum())) for n, lo, hi, h, a, b in wins]
    return out


def save_joins(d, region, net, lab, tr, emap, c):
    """The joins (end points, length, tree) and the longest tree with its longest path, for the figures."""
    js = [e for e in net.edges if e[2] == 2]
    big = max(tr, key=lambda t: t["dist"][np.isfinite(t["dist"])].max())
    far = big["tips"][np.argmax(big["dist"][big["tips"]])]
    path, _ = joined.path_polyline(net, big, far, emap)
    np.savez_compressed(d / f"joins_{region}.npz", a=np.array([net.pt(*net.key[e[0]]) for e in js]),
                        b=np.array([net.pt(*net.key[e[1]]) for e in js]), length=np.array([e[6] for e in js]),
                        tree=np.array([lab[e[0]] for e in js]), big_tree=lab[big["nodes"][0]],
                        big_curves=np.unique([net.key[n][0] for n in big["nodes"]]), big_path=path)


def run(region):
    t0 = time.time()
    d = build.folder()
    c, hand = curves_data(d, region)
    xy, off = c["xy"], c["off"]
    x0, y0 = float(c["x0"]), float(c["y0"])
    net = joined.Net(xy, off, c["jc"], c["jxy"])
    nj = joined.make_joins(net, hand, x0, y0, cell=joined.cell_of(c))
    emap = joined.edge_map(net)
    tr = list(joined.trees(net))
    n_paths = sum(len(t["tips"]) for t in tr)
    print(f"{len(off) - 1} curves, {nj} joins, {len(tr)} trees, {n_paths} paths ({time.time() - t0:.0f} s)", flush=True)

    # curves
    edge_c = c["edge"].astype(bool)
    curve_tree = np.zeros(len(off) - 1, int)
    lab = net.labels()
    for ci in range(len(off) - 1):
        curve_tree[ci] = lab[net.nodes[(ci, 0)]]
    n_max = (off[1:] - off[:-1]).max()
    ks_c = tamsd.lags(n_max)
    rows_c = rows_for((xy[off[i]:off[i + 1]] for i in range(len(off) - 1)), ks_c)
    windows = save_kind(d, region, "curves", rows_c, ks_c, np.arange(len(off) - 1), c["length"], edge_c, curve_tree)
    print(f"curves tamsd done ({time.time() - t0:.0f} s)", flush=True)

    # paths: the lag grid comes from the longest path
    n_long = int(max(t["dist"][np.isfinite(t["dist"])].max() for t in tr) / tamsd.STEP) + 1
    ks_p = tamsd.lags(n_long)
    rows, group, length, edge, longest = [], [], [], [], []
    for t in tr:
        far = t["tips"][np.argmax(t["dist"][t["tips"]])]
        for j in t["tips"]:
            p, used = joined.path_polyline(net, t, j, emap)
            rows.append(tamsd.tamsd(p, ks_p))
            group.append(t["id"])
            length.append((len(p) - 1) * tamsd.STEP)
            edge.append(any(edge_c[u] for u in used))
            if j == far and length[-1] >= LONG_PATH_M:
                longest.append((t["id"], p))
        if len(rows) % 2000 < len(t["tips"]):
            print(f"  {len(rows)} paths ({time.time() - t0:.0f} s)", flush=True)
    rows, group, length, edge = np.array(rows), np.array(group), np.array(length), np.array(edge)
    print(f"{len(rows)} paths, median {np.median(length) / 1e3:.1f} km, longest {length.max() / 1e3:.0f} km", flush=True)
    windows += save_kind(d, region, "paths", rows, ks_p, group, length, edge, group)
    origin = np.array([x0, y0])
    pts = np.concatenate([(p - origin).astype(np.float32) for _, p in longest])
    np.savez_compressed(d / f"paths_{region}.npz", xy=pts, off=np.r_[0, np.cumsum([len(p) for _, p in longest])],
                        tree=np.array([t for t, _ in longest]), origin=origin,
                        length=np.array([(len(p) - 1) * tamsd.STEP for _, p in longest]))
    save_joins(d, region, net, lab, tr, emap, c)
    f = d / "hgeo_windows.csv"
    new = pd.DataFrame(windows)
    if f.exists():
        old = pd.read_csv(f)
        new = pd.concat([old[old.region != region], new])
    new.to_csv(f, index=False, float_format="%.4f")
    print(new[new.region == region].to_string(index=False), flush=True)
    print(f"{len(longest)} long paths exported; total {time.time() - t0:.0f} s", flush=True)
    build.record_peak(d, f"{region}_hgeo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=list(geo.REGIONS))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run(ap.parse_args().region)
