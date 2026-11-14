"""Select the FFVL flights of the ridges analysis and run flights.py on their cleaned fixes.

Candidates: paragliders and hang gliders whose declared type is a distance flight (as in ../groups/openflights.py)
and that the old cleaning kept (cleaned.py reads their fixes). flights.py keeps those with a median sampling
interval of at most MAX_DT_S on the cleaned fixes, an opening (first to last fix over the path flown) of at least
MIN_OPENING and at least REGION_MIN of the fixes over the mask of one massif.

The full run goes by chunks of CHUNK row groups of fixes.parquet; each chunk is saved in the cache folder as soon
as it is done and skipped when the run is started again.

    uv run python run_flights.py --sample 200 [--plot 10]    # trial, writes out/sample/
    uv run python run_flights.py --full --jobs 4             # every candidate, writes the SSD_DISANTE cache
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cleaned as cl
import flights as fl

DISK = Path("/Volumes/SSD_DATA")
LAB = Path("/Volumes/SSD_DISANTE/xc-lab/ridges")
DECLARED_OPEN = {"Dist libre", "Dist 1 pt", "Dist 2 pts", "Dist 3 pts"}
CHUNK = 10  # row groups of fixes.parquet per saved chunk (about 1 400 flights)
SEED = 0


def newest_data():
    """The newest crest folder of LAB that holds a complete hgeo_windows.csv (and masks, paths, TAMSD rows)."""
    need = ["hgeo_windows.csv", "mask_alps.npz", "mask_pyrenees.npz", "tamsd_rows_alps_paths.npz", "tamsd_rows_pyrenees_paths.npz"]
    done = [d for d in LAB.glob("bdalti*") if all((d / f).exists() for f in need)]
    return max(done, key=lambda d: (d / "hgeo_windows.csv").stat().st_mtime)


def declared_open(discipline):
    """Flight ids of the declared distance flights of the catalogue."""
    c = pd.read_csv(DISK / cl.FOLDERS[discipline] / "catalog/catalog.csv", dtype=str, usecols=["flight_id", "flight_type"])
    return set(c.loc[c["flight_type"].isin(DECLARED_OPEN), "flight_id"])


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.rename(path)


def full_run(cfg, jobs, out):
    """Results of every candidate, chunk by chunk; chunks already in out/chunks are loaded, not computed."""
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    results = []
    with Pool(jobs, fl._init, (cfg,)) as pool:
        for disc in cl.FOLDERS:
            ids, n = declared_open(disc), cl.n_row_groups(disc)
            for a in range(0, n, CHUNK):
                f = out / "chunks" / f"{disc}_{a:05d}.pkl"
                if not f.exists():
                    res = list(pool.imap(fl.process, cl.stream(disc, ids, a, a + CHUNK), chunksize=4))
                    save_atomic(res, f)
                    print(f"{disc} row groups {a}-{min(a + CHUNK, n)} of {n}: {len(res)} flights", flush=True)
                results += pickle.loads(f.read_bytes())
    return results


def plot_tracks(items, results, cfg, out, n=10):
    """Tracks of the first n flights that passed, circling runs in red."""
    ok = {r["info"]["flight"] for r in results if r["rows"] is not None}
    items = [it for it in items if it["flight"] in ok][:n]
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.4))
    for ax, it in zip(axs.flat, items):
        info, pieces = fl.read_flight(it, cfg)
        for t, xy in pieces:
            p = fl.regrid(t, xy)[1]
            c = fl.circling(p)
            ax.plot(p[:, 0] / 1000, p[:, 1] / 1000, color="0.5", lw=0.6)
            q = np.where(c[:, None], p, np.nan)
            ax.plot(q[:, 0] / 1000, q[:, 1] / 1000, color="C3", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"{it['discipline'][:4]} {it['flight']} {info['region']}\n{info['track_km']:.0f} km", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "tracks.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=int, default=0, help="trial on this many flights, into out/sample/")
    ap.add_argument("--full", action="store_true", help="every candidate")
    ap.add_argument("--data", default=None, help="crest folder (default: the newest complete one of LAB)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot", type=int, default=0)
    a = ap.parse_args()
    if bool(a.sample) == a.full:
        ap.error("give --sample N or --full")
    data = a.data or str(newest_data())
    cfg = fl.configure(data)
    if a.full:
        name = f"flights_cleaned_v{fl.V_KMH:g}_open{fl.MIN_OPENING:g}_dt{fl.MAX_DT_S:g}_reg{fl.REGION_MIN:g}"
        out = LAB / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "PARAMETERS.md").write_text(
            f"# {name}\n\nrun_flights.py --full: declared distance flights of {sorted(cl.FOLDERS)} kept by the cleaning of the old "
            f"repository (derived/fixes.parquet, positions from the local frame of flights_meta to Lambert-93); v = {fl.V_KMH:g} "
            f"km/h, opening >= {fl.MIN_OPENING:g}, median dt <= {fl.MAX_DT_S:g} s, region >= {fl.REGION_MIN:g}, {fl.N_BOOT} "
            f"bootstrap resamples; crest data {data}. Chunks of {CHUNK} row groups in chunks/. Written by flights.py.\n")
        results = full_run(cfg, a.jobs, out)
        items = []
    else:
        out = Path(__file__).parent / "out/sample"
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(SEED)
        items = []
        for disc in cl.FOLDERS:
            ids = sorted(declared_open(disc) & set(cl.frames(disc)))
            pick = set(rng.choice(ids, a.sample // 2, replace=False))
            items += list(cl.stream(disc, pick, 0, cl.n_row_groups(disc)))
        results = fl.run(items, cfg, a.jobs)
    fl.finish(results, out, data)
    t = pd.DataFrame([r["info"] for r in results])
    print(t.groupby(["discipline", "status"]).size().unstack(fill_value=0).to_string())
    print(t[t["status"] == "ok"].groupby(["discipline", "region"]).size().to_string())
    if a.plot:
        plot_tracks(items, results, cfg, out, a.plot)
