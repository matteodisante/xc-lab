"""Can 2-s, 3-s and 5-s loggers be labelled? Keep one fix every k seconds of 1-Hz flights,
interpolate back to 1 s, label, and compare with the labels of the full flight on the fixes
both keep. Run from the root of xc-lab: uv run python methods/phase-labelling/cadence.py"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xc_dynamics.preprocessing.flight import preprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases  # noqa: E402

DYNAMICS = HERE.parents[2] / "xc-dynamics"
ROOT = Path("/Volumes/SSD_DATA/paragliders/ffvl_cfd_igc")
TRACKS = Path("/Volumes/SSD_DATA/xc-viewer-data/paragliders")
VARIANTS = [(k, cubic) for cubic in (False, True) for k in (2, 3, 4, 5, 10)]


def preprocessed(path, cfg):
    """The fixes of one paraglider track after every step of the xc-dynamics pre-processing,
    and the row of its counters."""
    return preprocess(path, cfg["altitude"], cfg["speed"]["median_max_kmh"]["para"], cfg["speed"]["max_kmh"]["para"],
                      cfg["excursion"]["window_s"], cfg["vertical"]["max_ms"]["para"], cfg["airborne"],
                      cfg["sampling"]["max_s"])


def max_gap(k):
    # A gap longer than two logging steps, and never shorter than 4 s, ends a piece.
    return max(4.0, 2.0 * k)


def labels_on_fixes(t, xyz, k, cubic, alpha):
    """Label the fixes at times t (cadence k) through a regular 1-s grid; -1 where unlabelled."""
    out = np.full(len(t), -1)
    for grid, vals in phases.regular_grid(t, xyz, max_gap(k), cubic):
        if len(grid) < phases.WINDOW:
            continue
        states, _ = phases.label(grid, *vals, alpha)
        sel = (t >= grid[0]) & (t <= grid[-1])
        out[sel] = states[(t[sel] - grid[0]).astype(int)]
    return out


def one_flight(path, cfg, keep_example=False):
    fixes, row = preprocessed(path, cfg)
    if not (row["gnss_ok"] and row["speed_ok"] and row.get("airborne_ok", False)):
        return [], None
    xyz = np.vstack(phases.enu(fixes.lat.to_numpy(), fixes.lon.to_numpy(), fixes.gnss_alt.to_numpy()))
    t = np.round(fixes.t.to_numpy(float))
    alpha = phases.ALPHA["para"]
    results, example = [], {}
    for seg in np.unique(fixes.segment):
        s = (fixes.segment == seg).to_numpy()
        ts, xs = t[s], xyz[:, s]
        native = labels_on_fixes(ts, xs, 1, False, alpha)
        example.setdefault("t", []).append(ts)
        example.setdefault("xyz", []).append(xs)
        example.setdefault("native", []).append(native)
        for k, cubic in VARIANTS:
            keep = (ts % k) == 0
            coarse = labels_on_fixes(ts[keep], xs[:, keep], k, cubic, alpha)
            ref = native[keep]
            both = (ref >= 0) & (coarse >= 0)
            conf = np.zeros((3, 3), dtype=int)
            np.add.at(conf, (ref[both], coarse[both]), 1)
            results.append(dict(k=k, cubic=cubic, segment=int(seg), compared=int(both.sum()),
                                **{f"c{a}{b}": int(conf[a, b]) for a in range(3) for b in range(3)}))
            if keep_example:
                full = np.full(len(ts), -1)
                full[keep] = coarse
                example.setdefault(f"k{k}{'c' if cubic else 'l'}", []).append(full)
    return results, (example if keep_example else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flights", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    cfg = yaml.safe_load((DYNAMICS / "configs/preprocessing.yaml").read_text())
    table = pd.read_csv(ROOT / "derived/preprocessing/vertical/flights.csv")
    ok = table[(table.sampling_s == 1.0) & table.gnss_ok.astype(bool) & table.speed_ok.astype(bool)
               & (table.fixes_read >= 3600)]
    ids = ok.sample(args.flights, random_state=args.seed).flight_id.tolist()
    paths = pd.read_csv(ROOT / "catalog/catalog.csv", usecols=["flight_id", "local_path"]).set_index("flight_id").local_path
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    rows = []
    for n, fid in enumerate(ids):
        res, example = one_flight(TRACKS / paths[fid], cfg, keep_example=(n == 0))
        rows += [dict(flight_id=fid, **r) for r in res]
        if example is not None:
            np.savez(out / "example.npz", flight_id=fid,
                     **{key: np.concatenate(v, axis=-1) for key, v in example.items()})
        print(f"{n + 1}/{len(ids)} {fid}", flush=True)
    pd.DataFrame(rows).to_csv(out / "cadence.csv", index=False)
    commit = subprocess.run(["git", "-C", str(DYNAMICS), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (out / "cadence.yaml").write_text(yaml.safe_dump(dict(flights=len(ids), seed=args.seed,
                                                          xc_dynamics=commit)))


if __name__ == "__main__":
    main()
