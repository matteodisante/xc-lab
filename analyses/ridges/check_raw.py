"""Raw IGC tracks against the cleaned fixes, on the paragliders of a sample run of run_flights.py.

The same flights are run through flights.py from the raw IGC file (raw_igc.py: cut to the air) and from the
cleaned fixes (the sample's flights_rows_*.npz). H of dyn and route is fitted by the rule of decompose.py and on
the fixed windows, at v = V_KMH, with the same bootstrap resamples of flights for both, so that the difference
has its own interval.

    uv run python check_raw.py [--sample out/sample] [--out FILE.json]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cleaned as cl
import decompose as dc
import flights as fl
import raw_igc as ri
import run_flights as rf

CURVES = ("dyn", "route")
DISC = "paragliders"  # the discipline of the report


def raw_rows(sample, cfg):
    """TAMSD rows of the sample's flights computed from the raw IGC files: {(discipline, flight): rows by kind}."""
    out = {}
    for disc, folder in cl.FOLDERS.items():
        c = pd.read_csv(rf.DISK / folder / "catalog/catalog.csv", dtype=str, usecols=["flight_id", "local_path"])
        paths = dict(zip(c["flight_id"], c["local_path"]))
        for f in sample.loc[sample["discipline"] == disc, "flight"]:
            if not isinstance(paths.get(f), str):
                continue
            raw = ri.raw_flight(rf.DISK / "xc-viewer-data" / disc / paths[f])
            if raw is None:
                continue
            r = fl.process({"flight": f, "discipline": disc, **raw})
            if r["rows"] is not None:
                out[(disc, f)] = r
    return out


def compare(sample_dir, v_kmh=fl.V_KMH, n_boot=dc.N_BOOT, seed=1):
    sample_dir = Path(sample_dir)
    data = rf.newest_data()
    cfg = fl.configure(data, max_dt=5.0, pauses="turn")  # raw tracks carry no phases: the sample must use the same
    fl._init(cfg)
    flights = pd.read_csv(sample_dir / "flights.csv", dtype={"flight": str})
    flights = flights[flights["discipline"] == DISC]
    z = {k: np.load(sample_dir / f"flights_rows_{k}.npz") for k in CURVES}
    clean = {(d, f): i for i, (d, f) in enumerate(zip(z["dyn"]["discipline"], z["dyn"]["flight"])) if d == DISC}
    raw = raw_rows(flights, cfg)
    keys = sorted(set(clean) & set(raw))
    rows = {("clean", k): np.array([z[k]["rows"][clean[x]] for x in keys]) for k in CURVES}
    rows |= {("raw", k): np.array([raw[x]["rows"][k] for x in keys]) for k in CURVES}
    v = v_kmh / 3.6
    tau = {"dyn": fl.LAGS_T.astype(float), "route": fl.LAGS_K * fl.STEP_M / v}
    counts = dc.draw_counts(len(keys), n_boot, np.random.default_rng(seed))
    lnm = {k: dc.log_means(r, np.arange(len(keys)), counts, min_rows=fl.MIN_PIECES) for k, r in rows.items()}
    tau_max = min(dc.last_lag(lnm[(s, k)], tau[k]) for s in ("raw", "clean") for k in CURVES)
    out = {"n_in_sample": len(flights), "n_clean": len(clean), "n_raw": len(raw), "n_both": len(keys), "v_kmh": v_kmh, "tau_max": tau_max,
           "h": {}}
    for k in CURVES:
        fits = {}
        for s in ("raw", "clean"):
            fits[s] = {"rule": dc.rule_fit(lnm[(s, k)], tau[k], dc.FLOOR_M / v, tau_max)[0]}
            for lo, hi in fl.WINDOWS_M:
                fits[s][fl.fmt_window(lo, hi)] = fl.slope_h(lnm[(s, k)], tau[k], lo / v, hi / v)
        out["h"][k] = {w: {"raw": dc.ci(fits["raw"][w]), "clean": dc.ci(fits["clean"][w]), "raw_minus_clean": dc.ci(fits["raw"][w] - fits["clean"][w])}
                       for w in fits["raw"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", default=str(Path(__file__).parent / "out/sample"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = compare(a.sample)
    text = json.dumps(res, indent=1, default=float)
    (Path(a.out) if a.out else Path(a.sample) / "rawcheck.json").write_text(text)
    print({k: v for k, v in res.items() if k != "h"})
    for k, d in res["h"].items():
        for w, x in d.items():
            print(k, w, {s: [round(y, 3) for y in v] for s, v in x.items()})
