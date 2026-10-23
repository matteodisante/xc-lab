"""tau* by the time shift of the surrogates: the false discovery rate of a run recomputed with
only the surrogate pairs whose |r| falls in each class of CLASSES.

A class takes its surrogate pairs of each date with their exposure S_d and their intervals,
so rho0 of the date is that of the class. The classes are compared on the dates that have
surrogates in every class, "all" included. Bands: tau* over N_BOOT draws of these dates with
replacement, as calibration.py.

    uv run python shifts.py [RUN]    # from this folder, after run.py; writes RUN/calibration_shifts.json

RUN is the folder of a run; by default the chosen one on SSD_DISANTE.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

import calibration
import run

CLASSES = {"all": (0, np.inf), "short": (0, 1800), "long": (3600, np.inf)}  # |r| in [lo, hi), s
AT = (600, 1800)  # durations at which the false discovery rate is reported, s


def per_date(days, dates, r_s):
    """Per date and tau: the real intervals of the pairs with takeoffs within r_s lasting at
    least tau, and for each class those chance would give them; and the surrogate pairs and
    exposure of each class."""
    real = np.zeros((len(dates), len(run.TAUS)), np.float32)
    chance = {c: np.zeros_like(real) for c in CLASSES}
    pairs = {c: np.zeros(len(dates), int) for c in CLASSES}
    exposure = {c: np.zeros(len(dates)) for c in CLASSES}
    for k, date in enumerate(dates):
        r = pickle.loads((days / f"{date}.pkl").read_bytes())
        iv = r["intervals"]
        real[k] = run.at_least((iv["t_off"] - iv["t_on"])[iv["d_dec"] < r_s].to_numpy(), run.TAUS)
        null = r["null_pairs"].assign(shift=lambda x: x["r"].abs())
        durations = r["null_intervals"].merge(null[["i", "j", "shift"]], on=["i", "j"])
        for c, (lo, hi) in CLASSES.items():
            mine = null[(null["shift"] >= lo) & (null["shift"] < hi)]
            pairs[c][k], exposure[c][k] = len(mine), mine["exposure"].sum()
            if exposure[c][k] > 0:
                d = durations.loc[(durations["shift"] >= lo) & (durations["shift"] < hi), "duration"].to_numpy()
                chance[c][k] = run.at_least(d, run.TAUS) / exposure[c][k] * r["exposure"]
    return real, chance, pairs, exposure


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else run.DATA / "runs" / "d120_eh500_ez300_th45_g120_q0.1"
    p = json.loads((out / "calibration.json").read_text())["params"]
    q = p["q"]
    days = out / "days"
    dates = sorted(f.stem for f in days.glob("*.pkl") if not f.name.startswith("."))
    real, chance, pairs, exposure = per_date(days, dates, p["r_s"])
    keep = np.all([exposure[c] > 0 for c in CLASSES], axis=0)
    summary = {"q": q, "dates": int(keep.sum()), "of_dates": len(dates), "classes": {}}
    for c, (lo, hi) in CLASSES.items():
        n, x = real[keep].sum(axis=0), chance[c][keep].sum(axis=0)
        fdr = np.divide(x, n, out=np.full(len(n), np.inf), where=n > 0)
        draws = calibration.band(real[keep], chance[c][keep], q, np.random.default_rng(calibration.SEED))
        summary["classes"][c] = {
            "shift_s": [lo, None if np.isinf(hi) else hi], "pairs": int(pairs[c][keep].sum()),
            "fdr_at": {str(t): float(fdr[np.searchsorted(run.TAUS, t)]) for t in AT},
            "tau_star": float(calibration.tau_star(n, x, q)),
            "percentiles_5_50_95": np.nanpercentile(draws, [5, 50, 95]).tolist()}
    (out / "calibration_shifts.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
