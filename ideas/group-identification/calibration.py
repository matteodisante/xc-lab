"""The figure of the calibration (figure "fdr" of the note) on the flight data: the real
intervals lasting at least tau, those that chance alone would give, and their ratio, the
estimated false discovery rate, with tau*. Also how much tau* moves when the dates are
resampled with replacement (N_BOOT draws), which redraws their surrogates too.

    uv run python calibration.py [RUN]    # from this folder, after run.py; writes RUN/calibration.*

RUN is the folder of a run; by default that of the starting values on SSD_DISANTE.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run

N_BOOT, SEED = 200, 20261021


def per_date(intervals, null, dates, p):
    """Per date and tau: the real intervals of the pairs with takeoffs within r_s lasting at
    least tau, and those chance would give them (rho0 of the date times its H)."""
    near = intervals[intervals["d_dec"] < p["r_s"]]
    durations = dict(tuple(near.groupby("date")["duration"]))
    real = np.zeros((len(dates), len(run.TAUS)), np.float32)
    chance = np.zeros_like(real)
    for k, (date, (exposure, surrogate, null_durations)) in enumerate(zip(dates, null)):
        if date in durations:
            real[k] = run.at_least(durations[date].to_numpy(), run.TAUS)
        if surrogate > 0:
            chance[k] = run.at_least(null_durations, run.TAUS) / surrogate * exposure
    return real, chance


def tau_star(real, chance, q):
    """The first tau whose estimated false discovery rate is at most q (NaN if none), for
    every row of the curves."""
    fdr = np.divide(chance, real, out=np.full(real.shape, np.inf), where=real > 0)
    ok = fdr <= q
    return np.where(ok.any(axis=-1), run.TAUS[ok.argmax(axis=-1)], np.nan)


def band(real, chance, q, rng):
    """tau* of N_BOOT draws of whole dates, with replacement."""
    n = len(real)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float32)
    return tau_star(weights @ real, weights @ chance, q)


def plot(c, draws, path):
    tau = np.array(c["tau"]) / 60
    keep = tau <= 180
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    left.plot(tau[keep], np.array(c["real"])[keep], color="#2a78d6", label=r"observed, $N(\tau)$")
    left.plot(tau[keep], np.array(c["chance"])[keep], color="#eb6834", ls="--",
              label=r"by chance, $\sum_d \rho_{0,d}(\tau) H_d$")
    left.set(yscale="log", xlabel=r"$\tau$ (min)", ylabel=r"intervals lasting at least $\tau$")
    left.legend(frameon=False)
    right.plot(tau[keep], np.array(c["fdr"])[keep], color="#0b0b0b")
    right.axhline(c["params"]["q"], color="#52514e", ls=":", lw=1)
    if c["tau_star"] is not None:
        lo, hi = np.nanpercentile(draws, [5, 95]) / 60
        right.axvspan(lo, hi, color="#52514e", alpha=0.15, lw=0)
        right.axvline(c["tau_star"] / 60, color="#52514e", ls="--", lw=1)
        right.text(c["tau_star"] / 60, 0.9, rf" $\tau^* = {c['tau_star'] / 60:.1f}$ min",
                   transform=right.get_xaxis_transform())
    right.set(xlabel=r"$\tau$ (min)", ylabel=r"$\widehat{\mathrm{FDR}}(\tau)$", ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else run.DATA / "runs" / run.run_name(run.PARAMS)
    c = json.loads((out / "calibration.json").read_text())
    days = out / "days"
    dates = sorted(p.stem for p in days.glob("*.pkl") if not p.name.startswith("."))
    _, intervals, null = run.load(days, dates)
    real, chance = per_date(intervals, null, dates, c["params"])
    assert tau_star(real.sum(axis=0), chance.sum(axis=0), c["params"]["q"]) == c["tau_star"]
    draws = band(real, chance, c["params"]["q"], np.random.default_rng(SEED))
    summary = {"tau_star": c["tau_star"], "draws": N_BOOT, "without_tau_star": int(np.isnan(draws).sum()),
               "percentiles_5_50_95": np.nanpercentile(draws, [5, 50, 95]).tolist()}
    (out / "calibration_band.json").write_text(json.dumps(summary, indent=1))
    plot(c, draws, out / "calibration.pdf")
    print(json.dumps(summary))
