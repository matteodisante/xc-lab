"""The figures of the note on the tuning (parameter_tuning.tex), from the runs on SSD_DISANTE.

    uv run python tuning.py    # from this folder; writes figures/tuning_*.pdf

- tuning_fdr: intervals lasting at least tau, real and by chance, and the false discovery
  rate, for delta = 5 min and 2 min;
- tuning_delays: the delay profile of delays.py (real and chance per delay, their ratio,
  the distance at the same instant);
- tuning_ell: the median delay of the pair intervals by duration (delta = 5 min);
- tuning_q: tau* as a function of q, and the flights ever in a group for three q.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import delays
import run

FIGURES = Path(__file__).with_name("figures")
RUNS = run.DATA / "runs"
WIDE = "d300_eh500_ez300_th45_g120_q0.05"  # delta = 5 min, the starting values of the note
NARROW = "d120_eh500_ez300_th45_g120_q{}"  # delta = 2 min
PROFILE = run.DATA / "delays" / "sample300_d900_eh500_ez300_th45_g120_q0.05" / "profile.json"
QS = (0.05, 0.1, 0.2)
CLASSES = ((0, 10), (10, 30), (30, 90), (90, np.inf))  # durations of the intervals, min
BLUE, ORANGE, INK, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def calibration(name):
    return json.loads((RUNS / name / "calibration.json").read_text())


def tau_star(c, q):
    """The first tau whose FDR is at most q, s."""
    fdr = np.array(c["fdr"])
    return float(np.array(c["tau"])[np.argmax(fdr <= q)]) if (fdr <= q).any() else np.nan


def ell_classes(intervals):
    """|median delay| of the intervals of each class of duration."""
    iv = intervals[intervals["ell_med"].notna()]
    minutes = iv["duration"] / 60
    return {(lo, hi): iv.loc[(minutes >= lo) & (minutes < hi), "ell_med"].abs().to_numpy() for lo, hi in CLASSES}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fdr_figure(wide, narrow):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for c, style, label in ((wide, "-", r"$\delta$ = 5 min"), (narrow, "-", r"$\delta$ = 2 min")):
        tau = np.array(c["tau"]) / 60
        keep = tau <= 120
        colour = GREY if label.endswith("5 min") else BLUE
        left.plot(tau[keep], np.array(c["real"])[keep], color=colour, ls="-", label=f"observed, {label}")
        left.plot(tau[keep], np.array(c["chance"])[keep], color=colour, ls="--", label=f"by chance, {label}")
        right.plot(tau[keep], np.array(c["fdr"])[keep], color=colour, label=label)
    for q in QS:
        right.axhline(q, color=GREY, ls=":", lw=0.8)
        right.text(119, q, f"q = {q}", ha="right", va="bottom", fontsize=8, color=GREY)
        right.plot(tau_star(narrow, q) / 60, q, "o", color=BLUE, ms=4)
    left.set(yscale="log", xlabel=r"$\tau$ (min)", ylabel=r"pair intervals lasting at least $\tau$")
    left.legend(frameon=False, fontsize=8)
    right.set(xlabel=r"$\tau$ (min)", ylabel=r"$\widehat{\mathrm{FDR}}(\tau)$", ylim=(0, 0.85))
    right.legend(frameon=False, fontsize=8)
    save(fig, "tuning_fdr")


def ell_figure(classes):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bins = np.arange(0, 301, 15)
    for (lo, hi), colour in zip(classes, ("#9a9893", "#eb6834", "#1baf7a", "#2a78d6"), strict=True):
        x = classes[lo, hi]
        label = f"{lo}–{hi:g} min" if np.isfinite(hi) else f"$\\geq$ {lo} min"
        ax.hist(x, bins, weights=np.full(len(x), 1 / len(x)), histtype="step", lw=1.5, color=colour,
                label=f"{label} ({len(x):,})")
    ax.set(xlabel="|median delay| of the interval (s)", ylabel="share of the intervals")
    ax.legend(frameon=False, fontsize=8, title="duration of the interval", title_fontsize=8)
    save(fig, "tuning_ell")


def q_figure(wide, narrow, counts):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    qs = np.linspace(0.02, 0.4, 77)
    for c, colour, label in ((wide, GREY, r"$\delta$ = 5 min"), (narrow, BLUE, r"$\delta$ = 2 min")):
        left.plot(qs, [tau_star(c, q) / 60 for q in qs], color=colour, label=label)
    for q in QS:
        left.axvline(q, color=GREY, ls=":", lw=0.8)
    left.set(xlabel="q", ylabel=r"$\tau^*$ (min)")
    left.legend(frameon=False)
    taus = [tau_star(narrow, q) / 60 for q in QS]
    right.plot(taus, [counts[q] for q in QS], "o-", color=BLUE)
    for q, x, y in zip(QS, taus, [counts[q] for q in QS], strict=True):
        right.annotate(f"q = {q}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    right.set(xlabel=r"$\tau^*$ (min), $\delta$ = 2 min", ylabel="flights ever in a group")
    save(fig, "tuning_q")


def group_flights(name):
    f = pd.read_parquet(RUNS / name / "flights.parquet").merge(
        pd.read_parquet(RUNS / name / "flight_summary.parquet"), on="flight")
    return int((f["n_entries"] > 0).sum())


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    wide, narrow = calibration(WIDE), calibration(NARROW.format(0.05))
    fdr_figure(wide, narrow)
    delays.plot(json.loads(PROFILE.read_text()), FIGURES / "tuning_delays.pdf")
    (FIGURES / "tuning_delays.png").unlink()  # delays.plot also writes a PNG
    ell_figure(ell_classes(pd.read_parquet(RUNS / WIDE / "intervals_raw.parquet")))
    q_figure(wide, narrow, {q: group_flights(NARROW.format(q)) for q in QS})
    print("figures written")
