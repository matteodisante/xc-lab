"""The figures of the note assessing the group identification (method_assessment.tex), from the
runs on SSD_DISANTE.

    uv run python assessment.py    # from this folder; writes figures/assessment_*.pdf

- assessment_shifts: the time shifts of the surrogates of the chosen run;
- assessment_fdr: intervals lasting at least tau, real and by chance, and the false discovery
  rate, for delta = 5 min and 2 min;
- assessment_delays: the delay profile of delays.py (real and chance per delay, their ratio,
  the distance at the same instant);
- assessment_ell: the median delay of the pair intervals by duration (delta = 5 min);
- assessment_separation: the distance at the same instant of separation.py, real and chance,
  and their ratio g(d);
- assessment_tolerances: the profiles of tolerances.py (eps_h, eps_z, gap lengths);
- assessment_q: tau* as a function of q, and the flights ever in a group for three q;
- assessment_structure: the durations and the participants of the episodes and of the
  configurations, for the three q with delta = 2 min; assessment_structure_episodes and
  _configurations: one row each, with larger text, for the slides.
"""

import json
import pickle
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
CHOSEN = NARROW.format(0.1)
PROFILE = run.DATA / "delays" / "sample300_d900_eh500_ez300_th45_g120_q0.05" / "profile.json"
SEPARATION = run.DATA / "separation" / "sample300_d120_eh500_ez300_th45_g120_q0.05" / "profile.json"
TOLERANCES = run.DATA / "tolerances" / "sample300_d120_eh500_ez300_th45_g120_q0.05" / "profile.json"
QS = (0.05, 0.1, 0.2)
CLASSES = ((0, 10), (10, 30), (30, 90), (90, np.inf))  # durations of the intervals, min
GAP_EDGES = (5, 10, 20, 40, 80, 160, 320, 640, 1280, 1795)  # s; longer gaps share the last bin of tolerances.py
BLUE, ORANGE, INK, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def calibration(name):
    return json.loads((RUNS / name / "calibration.json").read_text())


def tau_star(c, q):
    """The first tau whose FDR is at most q, s."""
    fdr = np.array(c["fdr"])
    return float(np.array(c["tau"])[np.argmax(fdr <= q)]) if (fdr <= q).any() else np.nan


def shifts(name):
    """|r| of every surrogate pair of a run, s."""
    days = sorted((RUNS / name / "days").glob("[0-9]*.pkl"))  # not the ._ files of macOS
    return np.abs(np.concatenate([pickle.loads(f.read_bytes())["null_pairs"]["r"].to_numpy(float) for f in days]))


def ell_classes(intervals):
    """|median delay| of the intervals of each class of duration."""
    iv = intervals[intervals["ell_med"].notna()]
    minutes = iv["duration"] / 60
    return {(lo, hi): iv.loc[(minutes >= lo) & (minutes < hi), "ell_med"].abs().to_numpy() for lo, hi in CLASSES}


def gap_shares(gap):
    """Share of the gaps in each class of GAP_EDGES and beyond, real and chance."""
    s = np.arange(len(gap["real"])) * gap["bin"]
    edges = np.r_[GAP_EDGES, np.inf]
    out = {}
    for key in ("real", "chance"):
        x = np.array(gap[key])
        out[key] = np.array([x[(s >= lo) & (s < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]) / x.sum()
    return out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def shift_figure(r):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    minutes = r / 60
    ax.hist(minutes, np.logspace(np.log10(minutes.min()), np.log10(minutes.max()), 60),
            weights=np.full(len(r), 1 / len(r)), histtype="step", lw=1.5, color=INK)
    ax.axvline(np.median(minutes), color=BLUE, ls="--", lw=1, label=f"median, {np.median(minutes):.0f} min")
    ax.set(xscale="log", xlabel="|shift| of the surrogate (min)", ylabel="share of the surrogate pairs")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "assessment_shifts")


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
    save(fig, "assessment_fdr")


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
    save(fig, "assessment_ell")


def separation_figure(prof):
    km = np.array(prof["distance_m"])[:-1] / 1000  # the last bin holds the rest
    real, chance = np.array(prof["real"])[:-1], np.array(prof["chance"])[:-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    left.plot(km, real, color=BLUE, label="real pairs")
    left.plot(km, chance, color=ORANGE, ls="--", label="by chance")
    left.set(xscale="log", xlabel="distance at the same instant (km)",
             ylabel=f"share of the exposure per {prof['bin_m']} m")
    left.legend(frameon=False)
    right.plot(km, real / chance, color=INK)
    right.axhline(1, color=GREY, lw=0.8)
    right.set(xscale="log", yscale="log", xlabel="distance at the same instant (km)", ylabel="g(d) = real / chance")
    save(fig, "assessment_separation")


def tolerance_figure(prof):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, chosen, top, label in zip(
            axes[:2], ("horizontal", "vertical"), (500, 300), (3000, 1000),
            ("smallest horizontal distance (m)", "smallest height difference (m)"), strict=True):
        p = prof[key]
        real, chance = np.array(p["real"])[:-1], np.array(p["chance"])[:-1]  # the last bin holds the rest
        x = (np.arange(len(real)) + 0.5) * p["bin"]
        keep = x < top
        ax.plot(x[keep], real[keep] / chance[keep], color=INK)
        ax.axvline(chosen, color=BLUE, ls=":", lw=1)
        ax.axhline(1, color=GREY, lw=0.8)
        ax.set(xlabel=label, ylabel="real / chance", ylim=(0, None))
    shares = gap_shares(prof["gap"])
    for key, colour, style, label in (("real", BLUE, "-", "real pairs"), ("chance", ORANGE, "--", "by chance")):
        axes[2].stairs(shares[key][:-1], GAP_EDGES, color=colour, ls=style, lw=1.5, label=label)
    axes[2].axvline(120, color=BLUE, ls=":", lw=1)
    axes[2].set(xscale="log", xlabel="gap between instants of contact (s)", ylabel="share of the gaps")
    axes[2].legend(frameon=False, fontsize=8)
    save(fig, "assessment_tolerances")


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
    save(fig, "assessment_q")


def structure(name):
    """Durations (min) and participants of the episodes (distinct flights) and of the
    configurations (members at once) of a run."""
    e = pd.read_parquet(RUNS / name / "episodes.parquet")
    c = pd.read_parquet(RUNS / name / "configurations.parquet")
    return {"episodes": ((e["t_end"] - e["t_start"]).to_numpy() / 60, e["n_flights"].to_numpy()),
            "configurations": ((c["t_end"] - c["t_start"]).to_numpy() / 60, c["n"].to_numpy())}


def size_shares(duration, n):
    """Sizes, the share of the items of each size and the share of their total duration."""
    sizes, k = np.unique(n, return_inverse=True)
    return sizes, np.bincount(k) / len(n), np.bincount(k, weights=duration) / duration.sum()


def structure_figure(runs, kinds=("episodes", "configurations"), name="assessment_structure", size=(13, 3.2)):
    fig, axes = plt.subplots(len(kinds), 3, figsize=(size[0], size[1] * len(kinds)), squeeze=False)
    ticks = {"episodes": ((30, 60, 120, 300, 600), (2, 5, 10, 20, 50, 100)),
             "configurations": ((0.1, 1, 10, 100), (2, 5, 10, 20, 40))}
    for q, colour, style in zip(QS, (GREY, BLUE, ORANGE), (":", "-", "--"), strict=True):
        for row, kind in zip(axes, kinds, strict=True):
            duration, n = runs[q][kind]
            x = np.sort(duration)
            row[0].plot(x, 1 - np.arange(len(x)) / len(x), color=colour, ls=style, label=f"q = {q}")
            sizes, by_count, by_time = size_shares(duration, n)
            row[1].plot(sizes, by_count, "o", color=colour, ls=style, ms=3)
            row[2].plot(sizes, by_time, "o", color=colour, ls=style, ms=3)
            size = "flights in the episode" if kind == "episodes" else "pilots in the configuration"
            row[0].set(xscale="log", yscale="log", xlabel=f"duration of the {kind[:-1]} (min)",
                       ylabel="share lasting longer")
            row[1].set(xscale="log", yscale="log", xlabel=size, ylabel=f"share of the {kind}")
            row[2].set(xscale="log", yscale="log", xlabel=size, ylabel=f"share of the time in {kind}")
    for row, kind in zip(axes, kinds, strict=True):
        for ax, t in zip(row, (ticks[kind][0], ticks[kind][1], ticks[kind][1]), strict=True):
            ax.set_xticks(t, [f"{x:g}" for x in t])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
    axes[0, 0].legend(frameon=False, fontsize="small")
    save(fig, name)


def group_flights(name):
    f = pd.read_parquet(RUNS / name / "flights.parquet").merge(
        pd.read_parquet(RUNS / name / "flight_summary.parquet"), on="flight")
    return int((f["n_entries"] > 0).sum())


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    wide, narrow = calibration(WIDE), calibration(NARROW.format(0.05))
    shift_figure(shifts(CHOSEN))
    fdr_figure(wide, narrow)
    delays.plot(json.loads(PROFILE.read_text()), FIGURES / "assessment_delays.pdf")
    (FIGURES / "assessment_delays.png").unlink()  # delays.plot also writes a PNG
    ell_figure(ell_classes(pd.read_parquet(RUNS / WIDE / "intervals_raw.parquet")))
    separation_figure(json.loads(SEPARATION.read_text()))
    tolerance_figure(json.loads(TOLERANCES.read_text()))
    q_figure(wide, narrow, {q: group_flights(NARROW.format(q)) for q in QS})
    runs = {q: structure(NARROW.format(q)) for q in QS}
    structure_figure(runs)
    with plt.rc_context({"font.size": 13}):  # one row per slide of the meeting
        for kind in ("episodes", "configurations"):
            structure_figure(runs, (kind,), f"assessment_structure_{kind}", size=(12, 3.8))
    print("figures written")
