"""Survival of the phase durations of experienced and beginner paragliders, in a group and with no
contact, with a maximum likelihood fit.

Durations as in ../survival.py: Vilpellet's runs on 1-Hz tracks (a duration is
its number of fixes, in seconds), a search only when a climb follows, each run whole in the
population of the stretch that holds its middle, the 0.0001 and 99.9999 percentiles cut. Per
case: mean, median and empirical variance. Every family of families.py is fitted to every case and
scored by its AIC (the bulk of the durations) and by the root-mean-square of log10(model / empirical
survival) at the plotted points that have at least MIN_ABOVE runs above them (the whole range, the tail
included). The law of a phase is the same for its four cases: the family with the lowest log error summed
over them, since the aim is a law for every duration; the AIC is shown beside it. Its parameters come
with 5-95% bands over draws of the dependency blocks.

    uv run python phase_survival.py [--run NAME] [--replot]    # from this folder; writes out/<run>/phase_*
                                                               # (--replot draws the saved fits again)
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from population import COLORS, LEVELS, LABELS, OUT, POPULATIONS, SEED, keep, load, save
from families import FAMILIES, aic, bands
from phase_msd import RUNS
from survival import assign, counted
from tamsd import N_BOOT, stretches
from durations import ccdf, log_resample  # noqa: E402  (survival puts analyses/phases on the path)

PHASES = {"climb": "climb", "search": "search", "transition": "glide"}  # in the runs -> shown
MIN_ABOVE = 20  # runs above a point of the survival for it to enter the log error
STYLES = {"experienced": ("-", None), "beginners": ("--", "white")}  # line style, marker face


def inside(x):
    """Which durations lie strictly inside the 0.0001 and 99.9999 percentiles (durations.trim)."""
    lo, hi = np.percentile(x, [0.0001, 99.9999])
    return (x > lo) & (x < hi)


def durations(runs, classes, alone_flights):
    """Per phase, level and population: the trimmed durations of the runs and their flights."""
    out = {}
    for phase in PHASES:
        out[phase] = {}
        for level in LEVELS:
            mine = keep(runs["flight"].to_numpy(), runs["population"].to_numpy(), classes, alone_flights, level)
            out[phase][level] = {}
            for pop in POPULATIONS:
                r = runs[mine & (runs["phase"] == phase).to_numpy() & (runs["population"] == pop).to_numpy()]
                x = r["n_fixes"].to_numpy(float)
                ok = inside(x)
                out[phase][level][pop] = x[ok], r["flight"].to_numpy()[ok]
    return out


def log_error(x, law):
    """Root-mean-square of log10(model / empirical survival) at the plotted points of the survival of the
    durations x that have at least MIN_ABOVE runs above them."""
    px, py = log_resample(*ccdf(x))
    keep_points = py * len(x) >= MIN_ABOVE
    return float(np.sqrt(np.mean(np.log10(law.sf(px[keep_points]) / py[keep_points]) ** 2)))


def choose(data):
    """Per phase, family and case ("level, population"): the AIC and the log error of the fit. The law of
    each phase is the family with the lowest log error summed over its cases."""
    scores = {phase: {family: {} for family in FAMILIES} for phase in PHASES}
    for phase in PHASES:
        for level in LEVELS:
            for pop in POPULATIONS:
                x = data[phase][level][pop][0]
                values, w = np.unique(x, return_counts=True)
                for family, (estimate, _, frozen) in FAMILIES.items():
                    parameters = estimate(values, w)
                    scores[phase][family][f"{level}, {pop}"] = {
                        "aic": aic(family, values, w, parameters), "log_error": log_error(x, frozen(parameters))}
    total = {phase: {f: sum(c["log_error"] for c in cases.values()) for f, cases in scores[phase].items()}
             for phase in PHASES}
    return {phase: min(total[phase], key=total[phase].get) for phase in PHASES}, scores


def measure(data, block, law, rng):
    """Per phase, level and population: runs, flights, mean, median, variance and the parameters of the
    phase's law with their bands."""
    results = {}
    for phase in PHASES:
        estimate, names, _ = FAMILIES[law[phase]]
        results[phase] = {}
        for level in LEVELS:
            results[phase][level] = {}
            for pop in POPULATIONS:
                x, flights = data[phase][level][pop]
                parameters = estimate(*np.unique(x, return_counts=True))
                band = bands(estimate, x, block.reindex(flights).to_numpy(), rng, N_BOOT)
                results[phase][level][pop] = {
                    "runs": len(x), "flights": int(len(np.unique(flights))), "mean": float(x.mean()),
                    "median": float(np.median(x)), "variance": float(x.var(ddof=1)), "law": law[phase],
                    **{n: float(p) for n, p in zip(names, parameters)}, **{f"{n}_band": b for n, b in zip(names, band)}}
                print(f"{PHASES[phase]:7s} {level:12s} {pop:6s} {len(x):8d} runs, mean {x.mean():6.1f} s, median "
                      f"{np.median(x):5.1f} s, var {x.var(ddof=1):9.3g} s2: {law[phase]} "
                      + ", ".join(f"{n} = {p:.3f} [{b[0]:.3f}, {b[1]:.3f}]" for n, p, b in zip(names, parameters, band)))
    return results


def parameter_text(law, r):
    """What the legend shows of a fit: alpha and beta (the rate) of a gamma, c k (the exponent of the power
    law of the tail) of a Burr XII, else every parameter."""
    if law == "gamma":
        return rf"$\alpha$ = {r['shape']:.2f}, $\beta$ = {1 / r['scale']:.4f} s$^{{-1}}$"
    if law == "Burr XII":
        return rf"$ck$ = {r['c'] * r['k']:.2f}"
    return ", ".join(f"{n} = {r[n]:.3g}" for n in FAMILIES[law][1])


def plot_survival(data, results, law, path):
    """The survival of the durations of each phase, with the fitted law (line) beside the points."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (phase, name) in zip(axes, PHASES.items()):
        estimate, names, frozen = FAMILIES[law[phase]]
        for level in LEVELS:
            line, face = STYLES[level]
            for pop in POPULATIONS:
                x, r = data[phase][level][pop][0], results[phase][level][pop]
                ax.plot(*log_resample(*ccdf(x)), ls="none", marker="o", ms=3.5, color=COLORS[pop],
                        mfc=face or COLORS[pop],
                        label=f"{level}, {pop}: mean {r['mean']:.0f} s, median {r['median']:.0f} s, "
                              f"var {r['variance']:.2g} s$^2$\n{parameter_text(law[phase], r)}")
                t = np.geomspace(x.min(), x.max(), 100)
                ax.plot(t, frozen([r[n] for n in names]).sf(t), color=COLORS[pop], ls=line, lw=1)
        ax.set(xscale="log", yscale="log", title=f"{name}, paragliders: {law[phase]} fit", xlabel=r"$\tau$ (s)",
               ylabel=rf"$P(\tau_{{\mathrm{{{name}}}}} > \tau)$", ylim=(1e-5, 1.5))
        ax.legend(loc="lower left", framealpha=1, edgecolor="none", fontsize=6)
    save(fig, path)


def plot_scores(scores, law, path):
    """Top: the AIC of every family above the lowest of its case (symmetric log axis). Bottom: its log error.
    One column per phase, one point per case."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for col, (phase, name) in enumerate(PHASES.items()):
        for level, pop in [(level, pop) for level in LEVELS for pop in POPULATIONS]:
            case = f"{level}, {pop}"
            aics = np.array([scores[phase][family][case]["aic"] for family in FAMILIES])
            for row, y in enumerate((aics - aics.min(), [scores[phase][f][case]["log_error"] for f in FAMILIES])):
                axes[row, col].plot(y, range(len(FAMILIES)), ls="none", marker="o", ms=6, color=COLORS[pop],
                                    mfc=STYLES[level][1] or COLORS[pop], label=case)
        axes[0, col].set(xscale="symlog", title=f"{name} (law used: {law[phase]})", xlabel="AIC above the lowest of the case")
        axes[0, col].set_xlim(left=-0.5)
        axes[1, col].set(xlabel="log error: rms of log10(model / data survival)", xscale="log")
        axes[1, col].xaxis.set_minor_formatter(NullFormatter())
        for ax in axes[:, col]:
            ax.set_yticks(range(len(FAMILIES)), list(FAMILIES))
    axes[0, 0].set_ylim(len(FAMILIES) - 0.5, -0.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4,
               frameon=False, fontsize=7)
    save(fig, path)


def plot_parameters(results, law, path):
    """The parameters of the law of each phase, in every case, with their 5-95% bands: a row per phase,
    a column per parameter (the unused ones empty)."""
    fig, axes = plt.subplots(len(PHASES), 3, figsize=(11, 7))
    cases = [(level, pop) for level in LEVELS for pop in POPULATIONS]
    for row, phase in zip(axes, PHASES):
        names = FAMILIES[law[phase]][1]
        for col, ax in enumerate(row):
            if col >= len(names):
                ax.axis("off")
                continue
            for i, (level, pop) in enumerate(cases):
                r = results[phase][level][pop]
                lo, hi = r[f"{names[col]}_band"]
                ax.errorbar(r[names[col]], i, xerr=[[r[names[col]] - lo], [hi - r[names[col]]]], color=COLORS[pop],
                            capsize=3, marker="o", mfc=STYLES[level][1] or COLORS[pop])
            ax.set(xlabel=f"{PHASES[phase]}, {law[phase]}: {names[col]}", ylim=(len(cases) - 0.5, -0.5))
            ax.set_yticks(range(len(cases)), [f"{level}, {pop}" for level, pop in cases] if col == 0 else [], fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    save(fig, path)


if __name__ == "__main__":
    summary, classes, alone_flights = load()
    flights = pd.read_parquet(LABELS / "flights.parquet")
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary)
    numbers = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]]
    runs = assign(counted(pd.read_parquet(RUNS["para"])).merge(numbers, on="flight_id"), spans)
    data = durations(runs, classes, alone_flights)
    OUT.mkdir(parents=True, exist_ok=True)
    if "--replot" in sys.argv:
        saved = json.loads((OUT / "phase_survival.json").read_text())
        law, scores, results = saved["law"], saved["scores"], saved["cases"]
    else:
        law, scores = choose(data)
        for phase, families in scores.items():
            print(PHASES[phase], "law used:", law[phase])
            for family, cases in families.items():
                print(f"    {family:18s} AIC minus that of the law used, summed:"
                      f"{sum(c['aic'] for c in cases.values()) - sum(scores[phase][law[phase]][k]['aic'] for k in cases):9.0f}"
                      f"   log error: " + " ".join(f"{c['log_error']:.3f}" for c in cases.values()))
        results = measure(data, summary.set_index("flight")["block"], law, np.random.default_rng(SEED))
        (OUT / "phase_survival.json").write_text(json.dumps({"law": law, "scores": scores, "cases": results}, indent=1))
    plot_survival(data, results, law, OUT / "phase_survival.pdf")
    plot_scores(scores, law, OUT / "phase_scores.pdf")
    plot_parameters(results, law, OUT / "phase_parameters.pdf")
