"""Ch. 7 of the story: the phases of S, A and G per ground class, and the gaps G - A, A - S, G - S.

Populations as in contrasts.py (declared open paragliders, logger interval up to 5 s, `--dt5`): the pieces of S, A, G of
the cache of longlags.py, per ground class of ground.py and "all". The runs are those of the label set `--labels
refit|vilpellet` (phase_terms.py); a run belongs whole to the population and the class of the piece that holds its
middle (as raw.py). Unit = the flight part in a population; the statistics STATS are the estimator of budget.py
("average first, ratio after"); the bands are those of twoway.py, with the draws that contrasts.py uses, so that a
gap has the band of its own draws. The cycle of a glide and a climb is 60 / climbs per hour, in minutes. PHYSICAL adds the
number per hour and the mean duration of each phase, and the searches found after a climb (refound); KINDS the share of the
runs and of the time of each kind of search run (found, refound, failed, exit), which add up to 100%.

Durations of glides, of searches (found, refound and failed) and of climbs: the survival of S, A, G and the tail exponent
mu above one threshold per phase and class, the largest of the three Kolmogorov-Smirnov thresholds, chosen again in
every draw (tail_ks.py); the Hill mu(xmin) of each population on a grid of thresholds, and the mean and the median of the
same durations. The first row of the arrays of draws is the point estimate. For "all", powerlaw.Fit is run on each population
as a check of the scan.

    uv run python phase_gaps.py --declared-open --dt5 [--labels NAME] [--draws N]   # writes phases_<labels>.json
    # --rule context|plain (default context, see phase_terms.py): the files of the rule plain end in _plain
    uv run python phase_gaps.py --declared-open --dt5 [--labels NAME] --draw       # the figures, from the JSON
    uv run python phase_gaps.py --declared-open --dt5 --labels refit --common      # the refit on the flights his labels cover
    uv run python phase_gaps.py --declared-open --dt5 --compare                    # the label sets side by side
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget
import isolated
import longlags
import phase_terms
import raw
import survival
import tail
import tail_ks
import twoway
from blocks import with_takeoffs
from contrasts import ALL_COLOR, CLASS_COLORS, CLASSES
from tamsd import COLORS, LABELS, N_BOOT, OUT, SEED, read_flights, save

sys.path.insert(0, str(Path(__file__).parents[1] / "phases"))
from durations import fit, hill, trim  # noqa: E402

PLAIN = phase_terms.PLAIN  # "_plain" for --rule plain, the old rule: the outputs carry the suffix
# name: numerator, denominator (columns of phase_terms.table), scale, decimals shown
COMPOSITION = {
    "time in glide (%)": ("transition_s", "s", 100, 1),
    "time in search (%)": ("search_s", "s", 100, 1),
    "time in climb (%)": ("climb_s", "s", 100, 1),
    **({} if PLAIN else {"time in exit (%)": ("exit_s", "s", 100, 1)}),
    "climbs per hour": ("climbs", "hours", 1, 2),
    **({} if PLAIN else {"searches per hour": ("searches", "hours", 1, 2), "exits per hour": ("exits", "hours", 1, 2)}),
}
PERFORMANCE = {
    "glide speed (km/h)": ("chord_m", "glide_t", 3.6, 1),
    "height lost per km of glide (m/km)": ("loss_m", "chord_m", 1000, 0),
    "searches that end in a climb (%)": ("found", "searches", 100, 1),
    "climb rate (m/s)": ("gain_m", "climb_t", 1, 2),
    "distance gained per climb (m)": ("drift_m", "climbs_m", 1, 0),
    "height gained per climb (m)": ("gain_m", "climbs_m", 1, 0),
    "drift speed of a climb (m/s)": ("drift_m", "climb_t", 1, 2),
    **({} if PLAIN else {"mean exit duration (s)": ("exit_s", "exits", 1, 0)}),
}
STATS = {**COMPOSITION, **PERFORMANCE}
# Per hour and mean duration of each phase (a search is a found, refound or failed one; the mean duration is the time over the
# count, average first, ratio after) and the refound searches; kept apart from STATS, which redundancy.py reads.
PHYSICAL = {} if PLAIN else {
    "glides per hour": ("glides", "hours", 1, 2), "mean glide duration (s)": ("transition_s", "glides", 1, 0),
    "mean search duration (s)": ("search_s", "searches", 1, 0), "mean climb duration (s)": ("climb_s", "climbs", 1, 0),
    "refound searches per hour": ("refound", "hours", 1, 2), "mean refound duration (s)": ("refound_s", "refound", 1, 0),
    "found searches that follow a climb (%)": ("refound", "found", 100, 1)}
# The four kinds of search run (found after a glide, refound after a climb, failed, exit) as shares of their runs and of their time.
KINDS = {} if PLAIN else {f"{k}: share of {w} (%)": (c + ("_s" if w == "time" else ""), "kinds" + ("_s" if w == "time" else ""), 100, 1)
                          for w in ("runs", "time") for k, c in (("found", "glide_found"), ("refound", "refound"), ("failed", "failed"),
                                                                 ("exit", "exit" if w == "time" else "exits"))}
MEASURED = {**STATS, **PHYSICAL, **KINDS}
CYCLE = "cycle of a glide and a climb (min)"  # 60 / climbs per hour
EXIT_MEDIAN = "median exit duration (s)"
DERIVED = [CYCLE] + ([] if PLAIN else [EXIT_MEDIAN])
PHASES = {"transition": "glide", "search": "search that ends in a climb" if PLAIN else "search", "climb": "climb"}
CLASS_KEYS = (*CLASSES, "all")
POPS = {"solo": "S", "group_solo": "A", "group": "G"}  # the order of the draws
GAPS = {"G - A": ("G", "A"), "A - S": ("A", "S"), "G - S": ("G", "S")}
MAIN_GAPS = ("G - A", "A - S")
GAP_COLORS = {"G - A": raw.GAP_COLORS["G - A"], "A - S": raw.GAP_COLORS["A - S"]}
HILL_STEP = 2  # every HILL_STEP-th candidate threshold of tail_ks is drawn in the Hill plot
LABELS_NAME = phase_terms.SET
DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else N_BOOT
COMMON = "--common" in sys.argv  # only the flights that Vilpellet's labels cover (1 Hz): the refit on his flights
NAME = f"phases_{LABELS_NAME}" + ("_common" if COMMON else "") + PLAIN + (f"_draws{DRAWS}" if DRAWS != N_BOOT else "")
CHECK_CLASS = "all"  # the class on which powerlaw.Fit checks the scan


# ---- the runs of each population and class ----

def context():
    """The pieces and their populations (masks), the cell code of each flight, the draws (the first DRAWS of each of the
    three groups) and the ground class of each flight, as in contrasts.context."""
    flights = with_takeoffs(read_flights())
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    code, pilot, block = twoway.cells(flights, summary)
    counts = twoway.counts(pilot, block, np.random.default_rng(SEED))
    counts = np.vstack([counts[g * N_BOOT:g * N_BOOT + DRAWS] for g in range(3)])
    p = longlags.load_pieces()
    return p, code, counts, pd.read_parquet(OUT / "ground.parquet").set_index("flight")["class"]


def assign(runs, p, classes):
    """`runs` (phase_terms.table) with the population (0, 1, 2: S, A, G), the class and the piece of each run, for the
    runs whose middle lies in a piece of S, A or G; the others are dropped."""
    label = pd.Series(p["population"])
    keep = label.isin(list(POPS)).to_numpy()
    flight, a, b = p["flight"][keep], p["start"][keep], p["start"][keep] + p["duration"][keep]
    order = np.argsort(flight * isolated.RANK + a, kind="stable")
    start, end = (flight * isolated.RANK + a)[order], (flight * isolated.RANK + b)[order]
    key = runs["flight"].to_numpy(float) * isolated.RANK + runs["t_mid"].to_numpy(float)
    i = np.maximum(np.searchsorted(start, key, "right") - 1, 0)
    hit = (start[i] <= key) & (key < end[i])
    pop = label[keep].map({k: j for j, k in enumerate(POPS)}).to_numpy()
    runs = runs[hit].assign(pop=pop[order[i[hit]]])
    return runs.assign(cls=classes.reindex(runs["flight"]).fillna("unknown").to_numpy())


def stratum(runs, key):
    """The runs of the class `key` of CLASS_KEYS."""
    return runs if key == "all" else runs[runs["cls"] == key]


# ---- the statistics ----

def unit_statistics(rows, code, counts):
    """The STATS and PHYSICAL of the flights of `rows` (runs of one population): their values and the rows of draws, as an array
    (1 + draws, statistics + derived), the derived being DERIVED (the cycle, and with the rule `context` the median
    exit duration); and the number of flights."""
    time = rows.groupby("flight")["s"].sum()
    time = time[time > 0]  # a flight part whose runs were all dropped has no unit
    r, q = budget.unit_rates(rows, rows["flight"], time, MEASURED)
    value, draws = budget.mean_ratios(r, q, code.reindex(time.index).to_numpy(), counts, MEASURED)
    out = np.vstack([value.to_numpy(), draws])
    derived = [60 / out[:, list(STATS).index("climbs per hour")]]
    if not PLAIN:
        exits = rows[rows["phase"] == "exit"]
        cell = code.reindex(exits["flight"]).to_numpy()
        derived.append(np.r_[tail.median_draws(exits["s"].to_numpy(float), cell, np.ones((1, counts.shape[1]), np.uint8)),
                             tail.median_draws(exits["s"].to_numpy(float), cell, counts)] if len(exits) else np.full(len(out), np.nan))
    return np.column_stack([out, *derived]), len(time)


def quotes(rows, names):
    """Each column of `rows` (1 + draws, names) as {name: {value, band}}."""
    return {n: {"value": float(rows[0, j]), "band": [float(v) for v in twoway.band(rows[0, j], rows[1:, j])]} for j, n in enumerate(names)}


def contexts(mine):
    """The searches of the runs `mine` by context (rule `context`): found, refound (after a climb), failed, exit and unknown
    neighbour (dropped), as counts and as shares of all the searches."""
    n = {k: int((mine["phase"] == k).sum()) for k in (*phase_terms.SEARCHES, "exit", "unknown")}
    return {"count": n, "share": {k: v / sum(n.values()) for k, v in n.items()}}


def phase_statistics(runs, code, counts):
    """Per class: the flights and hours of S, A, G, their statistics (STATS, PHYSICAL and DERIVED) and the gaps of GAPS; with
    the rule `context`, also the searches by context."""
    names = [*MEASURED, *DERIVED]
    out = {}
    for key in CLASS_KEYS:
        mine = stratum(runs, key)
        got = {POPS[k]: unit_statistics(mine[mine["pop"] == i], code, counts) for i, k in enumerate(POPS)}
        rows = {pop: v[0] for pop, v in got.items()}
        out[key] = {"flights": {pop: v[1] for pop, v in got.items()},
                    "hours": {POPS[k]: float(mine.loc[mine["pop"] == i, "s"].sum() / 3600) for i, k in enumerate(POPS)},
                    "statistics": {pop: quotes(r, names) for pop, r in rows.items()},
                    "gaps": {gap: quotes(rows[a] - rows[b], names) for gap, (a, b) in GAPS.items()}}
        if not PLAIN:
            out[key]["contexts"] = {**{POPS[k]: contexts(mine[mine["pop"] == i]) for i, k in enumerate(POPS)}, "all": contexts(mine)}
        print(key, out[key]["flights"], flush=True)
    return out


# ---- the durations ----

def counted(rows, phase):
    """The durations and the cell code (flight) of the runs of `phase` that the survival counts (with the rule `plain` a
    search only when it ends in a climb, with `context` the found, refound and failed ones), trimmed as durations.trim."""
    if PLAIN:
        r = rows[(rows["phase"] == phase) & ((phase != "search") | (rows["found"] == 1))]
    else:
        r = rows[rows["phase"].isin(phase_terms.SEARCHES if phase == "search" else (phase,))]
    x = r["s"].to_numpy(float)
    keep = np.isin(x, trim(x)) if len(x) else np.zeros(0, bool)
    return x[keep], r["flight"].to_numpy()[keep]


def rows_of(v):
    """The point estimate and the draws (a vector, the point first) as {value, band}."""
    return {"value": float(v[0]), "band": [float(x) for x in twoway.band(v[0], v[1:])]}


def weighted_means(x, cell, counts, step=250):
    """The mean of the durations `x` (cell codes `cell`) in the point estimate (row 0) and in each draw, a run weighing the count
    of its cell: an array (1 + rows of `counts`)."""
    top, size = np.bincount(cell, x, minlength=counts.shape[1]), np.bincount(cell, minlength=counts.shape[1])
    return np.r_[top.sum() / size.sum(), *[(counts[a:a + step] @ top) / (counts[a:a + step] @ size) for a in range(0, len(counts), step)]]


def duration_summary(samples, counts):
    """The mean and the median of the durations of S, A, G (`samples` {population: (durations, cell codes)}) and their gaps, as
    {statistic: {"values": {population: {value, band}}, "gaps": {gap: {value, band}}}}, the draws of both sides paired."""
    ones = np.ones((1, counts.shape[1]), np.uint8)
    draws = {"mean": {k: weighted_means(x, c, counts) for k, (x, c) in samples.items()},
             "median": {k: np.r_[tail.median_draws(x, c, ones), tail.median_draws(x, c, counts)] for k, (x, c) in samples.items()}}
    return {name: {"values": {k: rows_of(v) for k, v in d.items()}, "gaps": {gap: rows_of(d[a] - d[b]) for gap, (a, b) in GAPS.items()}}
            for name, d in draws.items()}


def tail_statistics(rows_by_pop, code, counts):
    """The tail of one phase of one class: `rows_by_pop` {S, A, G: runs of the phase}. The survival of each population, its
    mu above the common threshold of the draw (thresholds of tail_ks), the gaps of mu, the thresholds, the Hill mu(xmin), and
    the mean and the median of the durations (duration_summary)."""
    cells = counts.shape[1]
    samples = {}
    for pop, (x, flight) in rows_by_pop.items():
        samples[pop] = (x, code.reindex(flight).to_numpy())
    if min(len(x) for x, _ in samples.values()) < tail_ks.MIN_TAIL:
        return None
    point = tail_ks.thresholds(samples, np.ones((1, cells)), cells)
    drawn = tail_ks.thresholds(samples, counts, cells)
    mu = {k: np.r_[point["mu"][k], drawn["mu"][k]] for k in samples}
    xmin = np.r_[point["xmin"], drawn["xmin"]]
    pooled = np.concatenate([x for x, _ in samples.values()])
    grid = tail_ks.edges_of(pooled)[tail_ks.candidates(pooled, tail_ks.edges_of(pooled))][::HILL_STEP]
    return {"runs": {k: len(x) for k, (x, _) in samples.items()},
            "xmin": {"common": float(xmin[0]), "common_5_95": np.nanpercentile(xmin[1:], [5, 95]).tolist(),
                     "own": {k: float(point["own"][k][0]) for k in samples},
                     "own_5_95": {k: np.nanpercentile(drawn["own"][k], [5, 95]).tolist() for k in samples}},
            "runs_above": {k: int(point["n"][k][0]) for k in samples},
            "mu": {k: rows_of(v) for k, v in mu.items()},
            "gaps": {gap: rows_of(mu[a] - mu[b]) for gap, (a, b) in GAPS.items()},
            "durations": duration_summary(samples, counts),
            "survival": {k: survival.survival_curve(x, c, counts, band=twoway.share_band) for k, (x, c) in samples.items()},
            "hill": {k: tail.mu_curve(x, c, counts, grid) for k, (x, c) in samples.items()}}


def powerlaw_check(by_pop, own):
    """powerlaw.Fit of each population (its xmin, mu and runs above) next to the scan alone: its threshold `own` ({population:
    xmin}) and the Hill mu there."""
    out = {}
    for k, (x, _) in by_pop.items():
        f = fit(x)
        out[k] = {"powerlaw": f, "scan": {"xmin": own[k], "mu": hill(x, own[k]), "n_tail": int((x >= own[k]).sum())}}
    return out


def duration_statistics(runs, code, counts):
    """Per class and phase: tail_statistics of the runs of S, A, G."""
    out = {}
    for key in CLASS_KEYS:
        mine = stratum(runs, key)
        out[key] = {}
        for phase in PHASES:
            by_pop = {POPS[k]: counted(mine[mine["pop"] == i], phase) for i, k in enumerate(POPS)}
            out[key][phase] = tail_statistics(by_pop, code, counts)
            if out[key][phase] and key == CHECK_CLASS:
                out[key][phase]["powerlaw"] = powerlaw_check(by_pop, out[key][phase]["xmin"]["own"])
            print(key, phase, None if out[key][phase] is None else out[key][phase]["xmin"]["common"], flush=True)
    return out


# ---- the cycle against the lag of the peak ----

def peaks():
    """The lag (s) of the peak of the normalised gaps G - A and G - S of contrasts.py, per class, where selection-free
    (as comparisons.ground_tex), or {} when contrasts_ground.json is not there."""
    path = longlags.RESULTS / "contrasts_ground.json"
    if not path.exists():
        return {}
    s = json.loads(path.read_text())
    out = {}
    for key in CLASS_KEYS:
        out[key] = {}
        for gap in ("G - A", "G - S"):
            r = s[key]["relative gaps"][gap]
            i = max((t for t, ok in enumerate(r["solid"]) if ok), key=lambda t: r["value"][t])
            out[key][gap] = {"lag": r["lags"][i], "value": r["value"][i]}
    return out


# ---- the figures ----

def forest(ax, result, stat, key="statistics"):
    """The gaps G - A and A - S of `stat` per class (rows): the value with its band."""
    ys = np.arange(len(CLASS_KEYS))
    for off, gap in zip((-0.15, 0.15), MAIN_GAPS):
        for y, c in zip(ys, CLASS_KEYS):
            g = result["phases"][c]["gaps"][gap][stat]
            ax.errorbar(g["value"], y + off, xerr=[[g["value"] - g["band"][0]], [g["band"][1] - g["value"]]], fmt="o", ms=4,
                        color=GAP_COLORS[gap], elinewidth=1.3, capsize=2, label=gap if y == 0 else None)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(ys, CLASS_KEYS)
    ax.invert_yaxis()
    ax.set_title(stat, fontsize=8)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    ax.tick_params(labelsize=7)


def draw_forests(result, stats, path, per_row=3):
    """One forest panel per statistic of `stats`."""
    rows = -(-len(stats) // per_row)
    fig, axes = plt.subplots(rows, min(per_row, len(stats)), figsize=(3.2 * min(per_row, len(stats)), 2.8 * rows), squeeze=False,
                             sharey=True, constrained_layout=True)
    for ax, stat in zip(axes.flat, stats):
        forest(ax, result, stat)
    for ax in axes.flat[len(stats):]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=7)
    save(fig, path)


def kept_stats(group):
    """The statistics of `group` that redundancy.py keeps, or all of them when it has not been run."""
    path = longlags.RESULTS / f"phases_redundancy_{LABELS_NAME}{PLAIN}.json"
    kept = json.loads(path.read_text())["kept"] if path.exists() else list(STATS)
    return [s for s in group if s in kept]


def draw_tails(result, path):
    """Rows: the phases; columns: the classes. The survival of S, A, G with its band and the common threshold (dotted)."""
    fig, axes = plt.subplots(len(PHASES), len(CLASS_KEYS), figsize=(2.7 * len(CLASS_KEYS), 2.4 * len(PHASES)), constrained_layout=True,
                             sharey="row")
    for i, phase in enumerate(PHASES):
        for j, c in enumerate(CLASS_KEYS):
            ax, t = axes[i, j], result["tails"][c][phase]
            if t is None:
                ax.axis("off")
                continue
            for pop, short in POPS.items():
                s, mu = t["survival"][short], t["mu"][short]
                g, v, b = np.array(s["grid"]), np.array(s["value"]), np.array(s["band"])
                ok = v > 0
                ax.fill_between(g[ok], np.maximum(b[0][ok], 1e-9), b[1][ok], color=COLORS[pop], alpha=0.2, lw=0)
                ax.plot(g[ok], v[ok], color=COLORS[pop], lw=1.2,
                        label=rf"{short}: $\mu$ = {mu['value']:.2f} [{mu['band'][0]:.2f}, {mu['band'][1]:.2f}]")
            ax.axvline(t["xmin"]["common"], color="#888888", lw=0.8, ls=":")
            ax.set(xscale="log", yscale="log", ylim=(1e-5, 1.2))
            ax.set_title(f"{PHASES[phase]}, {c}", fontsize=8)
            ax.legend(frameon=False, fontsize=6, loc="lower left")
            ax.grid(color="#eeeeee", lw=0.5)
            ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau$ (s)", fontsize=8)
    save(fig, path)


def draw_hill(result, path):
    """The Hill mu(xmin) of S, A, G with bands: rows the phases, columns the classes; the common threshold dotted."""
    fig, axes = plt.subplots(len(PHASES), len(CLASS_KEYS), figsize=(2.7 * len(CLASS_KEYS), 2.4 * len(PHASES)), constrained_layout=True,
                             sharey="row")
    for i, phase in enumerate(PHASES):
        for j, c in enumerate(CLASS_KEYS):
            ax, t = axes[i, j], result["tails"][c][phase]
            if t is None:
                ax.axis("off")
                continue
            for pop, short in POPS.items():
                h = t["hill"][short]
                ax.fill_between(h["xmin"], *h["band"], color=COLORS[pop], alpha=0.2, lw=0)
                ax.plot(h["xmin"], h["value"], color=COLORS[pop], lw=1.2, label=f"{short}, KS {t['xmin']['own'][short]:.0f} s")
            ax.axvline(t["xmin"]["common"], color="#888888", lw=0.8, ls=":")
            ax.set(xscale="log")
            ax.set_title(f"{PHASES[phase]}, {c}", fontsize=8)
            ax.legend(frameon=False, fontsize=6)
            ax.grid(color="#eeeeee", lw=0.5)
            ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$x_\mathrm{min}$ (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mu$", fontsize=8)
    save(fig, path)


def draw_mu(result, path):
    """mu of S, A, G above the common threshold per phase (panels) and class (rows), with bands, and the gaps."""
    fig, axes = plt.subplots(1, len(PHASES), figsize=(3.2 * len(PHASES), 3), constrained_layout=True, sharey=True)
    ys = np.arange(len(CLASS_KEYS))
    for ax, phase in zip(axes, PHASES):
        for off, (pop, short) in zip((-0.2, 0, 0.2), POPS.items()):
            for y, c in zip(ys, CLASS_KEYS):
                t = result["tails"][c][phase]
                if t:
                    m = t["mu"][short]
                    ax.errorbar(m["value"], y + off, xerr=[[m["value"] - m["band"][0]], [m["band"][1] - m["value"]]], fmt="o", ms=4,
                                color=COLORS[pop], elinewidth=1.2, capsize=2, label=short if y == 0 else None)
        ax.set_yticks(ys, CLASS_KEYS)
        ax.set_title(PHASES[phase], fontsize=8)
        ax.set_xlabel(r"$\mu$", fontsize=8)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, path)


def draw(result):
    """The figures of one label set; those of his labels carry the suffix _vilpellet."""
    folder, end = longlags.RESULTS, ("" if LABELS_NAME == "refit" else f"_{LABELS_NAME}") + PLAIN
    draw_forests(result, kept_stats(list(COMPOSITION)), folder / f"phases_composition{end}.pdf")
    draw_forests(result, kept_stats(list(PERFORMANCE)), folder / f"phases_performance{end}.pdf")
    draw_tails(result, folder / f"phases_tails{end}.pdf")
    draw_mu(result, folder / f"phases_mu{end}.pdf")
    draw_hill(result, folder / f"phases_hill{end}.pdf")


def compare(sets=("refit", "refit_common", "vilpellet")):
    """The main gaps of the label sets side by side (those whose JSON exists): per statistic a panel, per class two
    rows (G - A, A - S); the refit filled, the refit on the flights his labels cover as squares, his labels hollow."""
    results = {k: json.loads((longlags.RESULTS / f"phases_{k}{PLAIN}.json").read_text()) for k in sets if (longlags.RESULTS / f"phases_{k}{PLAIN}.json").exists()}
    marks = {"refit": ("o", True), "refit_common": ("s", True), "vilpellet": ("o", False)}
    stats = kept_stats(list(COMPOSITION)) + kept_stats(list(PERFORMANCE))
    per_row = 3
    rows = -(-len(stats) // per_row)
    fig, axes = plt.subplots(rows, per_row, figsize=(3.4 * per_row, 3 * rows), squeeze=False, sharey=True, constrained_layout=True)
    ys = np.arange(len(CLASS_KEYS))
    n = len(results)
    offsets = {(g, k): (i * n + j - (2 * n - 1) / 2) * 0.1 for i, g in enumerate(MAIN_GAPS) for j, k in enumerate(results)}
    for ax, stat in zip(axes.flat, stats):
        for (gap, k), off in offsets.items():
            m, filled = marks[k]
            for y, c in zip(ys, CLASS_KEYS):
                g = results[k]["phases"][c]["gaps"][gap][stat]
                ax.errorbar(g["value"], y + off, xerr=[[g["value"] - g["band"][0]], [g["band"][1] - g["value"]]], fmt=m, ms=4,
                            color=GAP_COLORS[gap], mfc=GAP_COLORS[gap] if filled else "white", elinewidth=1.1, capsize=2,
                            label=f"{gap}, {k}" if y == 0 else None)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(ys, CLASS_KEYS)
        ax.set_title(stat, fontsize=8)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(frameon=False, fontsize=6)
    for ax in axes.flat[len(stats):]:
        ax.axis("off")
    save(fig, longlags.RESULTS / f"phases_compare_labels{PLAIN}.pdf")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
        sys.exit()
    if "--draw" in sys.argv:
        draw(json.loads((longlags.RESULTS / f"{NAME}.json").read_text()))
        sys.exit()
    p, code, counts, classes = context()
    kept = phase_terms.kept_flights()
    runs = assign(phase_terms.table(LABELS_NAME, kept), p, classes)
    if COMMON:
        runs = runs[runs["flight"].isin(phase_terms.read_runs("vilpellet", kept)["flight"])]
    result = {"labels": LABELS_NAME, "rule": phase_terms.RULE, "draws": DRAWS, "classes": CLASS_KEYS, "statistics": list(MEASURED), "cycle": CYCLE,
              "phases": phase_statistics(runs, code, counts), "peaks": peaks()}
    result["tails"] = duration_statistics(runs, code, counts)
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    (longlags.RESULTS / f"{NAME}.json").write_text(json.dumps(result, indent=1))
