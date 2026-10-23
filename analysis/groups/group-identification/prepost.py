"""The same flight before, during and after the group: the group, or the moment pilots choose it?

For every stretch of group time (tamsd.stretches) the windows of WINDOW_S seconds of the flight clock:

    pre    the WINDOW_S before it starts, if a solo stretch of the same flight, at least that long, precedes it
    start  its first WINDOW_S, if it lasts at least that long
    end    its last WINDOW_S, if it lasts at least that long
    post   the WINDOW_S after it ends, if a solo stretch of the same flight, at least that long, follows it

The windows of one stretch are one stratum (an event). If the straighter progress belongs to the
group, H rises from pre to start and falls from end to post; if it belongs to the moment or to the
pilots that join, pre is already high and the changes are those of two consecutive windows without
a change of state. Those controls are consecutive windows (first, second), drawn at random in the
solo time of the group flights (ctrlA) and in the solo flights (ctrlS), one pair per stretch.

Compared over every lag (ensemble MSD of each window on the events that have both windows, then
H_loc and the fitted H) and on the phase statistics of the runs placed in the windows by their
middle, with 5-95% bands from blocks that join episodes and site-days (decompose.block_counts).

    uv run python prepost.py [--run NAME] [--window SECONDS] [--recompute]    # after tamsd.py and performance.py
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget
import decompose
import zones
from blocks import with_takeoffs
from tamsd import CACHE, COLORS, FIXES, LABELS, LAGS, OUT, SEED, read_pieces, save, save_pieces, stretches

WINDOW_S = 1260  # s, the shortest window: the 1250 s of a piece of the TA-MSD on the 10-s grid; --window SECONDS
LONG_S = 3000  # s, the default of the command line: lags up to 1000 s keep 200 origins
COMPARISONS = {  # name: (P, Q); the difference is P - Q
    "entry": ("start", "pre"), "exit": ("end", "post"), "after minus before": ("post", "pre"),
    "control, group flights alone": ("ctrlA2", "ctrlA1"), "control, solo flights": ("ctrlS2", "ctrlS1")}
STAGES = ("pre", "start", "end", "post")
STATS = {**decompose.BUDGET, **{k: v for k, v in decompose.PERFORMANCE.items() if k != decompose.XC}}


def windows(spans, rng, width=WINDOW_S):
    """The windows of `width` seconds of the module docstring and its controls: flight, population (pre, start, end,
    post, ctrlA1, ctrlA2, ctrlS1, ctrlS2), episode (the stratum: the stretch of group time, or the pair
    of control windows) and the interval [a, b) on the flight clock."""
    s = spans.sort_values(["flight", "a"], kind="stable").reset_index(drop=True)
    flight, pop, a, b = (s[c].to_numpy() for c in ("flight", "population", "a", "b"))
    w, rows = float(width), []
    for i in np.flatnonzero(pop == "group"):
        before = i > 0 and flight[i - 1] == flight[i] and pop[i - 1] == "group_solo" and b[i - 1] == a[i] \
            and b[i - 1] - a[i - 1] >= w
        after = i + 1 < len(s) and flight[i + 1] == flight[i] and pop[i + 1] == "group_solo" and a[i + 1] == b[i] \
            and b[i + 1] - a[i + 1] >= w
        if before:
            rows.append((flight[i], "pre", i, a[i] - w, a[i]))
        if b[i] - a[i] >= w:
            rows += [(flight[i], "start", i, a[i], a[i] + w), (flight[i], "end", i, b[i] - w, b[i])]
        if after:
            rows.append((flight[i], "post", i, b[i], b[i] + w))
    offset = len(s)
    for i in np.flatnonzero(np.isin(pop, ("group_solo", "solo")) & (b - a >= 2 * w)):
        c0 = a[i] + rng.integers(int((b[i] - a[i]) // (2 * w))) * 2 * w
        tag = "A" if pop[i] == "group_solo" else "S"
        rows += [(flight[i], f"ctrl{tag}1", offset + i, c0, c0 + w), (flight[i], f"ctrl{tag}2", offset + i, c0 + w, c0 + 2 * w)]
    return pd.DataFrame(rows, columns=["flight", "population", "episode", "a", "b"])


def place(terms, wins):
    """The rows of `terms` (flight, t_mid) in the window of `wins` that holds t_mid, with its population and episode."""
    both = terms.drop(columns=["population"]).merge(wins, on="flight")
    return both[(both["a"] <= both["t_mid"]) & (both["t_mid"] < both["b"])].reset_index(drop=True)


def strata_with_both(units, P, Q):
    """How many strata have units of both populations."""
    s = pd.factorize(units.stratum)[0]
    return int(((np.bincount(s[units.population == P], minlength=s.max() + 1) > 0)
                & (np.bincount(s[units.population == Q], minlength=s.max() + 1) > 0)).sum())


def compare(units, counts, comparisons=COMPARISONS):
    """Per comparison: events, H_loc of P and Q over every lag, fitted H, and the differences P - Q (point
    estimate and 5-95% band), as zones.compare does for strata; the per-draw arrays stay in `draws`."""
    out, draws = {}, {}
    for name, (P, Q) in comparisons.items():
        mp, mq, _, _ = zones.standardized(units, P, Q, counts)
        hp, fp = zones.curve(mp)
        hq, fq = zones.curve(mq)
        draws[name] = (hp - hq, fp - fq)
        entry = {"events": strata_with_both(units, P, Q), "lags": LAGS.tolist()}
        for label, h, f in ((P, hp, fp), (Q, hq, fq)):
            v, b = zones.band(h)
            entry[label] = {"H_loc": v, "H_loc_band": b, "H": f[0], "H_band": np.percentile(f[1:], [5, 95]).tolist()}
        v, b = zones.band(hp - hq)
        entry["dH_loc"], entry["dH_loc_band"] = v, b
        entry["dH"], entry["dH_band"] = float((fp - fq)[0]), np.percentile((fp - fq)[1:], [5, 95]).tolist()
        out[name] = entry
        print(name, entry["events"], "events", round(entry["dH"], 4), flush=True)
    return out, draws


def profile(units, counts, labels=STAGES):
    """H and H_loc(100 s) and (1000 s) of every window of `labels` on the events that have them all:
    value and 5-95% band, and the number of events."""
    s = pd.factorize(units.stratum)[0]
    n_s = s.max() + 1
    masks = [units.population == label for label in labels]
    msd = [[] for _ in labels]
    for d in range(len(counts) + 1):
        w = np.ones(len(s)) if d == 0 else counts[d - 1, units.block].astype(float)
        n = [np.bincount(s[m], weights=w[m], minlength=n_s) for m in masks]
        sums = [np.column_stack([np.bincount(s[m], weights=w[m] * v, minlength=n_s) for v in units.values[m].T]) for m in masks]
        ok = np.all([x > 0 for x in n], axis=0)
        for i in range(len(labels)):
            msd[i].append(sums[i][ok].sum(axis=0) / n[i][ok].sum())
    k = {x: int(np.searchsorted(LAGS, x)) for x in (100, 1000)}
    out = {"events": int(np.all([np.bincount(s[m], minlength=n_s) > 0 for m in masks], axis=0).sum())}
    for label, m in zip(labels, msd):
        h, f = zones.curve(np.array(m))
        out[label] = {"H": zones.band(f[:, None])[0][0], "H_band": np.percentile(f[1:], [5, 95]).tolist(),
                      **{f"H_loc({x})": h[0][i] for x, i in k.items()},
                      **{f"H_loc({x})_band": np.percentile(h[1:, i], [5, 95]).tolist() for x, i in k.items()}}
    return out


def ratio_differences(table, key_stats, info, block, counts, comparisons=COMPARISONS):
    """The phase statistics of `key_stats` for every comparison: P - Q over the events with both windows."""
    units = decompose.ratio_units(table, "episode", info, block, key_stats)
    out = {}
    for name, (P, Q) in comparisons.items():
        values = decompose.effect(units, P, Q, counts)
        out[name] = {stat: decompose.described(values[:, j]) for j, stat in enumerate(key_stats)}
        out[name]["events"] = strata_with_both(units, P, Q)
    return out


def figure_curves(result, path):
    """H_loc(tau) before and after entering and leaving a group, and the differences with the controls."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    lags = np.array(result["entry"]["lags"])
    green, orange, blue = COLORS["group_solo"], COLORS["group"], COLORS["solo"]
    for ax, name, (P, Q), title in ((axes[0], "entry", COMPARISONS["entry"], "entering a group"),
                                    (axes[1], "exit", COMPARISONS["exit"], "leaving a group")):
        for label, color, text in ((Q, green, "alone"), (P, orange, "in the group")):
            e = result[name][label]
            ax.fill_between(lags, *np.array(e["H_loc_band"]), color=color, alpha=0.2, lw=0)
            ax.plot(lags, e["H_loc"], color=color, marker="o", ms=3, lw=1.3,
                    label=f"{text} ({label} window): H = {e['H']:.3f}")
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=f"{title}: {result[name]['events']:,} events".replace(",", " "))
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[0].set_ylabel(r"$H_{\rm loc}(\tau)$")
    ax = axes[2]
    for name, color, ls, label in (("entry", orange, "-", "entering: group minus before"),
                                   ("exit", "#8c2f0f", "-", "leaving: group minus after"),
                                   ("control, group flights alone", green, "--", "control: later minus earlier window, same flights alone"),
                                   ("control, solo flights", blue, "--", "control: later minus earlier window, solo flights")):
        e = result[name]
        ax.fill_between(lags, *np.array(e["dH_loc_band"]), color=color, alpha=0.15, lw=0)
        ax.plot(lags, e["dH_loc"], color=color, ls=ls, marker="o", ms=3, lw=1.3, label=label)
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel=r"$\tau$ (s)", title="differences over every lag")
    ax.set_ylabel(r"$\Delta H_{\rm loc}(\tau)$")
    ax.legend(frameon=False, fontsize=7)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    save(fig, path)


def figure_profile(profile_, path):
    """H, H_loc(100 s) and H_loc(1000 s) of the four windows of the events that have all of them."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    names = {"pre": "before", "start": "start of\nthe group", "end": "end of\nthe group", "post": "after"}
    for ax, key, title in zip(axes, ("H", "H_loc(100)", "H_loc(1000)"), ("fitted H, 10-1000 s", "H_loc(100 s)", "H_loc(1000 s)")):
        v = np.array([profile_[s][key] for s in STAGES])
        band = np.array([profile_[s][f"{key}_band"] if key != "H" else profile_[s]["H_band"] for s in STAGES])
        ax.errorbar(range(4), v, yerr=[v - band[:, 0], band[:, 1] - v], fmt="o-", color=COLORS["group"], capsize=3, lw=1.3)
        ax.set_xticks(range(4), [names[s] for s in STAGES])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"Paragliders: the same flights before, during and after a group ({profile_['events']:,} events with the four windows)"
                 .replace(",", " "), fontsize=9.5)
    save(fig, path)


if __name__ == "__main__":
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    width = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else LONG_S
    cache_file = CACHE / f"prepost_pieces_{width}.npz"
    spans = stretches(flights, membership, summary)
    wins = windows(spans, np.random.default_rng(SEED), width)
    print(wins["population"].value_counts().to_dict(), flush=True)
    if "--recompute" in sys.argv or not cache_file.exists():
        save_pieces(cache_file, flights, wins)
    info, block, n_blocks = decompose.flight_table(flights, summary)
    counts = decompose.block_counts(n_blocks, np.random.default_rng(SEED))
    units = decompose.msd_units(read_pieces(cache_file)["para"], "episode", info, block)
    result, _ = compare(units, counts)
    result["profile"] = profile(units, counts)
    runs = budget.phase_runs(flights, wins)
    terms = place(pd.read_parquet(CACHE / "performance_terms.parquet").dropna(subset=["t_mid"]), wins)
    in_runs = ratio_differences(runs, {k: STATS[k] for k in decompose.BUDGET}, info, block, counts)
    in_terms = ratio_differences(terms, {k: STATS[k] for k in STATS if k not in decompose.BUDGET}, info, block, counts)
    result["phase statistics"] = {
        name: {"events (runs)": in_runs[name]["events"], "events (glides and climbs)": in_terms[name]["events"],
               **{k: v for k, v in in_runs[name].items() if k != "events"},
               **{k: v for k, v in in_terms[name].items() if k != "events"}} for name in COMPARISONS}
    result["window_s"] = width
    (OUT / f"prepost_{width}.json").write_text(json.dumps(result, indent=1))
    figure_curves(result, OUT / f"prepost_{width}_curves.pdf")
    figure_profile(result["profile"], OUT / f"prepost_{width}_profile.pdf")
