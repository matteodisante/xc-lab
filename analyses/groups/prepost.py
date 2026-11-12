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

With --open (or --declared-open) the open flights are analysed by the method of the comparisons plan (point 4):
windows of OPEN_WINDOW_S s measured at the lags of longlags.LONG_LAGS up to SUPPORT x the window; the MSD
and the H_loc figures of the four windows and of the control pairs; the seven phase statistics by the
estimator that averages first (unit = window); post - pre and the controls on the events that have both
windows, and, on the events that have all four, D_in = start - pre, D_out = end - post and the drift-free

    g   = (D_in + D_out) / 2    the group effect, free of a drift linear in time
    d w = (D_in - D_out) / 2    the drift over one window

with the bands of twoway.py (the same draws of pilots, blocks and cells for every quantity). Windows of at least
performance.XC_S / SUPPORT s also give the cross-country speed (performance.XC_S), from the windows' own positions.

Mock events, the placebo of g: a profile of H along the flight (low after takeoff and before landing) makes g
nonzero with no group effect. For every real event with the four windows, up to MOCKS solo flights (population
"solo": flights never in a group) are drawn at random, where the four windows fit at the same flight-clock times
inside one solo stretch, under the calipers of matched.py (month, local hour of the middle of the event, same wing
class or level; the caliper matched.py chose, or --caliper NAME), and, in a second variant, also the same takeoff site
(route.SITE_LINK). They are measured and compared like the real windows. A mock event takes the resampling cell of its
real event and the mocks of one real event weigh one event. The result key "mock" gives D_in, D_out, g and d w of the
mock events, of the real events that have one, and their difference (real - mock), with the counts.

    uv run python prepost.py --open [--window SECONDS] [--sample N] [--recompute]    # after tamsd.py and performance.py --open
    uv run python prepost.py --open --redraw [--window SECONDS]    # the forest plot again from the JSON
"""

import json
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget
import curves
import decompose
import isolated
import longlags
import matched
import performance
import raw
import route
import tamsd
import twoway
import zones
from blocks import with_takeoffs
from common.msd import tamsd_sums
from tamsd import (CACHE, COLORS, FIXES, GRID_S, LABELS, LAGS, N_BOOT, OPEN, OUT, SEED, read_flights, read_pieces, save,
                   save_pieces, stretches)

WINDOW_S = 1260  # s, the shortest window: the 1250 s of a piece of the TA-MSD on the 10-s grid; --window SECONDS
LONG_S = 3000  # s, the default of the command line: lags up to 1000 s keep 200 origins
COMPARISONS = {  # name: (P, Q); the difference is P - Q
    "entry": ("start", "pre"), "exit": ("end", "post"), "after minus before": ("post", "pre"),
    "control, group flights alone": ("ctrlA2", "ctrlA1"), "control, solo flights": ("ctrlS2", "ctrlS1")}
STAGES = ("pre", "start", "end", "post")
OPEN_WINDOW_S = 2400  # s, the window of the open flights: lags up to 1840 s keep 56 origins
QUOTED_LAG = 1000  # s, the quoted H_loc
FOREST_TICKS = 5  # at most, on each axis of the forest plot
OPEN_COMPARISONS = {"post - pre": ("post", "pre"), "ctrlA2 - ctrlA1": ("ctrlA2", "ctrlA1"),
                    "ctrlS2 - ctrlS1": ("ctrlS2", "ctrlS1")}  # D_in and D_out: on the events of g (effects)
FOUR = ("g", "d w")  # the drift-free effects, on the events that have the four STAGES
CONTROLS = {"A": ("ctrlA1", "ctrlA2"), "S": ("ctrlS1", "ctrlS2")}
LINES = {  # window: (colour, line style)
    "pre": (COLORS["group_solo"], "-"), "start": (COLORS["group"], "-"), "end": ("#8c2f0f", "--"), "post": ("#0f6e4c", "--"),
    "ctrlA1": (COLORS["group_solo"], "-"), "ctrlA2": ("#0f6e4c", "--"), "ctrlS1": (COLORS["solo"], "-"), "ctrlS2": ("#16427a", "--")}
FOREST = {"D_in": "#eb6834", "D_out": "#8c2f0f", "post - pre": "#555555", "g": "#c24a1d", "d w": "#9a9893",
          "ctrlA2 - ctrlA1": "#1baf7a", "ctrlS2 - ctrlS1": "#2a78d6"}
FOREST_LABELS = {"D_in": "D_in = start - pre", "D_out": "D_out = end - post", "post - pre": "post - pre",
                 "g": "g = (D_in + D_out) / 2", "d w": "d w = (D_in - D_out) / 2",
                 "ctrlA2 - ctrlA1": "control, group flights alone", "ctrlS2 - ctrlS1": "control, solo flights"}
STATS = {**decompose.BUDGET, **{k: v for k, v in decompose.PERFORMANCE.items() if k != decompose.XC}}
XC = decompose.XC  # the cross-country speed, measured on the windows themselves (window_job)
MOCKS = 5  # mock events drawn per real event
MOCK_POOL = 2 * MOCKS  # solo flights measured per real event: the first MOCKS that have the four windows are kept
MOCK_VARIANTS = {"any site": False, "same site": True}  # name: whether the solo flight also took off at the site (route.SITE_LINK)
MOCK_EFFECTS = ("g", "d w", "D_in", "D_out")


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


def window_lags(width):
    """The lags of longlags.LONG_LAGS a window of `width` s measures without changing population: up to SUPPORT x width."""
    return longlags.LONG_LAGS[longlags.LONG_LAGS <= tamsd.SUPPORT * width]


def events_with(population, event, labels):
    """Mask over the rows: those of the events that have a row in every population of `labels`."""
    e = pd.factorize(event)[0]
    have = np.all([np.bincount(e[population == label], minlength=e.max() + 1) > 0 for label in labels], axis=0)
    return have[e]


def drift_free(d_in, d_out):
    """The group effect g = (D_in + D_out) / 2 and the drift over one window d w = (D_in - D_out) / 2."""
    return (d_in + d_out) / 2, (d_in - d_out) / 2


def shares(event, group, keep):
    """Per row: 1 / (the events of its `group` among the rows of `keep`), so that the events of a group weigh one
    together; 0 outside `keep`."""
    k = pd.factorize(group)[0]
    first = pd.Series(k[keep]).groupby(pd.factorize(event)[0][keep]).first()
    n = np.bincount(first.to_numpy(), minlength=k.max() + 1)
    out = np.zeros(len(event))
    out[keep] = 1.0 / n[k[keep]]
    return out


def curves_on(units, counts, labels, lags, group=None, only=None):
    """Per population of `labels`, on the events that have them all (unit = window, weight 1 or the count of its
    block): the ensemble MSD and, as `values`, H_loc at every lag, the fitted H and H_loc(QUOTED_LAG) with its
    window cut at QUOTED_LAG as on dataset `1000` of the other points, in rows (point estimate, then one row per
    draw of `counts`); windows, flights; and the events. `only`: mask over the units that limits the events;
    `group`: the group of each unit (the real event of a mock one), whose events weigh one together."""
    keep = events_with(units.population, units.stratum, labels)
    if only is not None:
        keep = keep & only
    share = np.ones(len(keep)) if group is None else shares(units.stratum, group, keep)
    out = {}
    for label in labels:
        m = keep & (units.population == label)
        w = np.vstack([np.ones(m.sum()), counts[:, units.block[m]]]).astype(float) * share[m]
        with np.errstate(invalid="ignore"):  # a draw may keep none of the windows
            msd = (w @ units.values[m]) / w.sum(axis=1)[:, None]
        h, f = zones.curve(msd, lags)
        cut = lags <= QUOTED_LAG
        quoted = np.array([tamsd.local_h(lags[cut], row[cut])[-1] for row in msd])
        out[label] = {"msd": msd, "values": np.column_stack([h, f, quoted]), "windows": int(m.sum()),
                      "flights": int(np.unique(units.flight[m]).size)}
    return out, int(np.unique(units.stratum[keep]).size)


def phase_on(meta, r, q, counts, labels, stats, group=None, only=None):
    """The same for the statistics of `stats` by the estimator that averages first: windows `meta` (flight,
    population, episode, block) with their rates r and q (budget.unit_rates); `values` has one column per statistic.
    `only`, `group`: as in curves_on."""
    keep = events_with(meta["population"].to_numpy(), meta["episode"].to_numpy(), labels)
    if only is not None:
        keep = keep & only
    share = None if group is None else shares(meta["episode"].to_numpy(), group, keep)
    out = {}
    for label in labels:
        m = keep & (meta["population"] == label).to_numpy()
        rm, qm = r[m], q[m]
        if share is not None:
            rm, qm = rm.mul(share[m], axis=0), qm.mul(share[m], axis=0)
        value, draws = budget.mean_ratios(rm, qm, meta["block"].to_numpy()[m], counts, stats)
        out[label] = {"values": np.vstack([value.to_numpy(), draws]), "windows": int(m.sum()),
                      "flights": int(meta["flight"][m].nunique())}
    return out, int(meta["episode"][keep].nunique())


def window_rates(tables, wins, block, stats_by_source, xc=None):
    """Per window of `wins` (flight, a, b): its phase time T_u from the runs that lie in it by their middle, and
    r_u, q_u of every statistic (budget.unit_rates), keeping the windows with T_u > 0; returns (meta, r, q, stats).
    `xc` (flight, population, episode, xc_m, xc_origins: window_job) adds the cross-country speed of each window."""
    flight, a, b = (wins[c].to_numpy() for c in ("flight", "a", "b"))
    time = pd.Series(isolated.window_sums(tables["b"], ["s"], flight, a, b)[:, 0])
    rates, stats = [], {}
    for src, st in stats_by_source.items():
        cols = sorted({c for num, den, *_ in st.values() for c in (num, den)})
        sums = pd.DataFrame(isolated.window_sums(tables[src], cols, flight, a, b), columns=cols)
        rates.append(budget.unit_rates(sums, np.arange(len(wins)), time, st))
        stats |= st
    stats = {n: stats[n] for n in raw.STATISTICS}
    r = pd.concat([x[0] for x in rates], axis=1)[list(stats)]
    q = pd.concat([x[1] for x in rates], axis=1)[list(stats)]
    if xc is not None:
        terms = wins[["flight", "population", "episode"]].merge(xc, how="left", on=["flight", "population", "episode"])
        rx, qx = budget.unit_rates(terms[["xc_m", "xc_origins"]].fillna(0.0), np.arange(len(wins)), time, {XC: performance.STATS[XC]})
        stats, r, q = {**stats, XC: performance.STATS[XC]}, pd.concat([r, rx], axis=1), pd.concat([q, qx], axis=1)
    ok = (time > 0).to_numpy()
    meta = wins.assign(block=block.reindex(wins["flight"]).to_numpy(int))[ok].reset_index(drop=True)
    return meta, r[ok].reset_index(drop=True), q[ok].reset_index(drop=True), stats


def effects(on, comparisons=OPEN_COMPARISONS, stages=STAGES):
    """The differences P - Q of `comparisons` (each on the events that have both windows), and D_in, D_out, g and
    d w on the events that have the `stages`, from `on(labels)` = (per population its `values`, events). Returns
    {name: (values, events)}: rows draws, columns what `on` measures."""
    out = {}
    for name, (P, Q) in comparisons.items():
        got, events = on((P, Q))
        out[name] = (got[P]["values"] - got[Q]["values"], events, got[P]["windows"], got[P]["flights"])
    got, events = on(stages)
    v = {k: x["values"] for k, x in got.items()}
    d_in, d_out = v["start"] - v["pre"], v["end"] - v["post"]
    g, dw = drift_free(d_in, d_out)
    shape = got["pre"]["windows"], got["pre"]["flights"]
    out.update({"D_in": (d_in, events, *shape), "D_out": (d_out, events, *shape),
                "g": (g, events, *shape), "d w": (dw, events, *shape)})
    return out


def describe_msd(x, lags):
    """A difference of `curves_on` (rows draws): Delta H_loc at QUOTED_LAG (window cut there), Delta H and Delta
    H_loc at every lag."""
    values = x[0]
    return {f"H_loc({QUOTED_LAG} s)": twoway.quote(values[:, -1]), "Delta H": twoway.quote(values[:, -2]),
            "dH_loc": twoway.quote(values[:, :-2])}


def curve_of(entry, lags):
    """A series for curves.series from one population of `curves_on`: MSD and H_loc with 5-95% bands, and H."""
    v, m = entry["values"], entry["msd"]
    p = lambda a: twoway.band(a[0], a[1:])
    return {"lags": lags.tolist(), "msd": m[0].tolist(), "msd_band": p(m), "H_loc": v[0, :-2].tolist(),
            "H_loc_band": p(v[:, :-2]), "H": float(v[0, -2]), "H_band": p(v[:, -2]),
            f"H_loc({QUOTED_LAG} s)": float(v[0, -1]), f"H_loc({QUOTED_LAG} s) band": p(v[:, -1]),
            "windows": entry["windows"], "flights": entry["flights"]}


def draw_curves(curves_by_label, stem, title):
    """The MSD and the H_loc figure of the windows of `curves_by_label` (label: curve_of), fitted H in the legend."""
    drawn = []
    for label, c in curves_by_label.items():
        lo, hi = c["H_band"]
        drawn.append(curves.series(c, f"{label} ({c['windows']:,} windows): H = {c['H']:.3f} [{lo:.3f}, {hi:.3f}]",
                                   *LINES[label]))
    curves.figures(drawn, stem, title)


def describe(m, p, lags, stats):
    """One entry of the result from an effect on the MSD and one on the phase statistics (values, events, windows,
    flights; None when no event has phase time): the counts, Delta H_loc(QUOTED_LAG), Delta H and Delta H_loc at every
    lag, and the statistics."""
    p, quoted = (p, True) if p is not None else ((None, 0, 0, 0), False)
    return {"events (MSD)": m[1], "windows (MSD)": m[2], "flights (MSD)": m[3],
            "events (phase)": p[1], "windows (phase)": p[2], "flights (phase)": p[3],
            **describe_msd(m, lags), **({s: twoway.quote(p[0][:, j]) for j, s in enumerate(stats)} if quoted else {})}


def open_analysis(units, meta, r, q, stats, counts, lags):
    """Everything of the open mode as a dict: window curves, differences of the MSD statistics and of the phase
    statistics, with bands from `counts`."""
    msd_on = lambda labels: curves_on(units, counts, labels, lags)
    phase = lambda labels: phase_on(meta, r, q, counts, labels, stats)
    m, p = effects(msd_on), effects(phase)
    result = {"curves": {}, "differences": {}, "phase statistics": {}}
    for key, labels in (("stages", STAGES), *((f"controls {k}", v) for k, v in CONTROLS.items())):
        got, events = msd_on(labels)
        result["curves"][key] = {"events": events, **{l: curve_of(c, lags) for l, c in got.items()}}
        got, events = phase(labels)
        result["phase statistics"][key] = {"events": events, **{l: {
            "windows": c["windows"], "flights": c["flights"], **{s: twoway.quote(c["values"][:, j])
                                                                   for j, s in enumerate(stats)}} for l, c in got.items()}}
    for name in m:
        result["differences"][name] = describe(m[name], p[name], lags, stats)
    return result


def window_job(flight, fixes, spans, lags, shortest):
    """tamsd.pieces of the windows `spans`, with one more column in the sums and the origins: the straight-line
    distance covered in performance.XC_S from every origin, summed, and the number of origins."""
    k = performance.XC_S // GRID_S
    found = tamsd.pieces(fixes, spans, lags, shortest)
    far = [np.hypot(*(pos[k:] - pos[:-k]).T) if len(pos) > k else np.zeros(0) for _, _, pos in tamsd.grids(fixes, spans, shortest)]
    return flight, [(*f[:6], np.r_[f[6], d.sum()], np.r_[f[7], len(d)]) for f, d in zip(found, far)]


def xc_table(p):
    """The windows of the pieces `p` of window_job with their cross-country terms."""
    return pd.DataFrame({"flight": p["flight"], "population": p["population"], "episode": p["episode"],
                         "xc_m": p["sums"][:, -1], "xc_origins": p["origins"][:, -1]})


# ---- mock events: the windows of the real events at the same flight-clock times in flights that never fly in a group

def event_times(wins):
    """The events of `wins` that have the four STAGES: episode, flight, t_in and t_out (start and end of the group
    stretch, flight clock)."""
    w = wins[wins["population"].isin(STAGES)]
    n = w.groupby("episode")["population"].nunique()
    w = w[w["episode"].isin(n.index[n == len(STAGES)])]
    start, end = (w[w["population"] == k].set_index("episode") for k in ("start", "end"))
    return pd.DataFrame({"flight": start["flight"], "t_in": start["a"], "t_out": end["b"]}).rename_axis("episode").reset_index()


def mock_info(flights, caliper):
    """Per flight (index): utc0, the wing class or level of matched.CALIPERS[caliper] (NaN where matched.select does
    not match) and the site of its takeoff (route.site_frames)."""
    cls = isolated.wing_class(flights)
    level = isolated.levels(flights)
    kind = cls.where(cls.index.isin(level.index)) if matched.CALIPERS[caliper][2] == "class" else level.reindex(cls.index)
    f = flights.set_index("flight")
    site = route.site_frames(f["lat0"].to_numpy(float), f["lon0"].to_numpy(float))[0]
    return pd.DataFrame({"utc0": f["utc0"], "kind": kind.reindex(f.index), "site": site}, index=f.index)


def mock_draws(events, solo, info, width, caliper, rng, same_site=False, pool=MOCK_POOL, first_id=0):
    """For each event of `events` (event_times), up to `pool` solo flights in random order (`rank`), as rows real,
    real_flight, flight, t_in, t_out, rank and an `id` from `first_id`. A solo flight is eligible when its four
    windows at the flight-clock times of the event, [t_in - w, t_out + w], lie in one of its solo stretches `solo`
    (flight, a, b), its takeoff is another flight's and, under matched.CALIPERS[caliper], the month and the local
    hour of the middle of the event are within the caliper and the wing class (or level) is the same; with
    `same_site`, also the takeoff site (`info`: mock_info)."""
    month, hour, _ = matched.CALIPERS[caliper]
    sf, sa, sb = (solo[c].to_numpy() for c in ("flight", "a", "b"))
    rows = []
    for e in events.itertuples():
        kind = info.at[e.flight, "kind"]
        if pd.isna(kind):
            continue
        f = np.unique(sf[(sa <= e.t_in - width) & (sb >= e.t_out + width)])
        f = f[f != e.flight]
        mine = info.loc[f]
        ok = (mine["kind"] == kind).to_numpy()
        if same_site:
            ok = ok & (mine["site"] == info.at[e.flight, "site"]).to_numpy()
        f, mine = f[ok], mine[ok]
        if len(f):
            mid = (e.t_in + e.t_out) / 2
            real = np.full(len(f), info.at[e.flight, "utc0"] + mid)
            near = (matched.month_gap(real, mine["utc0"].to_numpy() + mid) <= month) \
                & (matched.hour_gap(real, mine["utc0"].to_numpy() + mid) <= hour + 1e-9)
            rows += [(e.episode, e.flight, x, e.t_in, e.t_out, k) for k, x in enumerate(rng.permutation(f[near])[:pool])]
    out = pd.DataFrame(rows, columns=["real", "real_flight", "flight", "t_in", "t_out", "rank"])
    return out.assign(id=first_id + np.arange(len(out)))


def mock_windows(draws, width):
    """The windows (flight, population, episode = id, a, b) of the mock events of `draws`, as windows() places
    those of a real event."""
    w = float(width)
    parts = {"pre": (draws["t_in"] - w, draws["t_in"]), "start": (draws["t_in"], draws["t_in"] + w),
             "end": (draws["t_out"] - w, draws["t_out"]), "post": (draws["t_out"], draws["t_out"] + w)}
    return pd.concat([pd.DataFrame({"flight": draws["flight"], "population": k, "episode": draws["id"], "a": a, "b": b})
                      for k, (a, b) in parts.items()], ignore_index=True)


def valid_mocks(draws, p, keep=MOCKS):
    """The mock events of `draws` that have the four windows among the measured pieces `p`, the first `keep` of each
    real event (the rows of `draws` are in random order within it)."""
    got = pd.DataFrame({"id": p["episode"], "population": p["population"]}).drop_duplicates().groupby("id")["population"].nunique()
    return draws[draws["id"].isin(got.index[got == len(STAGES)])].groupby("real").head(keep)


def mock_units(p, mocks, block):
    """MSD units of the windows of `mocks` (valid_mocks) in the pieces `p`, with the resampling cell of the real
    event of each, and the real event of each unit. `block`: the cell of each flight."""
    q = {k: v[np.isin(p["episode"], mocks["id"])] for k, v in p.items()}
    units = decompose.msd_units(q, "episode", None, block)
    ids = pd.factorize(q["episode"])[1][units.stratum]  # the event of each stratum, as decompose.strata codes them
    by_id = mocks.set_index("id")
    units.block = block.reindex(by_id["real_flight"].reindex(ids)).to_numpy()
    return units, by_id["real"].reindex(ids).to_numpy()


def mock_rates(tables, wins, mocks, block, defs, xc=None):
    """window_rates of the mock windows `wins`, with the cell and the real event of each."""
    meta, r, q, stats = window_rates(tables, wins, block, defs, xc)
    by_id = mocks.set_index("id")
    return (meta.assign(block=block.reindex(by_id["real_flight"].reindex(meta["episode"])).to_numpy(),
                        real=by_id["real"].reindex(meta["episode"]).to_numpy()), r, q, stats)


def with_mock(population, event, group):
    """The groups (real events) that have a mock event with every STAGE."""
    return np.unique(np.asarray(group)[events_with(population, event, STAGES)])


def n_events(population, event):
    """How many events have a window in every STAGE."""
    return int(np.unique(event[events_with(population, event, STAGES)]).size)


def mock_results(real, mock, stats, counts, lags):
    """The effects of the mock events and of the real events that have one (the same draws of `counts`): per effect
    of MOCK_EFFECTS the entry of `differences` for the mock events, for those real events and their difference
    (real - mock); and the counts. `real`: units, id (the event of each unit), meta, r, q; `mock`: the same with
    `group` (units) and meta["real"] (windows): the real event of each. Without a mock event that has phase time
    the phase statistics are left out."""
    ids = (with_mock(mock.units.population, mock.units.stratum, mock.group),
           with_mock(mock.meta["population"].to_numpy(), mock.meta["episode"].to_numpy(), mock.meta["real"]) if len(mock.meta) else [])
    on = {"mock": (lambda labels: curves_on(mock.units, counts, labels, lags, group=mock.group),
                   lambda labels: phase_on(mock.meta, mock.r, mock.q, counts, labels, stats, group=mock.meta["real"].to_numpy())),
          "real": (lambda labels: curves_on(real.units, counts, labels, lags, only=np.isin(real.id, ids[0])),
                   lambda labels: phase_on(real.meta, real.r, real.q, counts, labels, stats,
                                           only=real.meta["episode"].isin(ids[1]).to_numpy()))}
    got = {k: (effects(f[0], {}, STAGES), effects(f[1], {}, STAGES) if len(ids[1]) else None) for k, f in on.items()}
    entry = lambda e, n: describe(e[0][n], e[1][n] if e[1] else None, lags, stats)
    net = {n: tuple(None if got["mock"][i] is None else (got["real"][i][n][0] - got["mock"][i][n][0], *got["real"][i][n][1:])
                    for i in (0, 1)) for n in MOCK_EFFECTS}
    msd = got["mock"][0]["g"]
    phase = lambda k: got[k][1]["g"] if got[k][1] else (None, 0, 0, 0)
    return {"counts": {"real events with the four windows (MSD)": n_events(real.units.population, real.units.stratum),
                       "real events with a mock event (MSD)": got["real"][0]["g"][1], "mock events (MSD)": msd[1],
                       "mock windows (MSD)": msd[2], "solo flights (MSD)": msd[3],
                       "real events with the four windows (phase)":
                           n_events(real.meta["population"].to_numpy(), real.meta["episode"].to_numpy()) if len(real.meta) else 0,
                       "real events with a mock event (phase)": phase("real")[1], "mock events (phase)": phase("mock")[1],
                       "solo flights (phase)": phase("mock")[3]},
            "mock": {n: entry(got["mock"], n) for n in MOCK_EFFECTS},
            "real events with a mock": {n: entry(got["real"], n) for n in MOCK_EFFECTS},
            "real minus mock": {n: describe(*net[n], lags, stats) for n in MOCK_EFFECTS}}


def forest(result, path, width):
    """The forest plot of Delta H_loc(1000 s) and the seven Delta for D_in, D_out, post - pre, g, d w and the controls."""
    diffs = result["differences"]
    decompose.forest(diffs, path, (f"H_loc({QUOTED_LAG} s)", *raw.STATISTICS), list(FOREST), FOREST_LABELS, FOREST,
                     f"Open paragliders, windows of {width} s: before, during and after a group", per_row=4, nticks=FOREST_TICKS)
    plt.close("all")


def draw_delta(result, lags, path, title):
    """H_loc of the four windows beside D_in, D_out, g (with its band), d w and the two controls, at every lag."""
    stages = result["curves"]["stages"]
    left = [{**stages[l], "label": l, "color": LINES[l][0], "ls": LINES[l][1]} for l in STAGES]
    right = [{"lags": lags[:len(result["differences"][k]["dH_loc"]["value"])].tolist(), **result["differences"][k]["dH_loc"],
              "label": FOREST_LABELS[k], "color": FOREST[k], "ls": "-" if k in ("D_in", "D_out", "g") else "--",
              "lw": 1.4 if k == "g" else 0.9} for k in ("D_in", "D_out", "g", "d w", "ctrlA2 - ctrlA1", "ctrlS2 - ctrlS1")]
    for r in right:
        if r["label"] != FOREST_LABELS["g"]:
            r["band"] = None
    curves.delta_figure(left, right, path, title)


def mock_caliper():
    """The caliper of the mock events: --caliper NAME, or the one matched.py chose (matched_counts.json); None when
    neither exists, and the mock events are left out."""
    if "--caliper" in sys.argv:
        return sys.argv[sys.argv.index("--caliper") + 1]
    path = OUT / "matched_counts.json"
    return json.loads(path.read_text())["chosen caliper"]["name"] if path.exists() else None


def mock_draws_of(wins, spans, flights, width, caliper):
    """{variant: mock_draws} for the real events of `wins`, from the solo stretches of `spans`; one generator, so
    the draws do not depend on the order of the variants' use."""
    events, info, rng = event_times(wins), mock_info(flights, caliper), np.random.default_rng([SEED, 1])
    solo, out, first = spans[spans["population"] == "solo"], {}, 2 * len(spans)  # the episodes of `windows` are below
    for variant, same in MOCK_VARIANTS.items():
        out[variant] = mock_draws(events, solo, info, width, caliper, rng, same, first_id=first)
        first += len(out[variant])
    return out


def run_open(flights, membership, summary, width, sample=None, recompute=False):
    """The open mode: windows, pieces (cached), analysis, JSON and figures in OUT."""
    tag = f"_sample{sample}" if sample else ""
    lags = window_lags(width)
    if sample:  # the first `sample` group flights and as many solo ones
        grouped = summary.set_index("flight")["n_entries"].reindex(flights["flight"]).to_numpy() > 0
        keep = np.r_[flights["flight"][grouped].to_numpy()[:sample], flights["flight"][~grouped].to_numpy()[:sample]]
        flights = flights[flights["flight"].isin(keep)]
    spans = stretches(flights, membership, summary)
    wins = windows(spans, np.random.default_rng(SEED), width)
    made = wins["population"].value_counts().to_dict()
    print(made, flush=True)
    cache_file = CACHE / f"prepost_pieces_{width}_lag{lags.max()}{tag}.npz"
    if recompute or not cache_file.exists():
        found = tamsd.measure("para", flights, wins, lags, width - tamsd.GRID_S)  # a window holds width / 10 grid points
        np.savez(cache_file, **{f"para/{k}": v for k, v in found.items()})
    p = read_pieces(cache_file)["para"]
    info = decompose.flight_table(flights, summary)[0]
    block, of_pilot, of_block = twoway.cells(flights, summary)  # the cell of each flight, as a block code
    counts = twoway.counts(of_pilot, of_block, np.random.default_rng(SEED))
    units = decompose.msd_units(p, "episode", info, block)
    measured = pd.DataFrame({k: p[k] for k in ("flight", "population", "episode")}).drop_duplicates()
    wins = wins.merge(measured, on=["flight", "population", "episode"])
    tables = isolated.run_terms(flights)
    defs = raw.stat_defs()
    caliper = mock_caliper()
    draws = mock_draws_of(wins, spans, flights, width, caliper) if caliper else {}
    xc_on = width >= performance.XC_S / tamsd.SUPPORT  # the 30-min displacement needs a piece of XC_S / SUPPORT s
    if draws or xc_on:  # one more pass over the fixes: the mock windows and the cross-country terms of every window
        every = pd.concat([wins, *[mock_windows(d, width) for d in draws.values()]], ignore_index=True)
        name = f"prepost_windows_{width}_lag{lags.max()}_{caliper}_{MOCKS}x{MOCK_POOL}{tag}.npz" if caliper else \
            f"prepost_windows_{width}_lag{lags.max()}{tag}.npz"
        window_file = CACHE / name
        if recompute or not window_file.exists():
            found = tamsd.measure("para", flights, every, lags, width - tamsd.GRID_S, job=window_job)
            np.savez(window_file, **{f"para/{k}": v for k, v in found.items()})
        pw = read_pieces(window_file)["para"]
        xc = xc_table(pw) if xc_on else None
    else:
        xc = None
    meta, r, q, stats = window_rates(tables, wins, block, defs, xc)
    result = open_analysis(units, meta, r, q, stats, counts, lags)
    if draws:
        msd_pieces = {**pw, "sums": pw["sums"][:, :-1], "origins": pw["origins"][:, :-1]}
        real = SimpleNamespace(units=units, id=pd.factorize(p["episode"])[1][units.stratum], meta=meta, r=r, q=q)
        result["mock"] = {"caliper": caliper, "mocks per event": MOCKS, "pool per event": MOCK_POOL, "variants": {}}
        for variant, d in draws.items():
            kept = valid_mocks(d, msd_pieces)
            if kept.empty:
                continue
            m_units, m_group = mock_units(msd_pieces, kept, block)
            m_meta, m_r, m_q, _ = mock_rates(tables, mock_windows(kept, width), kept, block, defs, xc)
            mock = SimpleNamespace(units=m_units, group=m_group, meta=m_meta, r=m_r, q=m_q)
            result["mock"]["variants"][variant] = mock_results(real, mock, stats, counts, lags)
    result |= {"window_s": width, "lags": lags.tolist(), "n_boot": N_BOOT, "windows made": made,
               "windows measured": wins["population"].value_counts().to_dict(),
               "windows with phase time": meta["population"].value_counts().to_dict()}
    (OUT / f"prepost_{width}{tag}.json").write_text(json.dumps(result, indent=1))
    for key, labels in (("stages", STAGES), *((f"controls {k}", v) for k, v in CONTROLS.items())):
        draw_curves({l: result["curves"][key][l] for l in labels}, OUT / f"prepost_{width}{tag}_{key.replace(' ', '_')}",
                    f"{key}: {result['curves'][key]['events']:,} events, windows of {width} s".replace(",", " "))
    forest(result, OUT / f"prepost_{width}{tag}_forest.pdf", width)
    draw_delta(result, lags, OUT / f"prepost_{width}{tag}_delta", f"{result['curves']['stages']['events']:,} events, windows of {width} s")
    return result


if __name__ == "__main__":
    if "--redraw" in sys.argv:  # the forest plot from the JSON, nothing recomputed
        w = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else OPEN_WINDOW_S
        tamsd.folders()
        tag = f"_sample{sys.argv[sys.argv.index('--sample') + 1]}" if "--sample" in sys.argv else ""
        forest(json.loads((OUT / f"prepost_{w}{tag}.json").read_text()), OUT / f"prepost_{w}{tag}_forest.pdf", w)
        sys.exit()
    flights = with_takeoffs(read_flights())
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    width = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else OPEN_WINDOW_S if OPEN else LONG_S
    if OPEN:
        tamsd.folders()
        sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
        out = run_open(flights, membership, summary, width, sample, "--recompute" in sys.argv)
        for variant, v in out.get("mock", {}).get("variants", {}).items():
            print("mock", variant, v["counts"], {k: [round(v[k]["g"]["H_loc(1000 s)"]["value"], 4),
                  np.round(v[k]["g"]["H_loc(1000 s)"]["band"], 4).tolist()] for k in ("mock", "real events with a mock", "real minus mock")})
        for name, d in out["differences"].items():
            print(name, {k: d[k] for k in d if k.startswith(("events", "windows"))},
                  {k: [round(d[k]["value"], 4), np.round(d[k]["band"], 4).tolist()] for k in d if k != "dH_loc" and isinstance(d[k], dict)})
        sys.exit()
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
