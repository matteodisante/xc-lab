"""The time to find a thermal: Kaplan-Meier survival of the searches, with censoring, per population S, A, G and ground class.

Every search of phase_terms.py (rule `context`) is an attempt to find a thermal: a search that is found (transition-
search-climb) ends at its duration d with the event "found"; a search that fails (transition-search-transition) is right-
censored at d, since the pilot glides on and would have found a thermal later. Exits, re-centrings and searches with an
unknown neighbour are not attempts. Durations are those of the runs (phase_gaps.py), in whole seconds, not trimmed.

Per population and class, with the weights of the shared two-way draws (each search weighs the count of its cell; row 0 is
the point estimate):
  S_km(t)    Kaplan-Meier P(not yet found after t s of search), failures censored at their duration;
  F_found(t), F_fail(t)    the cumulative incidences of finding and of giving up (Aalen-Johansen, competing events,
             S_all(t-) x events at t / searchers at t, summed over the times up to t);
  S_all(t)   P(still searching at t) = 1 - F_found - F_fail. F_found <= 1 - S_km, the difference being the searches that
             would have found after giving up.
At a time a censored search is still at risk (the standard KM convention). The summaries, with 90% bands of twoway.band
and their gaps G - A and A - S (paired draws): the KM median time to find (NaN where S_km does not reach 1/2), S_km at
30, 60 and 120 s, F_found and F_fail at 60 and 120 s, the median duration of the found searches alone (the conditional
view), and the share of searches that are found.

The curves are written on GRID: every GRID_STEP s up to GRID_FINE, then every GRID_COARSE s up to the 99th percentile of the
durations of all the searches (rounded up), all computed on one-second bins.

    uv run python search_km.py --declared-open --dt5 [--draws N]   # writes search_km[_drawsN].json, .pdf and .png
    uv run python search_km.py --declared-open --dt5 --draw        # the figure, from the JSON
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

import longlags
import phase_gaps
import phase_terms
import tail
import twoway
from phase_gaps import CLASS_KEYS, DRAWS, POPS, rows_of, stratum
from tamsd import COLORS, N_BOOT, save

NAME = "search_km" + (f"_draws{DRAWS}" if DRAWS != N_BOOT else "")
GAPS = {"G - A": ("G", "A"), "A - S": ("A", "S")}
GRID_STEP, GRID_FINE, GRID_COARSE = 5, 300, 30  # s
AT = (30, 60, 120)  # s: where the summaries read the curves
CHUNK = 250  # draws handled together
SUMMARIES = ["median time to find (s)", *[f"S_km({t} s)" for t in AT], *[f"F_found({t} s)" for t in AT[1:]],
             *[f"F_fail({t} s)" for t in AT[1:]], "median of the found searches (s)", "searches found (%)"]


def attempts(rows):
    """The durations (s, integer), the event (1 found, 0 failed) and the flight of the searches among `rows`."""
    r = rows[rows["phase"].isin(("found", "failed"))]
    return np.rint(r["s"].to_numpy(float)).astype(int), (r["phase"] == "found").to_numpy(), r["flight"].to_numpy()


def grid_of(d):
    """The times (s) of the curves: GRID_STEP up to GRID_FINE, then GRID_COARSE up to the 99th percentile of `d`."""
    top = int(np.ceil(np.percentile(d, 99) / GRID_COARSE) * GRID_COARSE)
    return np.unique(np.r_[np.arange(0, GRID_FINE + 1, GRID_STEP), np.arange(GRID_FINE, max(top, GRID_FINE) + 1, GRID_COARSE)])


def curves(d, event, cell, weights, grid):
    """The curves of the module docstring for the searches (durations `d`, event, cell code `cell`) under the draws `weights`
    (rows, cells): dict of arrays S_km, S_all, F_found, F_fail read on `grid` (rows, len(grid)), and "median" (rows), the KM median."""
    bins, n = d.max() + 1, weights.shape[1]
    shape = (n, bins)
    found = csr_matrix((event * 1.0, (cell, d)), shape=shape)
    failed = csr_matrix(((~event) * 1.0, (cell, d)), shape=shape)
    out = {k: [] for k in ("S_km", "S_all", "F_found", "F_fail", "median")}
    with np.errstate(divide="ignore", invalid="ignore"):
        for a in range(0, len(weights), CHUNK):
            w = weights[a:a + CHUNK].astype(float)
            f, c = (np.asarray((m.T @ w.T).T) for m in (found, failed))
            risk = np.cumsum((f + c)[:, ::-1], axis=1)[:, ::-1]  # searches with a duration >= t
            hf, hc = np.where(risk > 0, f / risk, 0.0), np.where(risk > 0, c / risk, 0.0)
            s_all = np.cumprod(1 - hf - hc, axis=1)
            before = np.hstack([np.ones((len(w), 1)), s_all[:, :-1]])
            km = np.cumprod(1 - hf, axis=1)
            for k, v in (("S_km", km), ("S_all", s_all), ("F_found", np.cumsum(before * hf, axis=1)), ("F_fail", np.cumsum(before * hc, axis=1))):
                out[k].append(at(v, grid))
            out["median"].append(half(km))
    return {k: np.concatenate(v) for k, v in out.items()}


def at(v, t):
    """The curve rows `v` (rows, bins) read at the times `t` (s), the last bin carried on: (rows, len(t))."""
    return v[:, np.minimum(t, v.shape[1] - 1)]


def half(s):
    """The first time (s) where each row of the survival `s` is at most 1/2, NaN where it never gets there."""
    below = s <= 0.5
    return np.where(below.any(axis=1), below.argmax(axis=1), np.nan).astype(float)


def population(d, event, flight, code, counts, grid):
    """The curves on `grid` and the summaries of one population: arrays over [point, draws]."""
    cell = code.reindex(flight).to_numpy()
    w = np.vstack([np.ones((1, counts.shape[1])), counts])
    c = curves(d, event, cell, w, grid)
    s = {SUMMARIES[0]: c.pop("median")}
    out = {"curves": c, "summaries": s}
    for t in AT:
        s[f"S_km({t} s)"] = c["S_km"][:, grid.tolist().index(t)]
    for k in ("F_found", "F_fail"):
        for t in AT[1:]:
            s[f"{k}({t} s)"] = c[k][:, grid.tolist().index(t)]
    ones = np.ones((1, counts.shape[1]), np.uint8)
    kept = event
    s["median of the found searches (s)"] = np.r_[tail.median_draws(d[kept].astype(float), cell[kept], ones),
                                                  tail.median_draws(d[kept].astype(float), cell[kept], counts)]
    total, size = np.bincount(cell, event * 1.0, minlength=w.shape[1]), np.bincount(cell, minlength=w.shape[1])
    s["searches found (%)"] = 100 * (w @ total) / (w @ size)
    return out


def curve_rows(v):
    """The rows [point, draws] of a curve array (rows, grid) as {value, band} lists over the grid."""
    return {"value": v[0].tolist(), "band": twoway.share_band(v[0], v[1:])}


def quote(v):
    """rows_of(v), or None where the point estimate is NaN (a median that is not reached)."""
    return None if np.isnan(v[0]) else rows_of(v)


def statistics(runs, code, counts):
    """Per class: the number of searches of S, A, G, their curves on one grid, the summaries and the gaps."""
    out = {}
    for key in CLASS_KEYS:
        mine = stratum(runs, key)
        data = {POPS[k]: attempts(mine[mine["pop"] == i]) for i, k in enumerate(POPS)}
        grid = grid_of(np.concatenate([d for d, _, _ in data.values()]))
        got = {pop: population(*x, code, counts, grid) for pop, x in data.items()}
        out[key] = {"searches": {pop: len(x[0]) for pop, x in data.items()},
                    "found": {pop: int(x[1].sum()) for pop, x in data.items()}, "grid": grid.tolist(),
                    "curves": {pop: {k: curve_rows(v) for k, v in g["curves"].items()} for pop, g in got.items()},
                    "summaries": {pop: {n: quote(v) for n, v in g["summaries"].items()} for pop, g in got.items()},
                    "gaps": {gap: {n: quote(got[a]["summaries"][n] - got[b]["summaries"][n]) for n in SUMMARIES}
                             for gap, (a, b) in GAPS.items()}}
        print(key, out[key]["searches"], flush=True)
    return out


# ---- the table and the figure ----

def q(v, d=1):
    return "not reached" if v is None else f"{v['value']:.{d}f} [{v['band'][0]:.{d}f}, {v['band'][1]:.{d}f}]"


def table(result):
    """Print, per class, the searches of S, A, G and, for the summaries of TABLE, the value of each population and the gaps."""
    for name, d in (("median time to find (s)", 0), ("S_km(60 s)", 3), ("median of the found searches (s)", 0), ("searches found (%)", 1)):
        print(f"\n{name}, 90% bands")
        print(f"{'class':15s}{'N_A':>8s}{'N_G':>8s}  {'S':22s}{'A':22s}{'G':22s}{'G - A':24s}{'A - S':24s}")
        for c in CLASS_KEYS:
            r = result["km"][c]
            p, g = r["summaries"], r["gaps"]
            print(f"{c:15s}{r['searches']['A']:8d}{r['searches']['G']:8d}  {q(p['S'][name], d):22s}{q(p['A'][name], d):22s}"
                  f"{q(p['G'][name], d):22s}{q(g['G - A'][name], d + 1):24s}{q(g['A - S'][name], d + 1):24s}")


def band(ax, x, v, color, **kw):
    ax.fill_between(x, v["band"][0], v["band"][1], color=color, alpha=0.25, lw=0)
    ax.plot(x, v["value"], color=color, **{"lw": 1.3, **kw})


def draw(result):
    """Left: S_km of S, A, G on every flight with bands; right: F_found (solid) and F_fail (dashed) of A and G."""
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})
    r = result["km"]["all"]
    grid = np.array(r["grid"])
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))
    for pop, short in POPS.items():
        band(axes[0], grid, r["curves"][short]["S_km"], COLORS[pop], label=short, lw=2.6 if short == "S" else 1.3)
        if short != "S":
            band(axes[1], grid, r["curves"][short]["F_found"], COLORS[pop], label=f"{short} found")
            axes[1].fill_between(grid, *r["curves"][short]["F_fail"]["band"], color=COLORS[pop], alpha=0.2, lw=0)
            axes[1].plot(grid, r["curves"][short]["F_fail"]["value"], color=COLORS[pop], lw=1.3, ls="--", label=f"{short} gave up")
    axes[0].set(xlabel="search time $t$ (s)", ylabel="P(not yet found)", title="survival (Kaplan-Meier)", xlim=(0, GRID_FINE), ylim=(0, 1))
    axes[1].set(xlabel="search time $t$ (s)", title="cumulative incidence", xlim=(0, GRID_FINE), ylim=(0, 0.6))
    for ax in axes:
        ax.grid(color="#eeeeee", lw=0.5)
        ax.legend(frameon=False, fontsize=7.5, handlelength=1.4)
    save(fig, longlags.RESULTS / f"{NAME}.pdf")


if __name__ == "__main__":
    if "--draw" in sys.argv:
        draw(json.loads((longlags.RESULTS / f"{NAME}.json").read_text()))
        sys.exit()
    p, code, counts, classes = phase_gaps.context()
    runs = phase_gaps.assign(phase_terms.table(phase_terms.SET, phase_terms.kept_flights()), p, classes)
    result = {"labels": phase_terms.SET, "rule": phase_terms.RULE, "draws": DRAWS, "classes": CLASS_KEYS, "summaries": SUMMARIES,
              "km": statistics(runs, code, counts)}
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    (longlags.RESULTS / f"{NAME}.json").write_text(json.dumps(result, indent=1))
    table(result)
    draw(result)
