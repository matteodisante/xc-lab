"""How long the direction of the route stays correlated, in time and in arc length, per ground class and population.

The pieces are those of route_arc.py (S, A, G; declared open paragliders, logger interval up to longlags.MAX_DT s).
The route of a piece is a function of real time on a 1-s grid: the path (cubic spline through the fixes), except
inside each climb run of our refit (same runs and chord rule as route_arc.route_piece, cut at the piece boundary),
where the position moves linearly in time from the first to the last point of the run. The heading at t is the angle
of the displacement of the route over HEAD_S s; it is undefined where that displacement is below MIN_MOVE_M. For each
lag tau of longlags.LONG_LAGS the part stores, per piece, the sum over the origins (every ORIGIN_S s) of
cos(theta(t + tau) - theta(t)) and the number of origins with both headings defined. The spatial version resamples
the route every route_arc.DS_M m of arc length, takes the heading of each step, and stores the same two numbers at the
lags of route_arc.LAGS steps.

The summary keeps a fixed cohort (pieces long enough for every lag drawn: duration >= T_MAX_S / tamsd.SUPPORT in time,
route length >= S_MAX_M / tamsd.SUPPORT in arc length) and gives per ground class and population C(tau) = sum of sums /
sum of counts (one weight per origin), tau_c and tau_half where C falls to 1/e and to 1/2 (linear in log tau), and the
same in metres (s_c); bands are the 90% interval of a bootstrap over flights.

    uv run python legs.py --declared-open --dt5 [--sample N]            # reads the fixes; resumable
        # one part per PART flights in CACHE_DIR (--sample: scratch folder, no resume)
    uv run python legs.py --declared-open --dt5 [--sample N] --summary  # legs.json, legs.pdf, a table
"""

import json
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import contrasts
import longlags
import phase_terms
import raw
import route_arc
import tamsd
from common.legacy import stream_flights
from tamsd import FIXES, LABELS, OUT, SUPPORT

HEAD_S = 10  # s: the displacement of the route that gives the heading
MIN_MOVE_M = 20.0  # m: below this displacement the heading is undefined
ORIGIN_S = 10  # s: the grid of origins (and the unit of the lags of LONG_LAGS)
T_MAX_S = 3600  # s: the longest lag of the summary (cohorts.SELECTION_FREE)
S_MAX_M = 26000.0  # m: the longest arc-length lag of the summary (about T_MAX_S at the median route speed)
N_BOOT = 200
BAND = (5, 95)
LEVELS = {"time": {"tau_c": np.exp(-1), "tau_half": 0.5}, "space": {"s_c": np.exp(-1), "s_half": 0.5}}
STRATA = (*contrasts.CLASSES, "all")
PART, WORKERS, SAMPLE, POPS = route_arc.PART, route_arc.WORKERS, route_arc.SAMPLE, route_arc.POPS
CACHE_DIR = route_arc.ROOT / LABELS.name / f"legs_head{HEAD_S}_{phase_terms.SET}_dt{longlags.MAX_DT}" / (f"sample{SAMPLE}" if SAMPLE else "full")
LAGS_T = longlags.LONG_LAGS // ORIGIN_S  # lags in origin steps
LAGS_S = route_arc.LAGS  # lags in steps of route_arc.DS_M
SUFFIX = f"_sample{SAMPLE}" if SAMPLE else ""


def correlate(u, valid, lags):
    """Sums of u(i + k) . u(i) (= cos of the heading change) and numbers of origins with both headings defined,
    at each k of `lags` (NaN and 0 where k is out of reach). u is zero where the heading is undefined."""
    sums, counts = np.full(len(lags), np.nan), np.zeros(len(lags), np.int64)
    for j, k in enumerate(lags):
        if k < len(u):
            sums[j], counts[j] = (u[k:] * u[:-k]).sum(), (valid[k:] & valid[:-k]).sum()
    return sums, counts


def unit(d):
    """Unit vectors of the rows of `d` (zero where the norm is below MIN_MOVE_M) and where they are defined."""
    n = np.hypot(*d.T)
    valid = n >= MIN_MOVE_M
    return np.where(valid[:, None], d / np.where(valid, n, 1.0)[:, None], 0.0), valid


def legs_piece(t, xy, a, b, climbs):
    """One piece [a, b] (s): the route in time, then (length, time sums, time counts, space sums, space counts)."""
    g = np.arange(a, b + 1e-6, route_arc.STEP_S)
    path = CubicSpline(t, xy)(g)
    keep = np.ones(len(g), bool)
    for t0, t1 in climbs:
        keep &= ~((g > max(t0, a)) & (g < min(t1, b)))
    route = np.column_stack([np.interp(g, g[keep], path[keep, c]) for c in (0, 1)])
    u, valid = unit((route[HEAD_S:] - route[:-HEAD_S])[::ORIGIN_S])
    time = correlate(u, valid, LAGS_T)
    pos, length = route_arc.resample(route)
    d = np.diff(pos, axis=0)
    return length, *time, *correlate(*unit(d), LAGS_S)


def job(fixes, rows, climbs):
    """The rows (index in the cache, start, duration) of one flight, measured on its fixes."""
    fixes = fixes.sort_values("t", kind="stable")
    seg = fixes["segment_id"].to_numpy()
    t, xy = fixes["t"].to_numpy(float), fixes[["E", "N"]].to_numpy(float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    out = []
    for i, a, d in rows:
        b = a + d
        for s in np.unique(seg):
            q = np.flatnonzero(seg == s)
            if t[q[0]] - 1e-6 <= a and b <= t[q[-1]] + 1e-6:
                m = route_arc.MARGIN_STEPS * dt
                lo, hi = np.searchsorted(t[q], a - m), np.searchsorted(t[q], b + m, "right")
                q = q[lo:hi]
                q = q[np.r_[True, np.diff(t[q]) > 0]]
                out.append((i, *legs_piece(t[q], xy[q], a, b, climbs[(climbs[:, 1] > a) & (climbs[:, 0] < b)])))
                break
    return out


def pack(done):
    """The arrays of a part from its (row, length, time sums, time counts, space sums, space counts) tuples."""
    row, length, st, ct, ss, cs = zip(*done)
    return {"row": np.array(row), "length": np.array(length), "sums_t": np.array(st), "counts_t": np.array(ct),
            "sums_s": np.array(ss), "counts_s": np.array(cs)}


def run():
    flights = phase_terms.kept_flights()
    p = longlags.load_pieces()
    todo, climbs = route_arc.pieces_by_flight(p), route_arc.climbs_by_flight(flights)
    wanted = set(list(todo)[:SAMPLE]) if SAMPLE else set(todo)
    number = {int(fid): n for fid, n in flights["flight"].items()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    state = {"chunk": 0, "pending": set(), "done": [], "seen": 0}

    def collect(done):
        state["done"] += [x for f in done for x in f.result()]

    def save():
        collect(wait(state["pending"])[0])
        state["pending"] = set()
        if state["done"]:
            np.savez(CACHE_DIR / f"part{state['chunk']:04d}.npz", **pack(state["done"]))
            print("part", state["chunk"], "saved,", state["seen"], "flights", flush=True)
        state["done"] = []

    with ProcessPoolExecutor(WORKERS) as pool:
        for fixes in stream_flights(FIXES["para"], ["segment_id", "t", "E", "N"]):
            flight = number.get(int(fixes["flight_id"].iat[0]))
            if flight not in wanted:
                continue
            chunk = state["seen"] // PART
            state["seen"] += 1
            if chunk != state["chunk"]:
                save()
                state["chunk"] = chunk
            if (CACHE_DIR / f"part{chunk:04d}.npz").exists():
                continue
            state["pending"].add(pool.submit(job, fixes, todo[flight], climbs.get(flight, np.zeros((0, 2)))))
            if len(state["pending"]) >= 4 * WORKERS:
                done, state["pending"] = wait(state["pending"], return_when=FIRST_COMPLETED)
                collect(done)
        save()


# ---- the summary ----

def load():
    """The pieces of the parts of CACHE_DIR with their flight, population and duration: a dict of arrays."""
    parts = [np.load(f) for f in sorted(CACHE_DIR.glob("part*.npz"))]
    d = {k: np.concatenate([q[k] for q in parts]) for k in parts[0].files}
    p = longlags.load_pieces()
    return {**d, **{k: p[k][d["row"]] for k in ("flight", "population", "duration")}}


def crossing(x, c, level):
    """Where the curve `c` (rows: curves) first falls below `level`, linear in log x between the points around it
    (NaN if it never does or starts below)."""
    out = np.full(len(c), np.nan)
    for i, row in enumerate(c):
        j = np.flatnonzero(row < level)
        if len(j) and j[0] > 0:
            j = j[0]
            w = (row[j - 1] - level) / (row[j - 1] - row[j])
            out[i] = np.exp(np.log(x[j - 1]) + w * (np.log(x[j]) - np.log(x[j - 1])))
    return out


def band(v):
    return [float(q) for q in np.nanpercentile(v, BAND)] if np.isfinite(v).any() else [None, None]


def estimate(flight, sums, counts, x, rng, levels):
    """C = sum / count over the pieces, its crossings, and their bootstrap bands over flights."""
    f, inv = np.unique(flight, return_inverse=True)
    if not len(f):
        nan = [None] * len(x)
        return {"value": nan, "band": [nan, nan], "flights": 0, "pieces": 0, **{k: {"value": float("nan"), "band": [None, None]} for k in levels}}
    s, n = np.zeros((len(f), sums.shape[1])), np.zeros((len(f), sums.shape[1]))
    np.add.at(s, inv, np.nan_to_num(sums))
    np.add.at(n, inv, counts)
    w = rng.multinomial(len(f), np.full(len(f), 1 / len(f)), N_BOOT)
    with np.errstate(invalid="ignore", divide="ignore"):
        c, boot = s.sum(0) / n.sum(0), (w @ s) / (w @ n)
    out = {"value": c.tolist(), "band": [np.nanpercentile(boot, q, axis=0).tolist() for q in BAND],
           "flights": len(f), "pieces": len(flight)}
    for name, level in levels.items():
        out[name] = {"value": float(crossing(x, c[None], level)[0]), "band": band(crossing(x, boot, level))}
    return out


def cell(d, mask, stratum, pop):
    return mask & (d["population"] == pop) & ((d["class"] == stratum) | (stratum == "all"))


def summarise(d):
    rng = np.random.default_rng(tamsd.SEED)
    cohort = {"t": d["duration"] >= T_MAX_S / SUPPORT, "s": d["length"] >= S_MAX_M / SUPPORT}
    lags_t, k_s = longlags.LONG_LAGS[longlags.LONG_LAGS <= T_MAX_S], LAGS_S[LAGS_S * route_arc.DS_M <= S_MAX_M]
    x = {"t": lags_t.astype(float), "s": k_s * route_arc.DS_M}
    cols = {"t": slice(0, len(lags_t)), "s": slice(0, len(k_s))}
    out = {"lags_s": x["t"].tolist(), "arc_m": x["s"].tolist(), "head_s": HEAD_S, "min_move_m": MIN_MOVE_M,
           "cohort": {"t_max_s": T_MAX_S, "min_duration_s": T_MAX_S / SUPPORT, "s_max_m": S_MAX_M,
                      "min_length_m": S_MAX_M / SUPPORT}, "n_boot": N_BOOT, "strata": {}}
    for st in STRATA:
        out["strata"][st] = {}
        for pop in raw.POPS:
            r = {}
            for kind, name in (("t", "time"), ("s", "space")):
                m = cell(d, cohort[kind], st, pop)
                r[name] = estimate(d["flight"][m], d[f"sums_{kind}"][m][:, cols[kind]], d[f"counts_{kind}"][m][:, cols[kind]],
                                   x[kind], rng, LEVELS[name])
            out["strata"][st][pop] = r
    return out


def table(r):
    f = lambda q: "   -   " if q["value"] != q["value"] else f"{q['value']:7.0f}"
    g = lambda q: "" if None in q["band"] else f"[{q['band'][0]:.0f}, {q['band'][1]:.0f}]"
    print(f"{'class':15s}{'pop':>4s}{'pieces t/s':>12s}{'tau_c [90%]':>24s}{'tau_half':>10s}{'s_c [90%]':>24s}")
    for st in STRATA:
        for pop, short in raw.POPS.items():
            c = r["strata"][st][pop]
            print(f"{st:15s}{short:>4s}{c['time']['pieces']:>6d}/{c['space']['pieces']:<5d}"
                  f"{f(c['time']['tau_c'])} {g(c['time']['tau_c']):>15s}{f(c['time']['tau_half']):>10s}"
                  f"{f(c['space']['s_c'])} {g(c['space']['s_c']):>15s}")


def figure(r, path):
    """C(tau) and C(s) of S per ground class (band: 90% over flights)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, kind, x, label in ((axes[0], "time", r["lags_s"], r"lag $\tau$ (s)"), (axes[1], "space", r["arc_m"], r"arc length $s$ (m)")):
        x = np.array(x)
        for st in STRATA:
            c = r["strata"][st]["solo"][kind]
            color = contrasts.ALL_COLOR if st == "all" else contrasts.CLASS_COLORS[st]
            ax.fill_between(x, *np.array(c["band"], float), color=color, alpha=0.15, lw=0)
            ax.plot(x, c["value"], color=color, lw=1.5, ls="--" if st == "all" else "-", label=st)
        ax.axhline(np.exp(-1), color="k", lw=0.5, ls=":")
        ax.set(xscale="log", xlabel=label, ylabel=r"$C$ (heading correlation, S)")
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].legend(frameon=False, fontsize=7)
    tamsd.save(fig, path)


def summary():
    d = load()
    d["class"] = pd.read_parquet(OUT / "ground.parquet").set_index("flight")["class"].reindex(d["flight"]).to_numpy()
    assert (d["counts_t"] >= 0).all() and (np.abs(d["sums_t"]) <= d["counts_t"] + 1e-6)[d["counts_t"] > 0].all()
    r = summarise(d)
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    (longlags.RESULTS / f"legs{SUFFIX}.json").write_text(json.dumps(r))
    figure(r, longlags.RESULTS / f"legs{SUFFIX}.pdf")
    table(r)


if __name__ == "__main__":
    assert longlags.DT5, "run with --dt5"
    summary() if "--summary" in sys.argv else run()
