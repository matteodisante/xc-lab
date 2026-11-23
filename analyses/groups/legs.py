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

The chained correlation C* (keys "chained", "chained path") uses every piece at every lag: C*(tau_1) = C_P(tau_1)(tau_1) and
C*(tau_{k+1}) = C*(tau_k) C_P(tau_{k+1})(tau_{k+1}) / C_P(tau_{k+1})(tau_k), with P(tau) the pieces of duration >= tau /
tamsd.SUPPORT and C_S(tau) = sum of sums / sum of counts over S. The curve stops where fewer than MIN_PIECES pieces of the class
and population remain or where a denominator is at or below MIN_DEN. The bands come from one multinomial weight per flight per
draw, shared by S, A, G of a class, so that the difference C*_G - C*_A has a paired band.

The path itself (the 1-s spline, circles kept, no chords) is measured in time with the same heading, origins and lags;
its summary is under the key "path" of legs.json. "peak test" compares, per ground class, the lag of the peak of the absolute
G - A of alpha* (contrasts_ground.json, lags up to PEAK_MAX_S) with tau_c and tau_half of A, for the path and the route.

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
MIN_PIECES = 30  # pieces of a class and population that a lag needs for the chained curve
MIN_DEN = 0.05  # the chain stops where the denominator C_P(tau_k)(tau_{k-1}) is at or below this
PEAK_MAX_S = 2250  # s: the lags of contrasts_ground.json searched for the peak of the absolute G - A
LEVELS = {"time": {"tau_c": np.exp(-1), "tau_half": 0.5}, "space": {"s_c": np.exp(-1), "s_half": 0.5}}
STRATA = (*contrasts.CLASSES, "all")
PART, WORKERS, SAMPLE, POPS = route_arc.PART, route_arc.WORKERS, route_arc.SAMPLE, route_arc.POPS
CACHE_DIR = route_arc.ROOT / LABELS.name / f"legs_head{HEAD_S}_path_{phase_terms.SET}_dt{longlags.MAX_DT}" / (f"sample{SAMPLE}" if SAMPLE else "full")
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
    """One piece [a, b] (s): (length, route time sums, counts, route space sums, counts, path time sums, counts)."""
    g = np.arange(a, b + 1e-6, route_arc.STEP_S)
    path = CubicSpline(t, xy)(g)
    keep = np.ones(len(g), bool)
    for t0, t1 in climbs:
        keep &= ~((g > max(t0, a)) & (g < min(t1, b)))
    route = np.column_stack([np.interp(g, g[keep], path[keep, c]) for c in (0, 1)])
    u, valid = unit((route[HEAD_S:] - route[:-HEAD_S])[::ORIGIN_S])
    time = correlate(u, valid, LAGS_T)
    walk = correlate(*unit((path[HEAD_S:] - path[:-HEAD_S])[::ORIGIN_S]), LAGS_T)
    pos, length = route_arc.resample(route)
    d = np.diff(pos, axis=0)
    return length, *time, *correlate(*unit(d), LAGS_S), *walk


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
    """The arrays of a part from its (row, length, route time sums and counts, route space sums and counts, path time
    sums and counts) tuples."""
    row, length, st, ct, ss, cs, sp, cp = zip(*done)
    return {"row": np.array(row), "length": np.array(length), "sums_t": np.array(st), "counts_t": np.array(ct),
            "sums_s": np.array(ss), "counts_s": np.array(cs), "sums_p": np.array(sp), "counts_p": np.array(cp)}


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
    rng, rng_path = np.random.default_rng(tamsd.SEED), np.random.default_rng(tamsd.SEED + 1)  # the route keeps its draws
    cohort = {"t": d["duration"] >= T_MAX_S / SUPPORT, "s": d["length"] >= S_MAX_M / SUPPORT}
    cohort["p"] = cohort["t"]
    lags_t, k_s = longlags.LONG_LAGS[longlags.LONG_LAGS <= T_MAX_S], LAGS_S[LAGS_S * route_arc.DS_M <= S_MAX_M]
    x = {"t": lags_t.astype(float), "s": k_s * route_arc.DS_M}
    x["p"] = x["t"]
    cols = {"t": slice(0, len(lags_t)), "s": slice(0, len(k_s)), "p": slice(0, len(lags_t))}
    out = {"lags_s": x["t"].tolist(), "arc_m": x["s"].tolist(), "head_s": HEAD_S, "min_move_m": MIN_MOVE_M,
           "cohort": {"t_max_s": T_MAX_S, "min_duration_s": T_MAX_S / SUPPORT, "s_max_m": S_MAX_M,
                      "min_length_m": S_MAX_M / SUPPORT}, "n_boot": N_BOOT, "strata": {}, "path": {}}
    for st in STRATA:
        out["strata"][st], out["path"][st] = {}, {}
        for pop in raw.POPS:
            r = {}
            for kind, name in (("t", "time"), ("s", "space")):
                m = cell(d, cohort[kind], st, pop)
                r[name] = estimate(d["flight"][m], d[f"sums_{kind}"][m][:, cols[kind]], d[f"counts_{kind}"][m][:, cols[kind]],
                                   x[kind], rng, LEVELS[name])
            out["strata"][st][pop] = r
            m = cell(d, cohort["p"], st, pop)
            out["path"][st][pop] = estimate(d["flight"][m], d["sums_p"][m][:, cols["p"]], d["counts_p"][m][:, cols["p"]],
                                            x["p"], rng_path, LEVELS["time"])
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


def nan(v):
    return float("nan") if v is None else v


def state(c, name, r):
    """'reached', 'start' (C below the level at the first lag) or 'never' (not below within the lags)."""
    level = LEVELS["time"][name]
    v = nan(c["value"][0])
    return "reached" if np.isfinite(c[name]["value"]) else ("start" if v < level else "never")


def show(c, name, r):
    s = state(c, name, r)
    return f"{c[name]['value']:.0f}" if s == "reached" else (f"<{int(r['lags_s'][0])}" if s == "start" else f">{int(r['lags_s'][-1])}")


def showband(c, name, r):
    b = c[name]["band"]
    return "" if state(c, name, r) != "reached" or None in b else f"[{b[0]:.0f}, {b[1]:.0f}]"


def table_path(r):
    """tau_c [band] and tau_half of the path (time), per class and population."""
    print("path (circles kept), time; <10: C is already below the level at the first lag, >last: never below")
    print("'-' in the band column: no crossing")
    print(f"{'class':15s}{'pop':>4s}{'pieces':>8s}{'tau_c [90%]':>22s}{'tau_half':>10s}{'C(last)':>9s}")
    for st in STRATA:
        for pop, short in raw.POPS.items():
            c = r["path"][st][pop]
            print(f"{st:15s}{short:>4s}{c['pieces']:>8d}{show(c, 'tau_c', r):>10s} {showband(c, 'tau_c', r):>11s}{show(c, 'tau_half', r):>10s}{nan(c['value'][-1]):9.3f}")


def peak_test(r):
    """Per class: the lag of the peak of the absolute G - A of alpha* (contrasts_ground.json, lags <= PEAK_MAX_S) against the
    memory of A, in the path and in the route, and the lag where C_G - C_A is largest."""
    saved = json.loads((longlags.RESULTS / "contrasts_ground.json").read_text())
    lags, last = np.array(r["lags_s"], float), int(r["lags_s"][-1])
    out = {}
    for st in STRATA:
        gap = saved[st]["gaps"]["G - A"]
        gl, gv = np.array(gap["lags"], float), np.array([np.nan if v is None else v for v in gap["value"]], float)
        gv[gl > PEAK_MAX_S] = np.nan
        peak = float(gl[np.nanargmax(gv)])
        res = {"peak_lag_s": peak, "peak_gap": float(np.nanmax(gv))}
        for kind, src in (("path", r["path"][st]), ("route", {p: v["time"] for p, v in r["strata"][st].items()})):
            c = {p: src[p] for p in ("group_solo", "group")}
            d = np.array(c["group"]["value"], float) - np.array(c["group_solo"]["value"], float)
            if np.isnan(d).all():
                d = np.zeros_like(d)
            k = {}
            for pop, tag in (("group_solo", "A"), ("group", "G")):
                k[tag] = {n: {**c[pop][n], "state": state(c[pop], n, r)} for n in ("tau_c", "tau_half")}
                k[tag]["C_last"] = nan(c[pop]["value"][-1])
            for n in ("tau_c", "tau_half"):
                a = k["A"][n]
                k[f"ratio_peak_{n}_A"] = peak / a["value"] if a["state"] == "reached" else None  # else tau(A) is out of the lags
            k["gap_C_peak_lag_s"], k["gap_C_peak"] = float(lags[np.nanargmax(d)]), float(np.nanmax(d))
            res[kind] = k
        out[st] = res
    print(f"\npeak test: peak = argmax of the absolute G - A of alpha* over lags <= {PEAK_MAX_S} s; last lag {last} s")
    for st in STRATA:
        res, src = out[st], {"path": r["path"][st], "route": {p: v["time"] for p, v in r["strata"][st].items()}}
        src = {kind: {"A": v["group_solo"], "G": v["group"]} for kind, v in src.items()}
        print(f"\n{st}: peak lag {res['peak_lag_s']:.0f} s (gap {res['peak_gap']:.3f})")
        for kind in ("path", "route"):
            k = res[kind]
            for tag in ("A", "G"):
                q = k[tag]
                cc = src[kind][tag]
                print(f"  {kind:5s} {tag}: tau_c {show(cc, 'tau_c', r):>6s} {showband(cc, 'tau_c', r):13s} tau_half {show(cc, 'tau_half', r):>6s} "
                      f"{showband(cc, 'tau_half', r):13s} C(10) {nan(cc['value'][0]):.3f}  C({last}) {q['C_last']:.3f}")
            ratio = lambda n: f"{k['ratio_peak_' + n + '_A']:.2f}" if k["ratio_peak_" + n + "_A"] is not None else (
                "n/a" if k["A"]["C_last"] != k["A"]["C_last"] else (f">{res['peak_lag_s'] / r['lags_s'][0]:.0f}" if k["A"][n]["state"] == "start"
                                                                    else f"<{res['peak_lag_s'] / last:.2f}"))
            print(f"  {kind:5s} peak/tau_c(A) {ratio('tau_c')}, peak/tau_half(A) {ratio('tau_half')}; "
                  f"max C_G - C_A = {k['gap_C_peak']:.3f} at {k['gap_C_peak_lag_s']:.0f} s")
    r["peak test"] = out


# ---- the chained correlation ----

def chain(idx, size, dur, sums, counts, w):
    """C* of the pieces (flight index `idx` in 0..size-1, duration, sums and counts at every lag of LONG_LAGS) for the point
    estimate and for the draws of flight weights `w`. Returns the lags used, the curve, the draws (N_BOOT x lags), the pieces
    per lag and why it stopped."""
    s, n = np.nan_to_num(sums), counts.astype(float)
    value, draws, pieces, stop = [], [], [], ("longest", None)
    cum, cum_draws = 1.0, np.ones(len(w))

    def agg(m, k):
        a = np.bincount(idx[m], s[m, k], size)
        b = np.bincount(idx[m], n[m, k], size)
        with np.errstate(invalid="ignore", divide="ignore"):
            return a.sum() / b.sum(), (w @ a) / (w @ b)

    for k, tau in enumerate(longlags.LONG_LAGS):
        m = dur >= tau / SUPPORT
        if m.sum() < MIN_PIECES:
            stop = ("pieces", int(tau))
            break
        num, num_d = agg(m, k)
        if k == 0:
            cum, cum_draws = num, num_d
        else:
            den, den_d = agg(m, k - 1)
            if not den > MIN_DEN:
                stop = ("denominator", int(tau))
                break
            den_d = np.where(den_d > 0, den_d, np.nan)
            cum, cum_draws = cum * num / den, cum_draws * num_d / den_d
        value.append(cum), draws.append(cum_draws), pieces.append(int(m.sum()))
    return {"lags": longlags.LONG_LAGS[:len(value)].astype(float), "value": np.array(value), "draws": np.array(draws).T,
            "pieces": pieces, "stop": stop}


def curve_summary(c, levels):
    """The chained curve `c` as a dict: value, band, pieces, stop, and tau_c / tau_half with their bands."""
    out = {"lags": c["lags"].tolist(), "value": c["value"].tolist(),
           "band": [np.nanpercentile(c["draws"], q, axis=0).tolist() for q in BAND], "pieces": c["pieces"],
           "stop": {"reason": c["stop"][0], "lag_s": c["stop"][1]}, "last_lag_s": float(c["lags"][-1])}
    for name, level in levels.items():
        out[name] = {"value": float(crossing(c["lags"], c["value"][None], level)[0]), "band": band(crossing(c["lags"], c["draws"], level))}
    return out


def gap_summary(a, g):
    """C*_G - C*_A on the lags both curves have, with the paired band (differences taken inside each draw), the lag of its
    maximum and the band of that lag."""
    m = min(len(a["value"]), len(g["value"]))
    lags, v, d = g["lags"][:m], g["value"][:m] - a["value"][:m], g["draws"][:, :m] - a["draws"][:, :m]
    k = int(np.nanargmax(v))
    top = np.array([lags[np.nanargmax(x)] if np.isfinite(x).any() else np.nan for x in d])
    return {"lags": lags.tolist(), "value": v.tolist(), "band": [np.nanpercentile(d, q, axis=0).tolist() for q in BAND],
            "max_lag_s": float(lags[k]), "max_value": float(v[k]), "max_value_band": band(d[:, k]), "max_lag_band": band(top)}


def chained_summary(d, kind):
    """The chained curves of the route (kind "t") or of the path ("p") per class and population, with the paired gap."""
    rng = np.random.default_rng(tamsd.SEED)
    out = {"n_boot": N_BOOT, "min_pieces": MIN_PIECES, "min_denominator": MIN_DEN, "support": SUPPORT, "strata": {}}
    for st in STRATA:
        inside = (d["class"] == st) | (st == "all")
        fl, inv = np.unique(d["flight"][inside], return_inverse=True)
        w = rng.multinomial(len(fl), np.full(len(fl), 1 / len(fl)), N_BOOT).astype(float)
        chains = {}
        for pop in raw.POPS:
            m = d["population"][inside] == pop
            chains[pop] = chain(inv[m], len(fl), d["duration"][inside][m], d[f"sums_{kind}"][inside][m], d[f"counts_{kind}"][inside][m], w)
        out["strata"][st] = {pop: curve_summary(c, LEVELS["time"]) for pop, c in chains.items()}
        out["strata"][st]["G - A"] = gap_summary(chains["group_solo"], chains["group"])
    return out


def fixed_check(r, key, fixed):
    """The largest |C* - C| per class (over the populations and the lags of the fixed cohort); `fixed(st, pop)` gives its curve."""
    lags = np.array(r["lags_s"], float)
    out = {}
    for st in STRATA:
        worst = (0.0, None, None)
        for pop in raw.POPS:
            c = r[key]["strata"][st][pop]
            x, v = np.array(c["lags"]), np.array(c["value"])
            k = x <= lags[-1]
            diff = np.abs(v[k] - np.array(fixed(st, pop), float)[:k.sum()])
            if diff.max() > worst[0]:
                worst = (float(diff.max()), pop, float(x[k][diff.argmax()]))
        out[st] = {"max_abs_diff": worst[0], "population": worst[1], "lag_s": worst[2]}
    r[key]["check_fixed_cohort"] = out
    return out


def peak_test_chained(r, key):
    """Per class, with the chained curves: the lag of the peak of the absolute G - A of alpha* (as peak_test) against tau_c and
    tau_half of A and G, and where C*_G - C*_A is largest, with the paired band at the peak lag."""
    saved = json.loads((longlags.RESULTS / "contrasts_ground.json").read_text())
    out = {}
    for st in STRATA:
        gap = saved[st]["gaps"]["G - A"]
        gl, gv = np.array(gap["lags"], float), np.array([np.nan if v is None else v for v in gap["value"]], float)
        gv[gl > PEAK_MAX_S] = np.nan
        peak = float(gl[np.nanargmax(gv)])
        c = r[key]["strata"][st]
        res = {"peak_lag_s": peak}
        for pop, tag in (("group_solo", "A"), ("group", "G")):
            res[tag] = {n: c[pop][n] for n in ("tau_c", "tau_half")} | {"last_lag_s": c[pop]["last_lag_s"], "stop": c[pop]["stop"]}
        for n in ("tau_c", "tau_half"):
            v = res["A"][n]["value"]
            res[f"ratio_peak_{n}_A"] = peak / v if np.isfinite(v) else None
        g = c["G - A"]
        x = np.array(g["lags"])
        j = int(np.argmin(np.abs(x - peak)))
        lo, hi = g["band"][0][j], g["band"][1][j]
        res["gap_C"] = {"max_lag_s": g["max_lag_s"], "max_value": g["max_value"], "max_value_band": g["max_value_band"],
                        "max_lag_band": g["max_lag_band"], "at_peak_lag_s": float(x[j]), "at_peak": g["value"][j],
                        "at_peak_band": [lo, hi], "peak_band_excludes_zero": bool(lo > 0 or hi < 0)}
        out[st] = res
    r[f"peak test {key}"] = out


def show_chained(r, key, title):
    c = r[key]
    f = lambda q: "   -  " if not np.isfinite(q["value"]) else f"{q['value']:6.0f}"
    g = lambda q: "" if None in q["band"] else f"[{q['band'][0]:.0f}, {q['band'][1]:.0f}]"
    print(f"\n{title}: chained curves; N_BOOT {N_BOOT}, >= {MIN_PIECES} pieces, denominator > {MIN_DEN}")
    print(f"{'class':15s}{'pop':>4s}{'last lag':>9s}{'stop':>16s}{'tau_c [90%]':>22s}{'tau_half [90%]':>22s}")
    for st in STRATA:
        for pop, short in raw.POPS.items():
            q = c["strata"][st][pop]
            stop = q["stop"]["reason"] + (f" {q['stop']['lag_s']}" if q["stop"]["lag_s"] else "")
            print(f"{st:15s}{short:>4s}{q['last_lag_s']:9.0f}{stop:>16s}{f(q['tau_c'])} {g(q['tau_c']):>15s}{f(q['tau_half'])} {g(q['tau_half']):>15s}")
    print("check against the fixed cohort (lags <= %d s): largest |C* - C|" % r["lags_s"][-1])
    for st in STRATA:
        q = c["check_fixed_cohort"][st]
        print(f"  {st:15s}{q['max_abs_diff']:.4f}  ({raw.POPS[q['population']]} at {q['lag_s']:.0f} s)")
    p = r[f"peak test {key}"]
    print(f"peak test, chained ({title})")
    print(f"{'class':15s}{'peak':>6s}{'tau_c(A)':>9s}{'ratio':>7s}{'max lag [90%]':>20s}{'max C*_G-C*_A [90%]':>26s}{'at peak [90%]':>24s}{'excl 0':>8s}")
    for st in STRATA:
        q = p[st]
        gp = q["gap_C"]
        ratio = "n/a" if q["ratio_peak_tau_c_A"] is None else f"{q['ratio_peak_tau_c_A']:.2f}"
        print(f"{st:15s}{q['peak_lag_s']:6.0f}{f(q['A']['tau_c']):>9s}{ratio:>7s}"
              f"{gp['max_lag_s']:8.0f} [{gp['max_lag_band'][0]:.0f}, {gp['max_lag_band'][1]:.0f}]"
              f"{gp['max_value']:9.3f} [{gp['max_value_band'][0]:.3f}, {gp['max_value_band'][1]:.3f}]"
              f"{gp['at_peak']:8.3f} [{gp['at_peak_band'][0]:.3f}, {gp['at_peak_band'][1]:.3f}]{str(gp['peak_band_excludes_zero']):>8s}")


def chained_all(d, r):
    for key, kind, fixed in (("chained", "t", lambda st, pop: r["strata"][st][pop]["time"]["value"]),
                             ("chained path", "p", lambda st, pop: r["path"][st][pop]["value"])):
        r[key] = chained_summary(d, kind)
        fixed_check(r, key, fixed)
        peak_test_chained(r, key)
        show_chained(r, key, "route" if kind == "t" else "path")


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
    figure(r, longlags.RESULTS / f"legs{SUFFIX}.pdf")
    table(r)
    table_path(r)
    peak_test(r)
    chained_all(d, r)
    (longlags.RESULTS / f"legs{SUFFIX}.json").write_text(json.dumps(r))


if __name__ == "__main__":
    assert longlags.DT5, "run with --dt5"
    summary() if "--summary" in sys.argv else run()
