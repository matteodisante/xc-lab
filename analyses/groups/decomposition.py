"""How much of the raw gap between group time and solo flight each factor accounts for: point 3 of the
comparisons plan, G side only.

Factors: R the route (route.RHO, cover MAIN_COVER, same direction: the rows of the match table), T the season
and hour (the month and hour calipers of the chosen caliper of matched), W the same wing (class or level, as
the chosen caliper). For each of the 8 subsets C of F = {R, T, W}, Delta(C) = G minus the controls that satisfy
only the conditions of C:

    with R     the rows of the match table (side G, isolation off, both durations long enough for the dataset)
               that satisfy the other conditions of C;
    without R  windows of the solo pieces of route.TRACKS (flights with no contact of any length) on another date,
               as long as the piece (same number of grid points), at most K per piece, one per pool flight, drawn
               with tamsd.SEED. With T the month of the window's middle is within the month caliper of the
               piece's and the window is placed at random among the positions whose middle is within the hour
               caliper (local time of day, across midnight); otherwise it starts at random. With W the pool
               piece has the wing of the piece.

Every cell uses the same pieces (those with a control in F on the dataset, and a window in every cell without R;
for the phase statistics also those with phase time and a phased control in every cell), the same pool and the same
draws of twoway.py (matched.Context). Quantities: H_loc(1000 s) on dataset `1000`, H_loc(1840 s) on `2760`, the fitted H of
both, the seven phase statistics (dataset `1000`). The Shapley value of each factor, for every draw and year:

    phi_k = sum over C in F without k of |C|! (2 - |C|)! / 3!  [Delta(C) - Delta(C with k)]
    Delta(none) = Delta(F) + phi_R + phi_T + phi_W        (exact)

    uv run python decomposition.py --open [--sample N]    # after matched.py (the match table and its counts)
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curves
import isolated
import matched
import route
import twoway
from blocks import with_takeoffs
from matched import DATASETS, LEVEL_NAMES, MAIN_COVER, METRICS, NAMES
from tamsd import CACHE, COLORS, GRID_S, LABELS, N_BOOT, OUT, SEED, SUPPORT, read_flights

K = 50  # windows per piece in the cells without R
R, T, W = 1, 2, 4
FACTORS = (("R", R), ("T", T), ("W", W))
FULL = R | T | W
SHAPLEY = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}  # |C|! (2 - |C|)! / 3!
TOL = 1e-9  # the identity holds to this on every draw
NEAR_ZERO = 1e-6  # |Delta(none)| below this has no shares
QUANTITIES = (("H_loc(1000 s)", "1000"), ("H_loc(1840 s)", "2760"), ("H (10 s to 1000 s)", "1000"), ("H (10 s to T)", "2760"),
              *((n, "phase") for n in NAMES))
FOREST = matched.FOREST
RAMP = ("#b9d3f0", "#7fb0e8", "#2a78d6", "#123f7a")  # controls: light to dark with the number of conditions
STYLES = ("-", "--", ":", "-.")
ROW_COLORS = {"Δ(none)": "#555555", "φ_W": "#8fb8e8", "φ_T": "#2a78d6", "φ_R": "#123f7a", "Δ(F)": COLORS["group"]}


def cell_name(c):
    """Names of the factors of the subset c (a mask), "none" for the empty set."""
    return "+".join(n for n, b in FACTORS if c & b) or "none"


def shapley(delta):
    """The Shapley value of R, T, W from `delta` = {mask: array}, the same shape for every mask (draws, years, ...).
    Asserts Delta(none) = Delta(F) + phi_R + phi_T + phi_W to TOL (NaN where any term is)."""
    phi = {name: sum(SHAPLEY[bin(c).count("1")] * (delta[c] - delta[c | k]) for c in range(8) if not c & k)
           for name, k in FACTORS}
    total = delta[FULL] + sum(phi.values())
    assert np.allclose(total, delta[0], rtol=0, atol=TOL, equal_nan=True), "Shapley identity broken"
    return phi


def shares(delta0, parts):
    """parts[k] / delta0 (floats); None where |delta0| < NEAR_ZERO or a value is not finite."""
    ok = np.isfinite(delta0) and abs(delta0) >= NEAR_ZERO
    return {k: (float(v / delta0) if ok and np.isfinite(v) else None) for k, v in parts.items()}


def month_of(utc):
    """Calendar month (1-12) of UTC seconds."""
    return pd.to_datetime(np.asarray(utc, float), unit="s", utc=True).month.to_numpy()


def circular(a, b):
    """Circular difference of months."""
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, 12 - d)


class Pool:
    """The pool pieces as arrays: UTC of the first point, points, date, wing label ("" when none) and flight."""

    def __init__(self, utc, n, date, label, flight):
        self.utc, self.n, self.date, self.label, self.flight = utc, n, date, label, flight
        self.tod0 = matched.time_of_day(utc)
        self.m0, self.m1 = month_of(utc), month_of(utc + (n - 1) * GRID_S)


def windows_of(pool, n, date, label, mid, cell, rng, calipers, k=K):
    """Up to k windows of n points inside pool pieces of at least n points, on another date, one per pool flight,
    for the piece (date, wing label, UTC of its middle `mid`), under the conditions of `cell` (T, W bits):
    (pool rows, first point). `calipers` = (month, hour). The month and hour are checked with matched.month_gap
    and hour_gap on the windows kept."""
    month, hour = calipers
    ok = (pool.n >= n) & (pool.date != date)
    if cell & W:
        ok &= (pool.label == label) & (label != "")
    if cell & T:
        m = month_of([mid])[0]
        ok &= (circular(pool.m0, m) <= month) | (circular(pool.m1, m) <= month)
        tod = matched.time_of_day([mid])[0]
    rows, starts, seen = [], [], set()
    for r in rng.permutation(np.flatnonzero(ok)):
        if pool.flight[r] in seen:
            continue
        s = np.arange(pool.n[r] - n + 1)
        if cell & T:
            d = np.abs((pool.tod0[r] + (2 * s + n - 1) * GRID_S / 7200) % 24 - tod)
            s = s[np.minimum(d, 24 - d) <= hour - 1e-6]
            if not len(s):
                continue
        rows.append(r)
        starts.append(rng.choice(s))
        seen.add(pool.flight[r])
        if len(rows) == k:
            break
    rows, starts = np.array(rows, int), np.array(starts, int)
    if cell & T and len(rows):
        mids = pool.utc[rows] + (2 * starts + n - 1) * GRID_S / 2
        good = (matched.month_gap(mids, np.full(len(mids), mid)) <= month) & (matched.hour_gap(mids, np.full(len(mids), mid)) <= hour + 1e-9)
        rows, starts = rows[good], starts[good]
    return rows, starts


def window_table(tracks, flights, pieces, level, label, calipers, first):
    """Rows (piece, solo, i, j, level_piece, cell) of the windows of the cells without R for `pieces` (track
    indices); the rows are labelled from `first`."""
    info = flights.set_index("flight").reindex(tracks["flight"])
    utc0, date = info["utc0"].to_numpy(float), info["date"].to_numpy()
    n, start, flight = tracks["n"], tracks["start"], tracks["flight"]
    lab = label.reindex(flight).fillna("").to_numpy(str)
    sol = np.flatnonzero(tracks["population"] == "solo")
    pool = Pool(utc0[sol] + start[sol], n[sol], date[sol], lab[sol], flight[sol])
    lv = level.reindex(flight[pieces]).to_numpy()
    out = []
    for cell in (c for c in range(8) if not c & R):
        for g, lvl in zip(pieces, lv):
            rng = np.random.default_rng([SEED, int(g), cell])
            rows, s = windows_of(pool, int(n[g]), date[g], lab[g], utc0[g] + start[g] + (n[g] - 1) * GRID_S / 2, cell, rng, calipers)
            out.append(pd.DataFrame({"piece": g, "solo": sol[rows], "i": s, "j": s + n[g] - 1, "level_piece": lvl, "cell": cell}))
    out = pd.concat(out, ignore_index=True)
    out.index = out.index + first
    return out


def cell_rows(base, win, cell, dataset, calipers, same):
    """The controls of `cell` on `dataset` from the match rows `base` (G side, cover and isolation of the main
    result) and the windows `win`."""
    if cell & R:
        month, hour = calipers
        shortest = DATASETS[dataset] / SUPPORT
        keep = (base["dur_piece"] >= shortest) & (base["dur_solo"] >= shortest)
        if cell & T:
            keep &= (base["dmonth"] <= month) & (base["dhour"] <= hour + 1e-9)
        if cell & W:
            keep &= base["level_piece"].notna() & (base[f"{same}_piece"] == base[f"{same}_solo"])  # as matched.select
        return base[keep]
    return win[win["cell"] == cell]


def counts_of(tracks, rows):
    """Pieces, controls and distinct control flights of `rows`."""
    return {"pieces": int(rows["piece"].nunique()), "controls": int(len(rows)),
            "control flights": int(np.unique(tracks["flight"][rows["solo"].to_numpy()]).size)}


def decompose(ctx, M, sets, pick):
    """For each cell the series of the quantities of QUANTITIES: `sets` maps "1000", "2760", "phase" to the pieces
    kept, `pick(cell, set name)` gives the rows of the cell for those pieces. Returns {cell: (e, counts)} with e the
    dict of matched.draws."""
    out = {}
    for cell in range(8):
        rows = {name: pick(cell, name) for name in sets}
        e = {"main": matched.msd_series(ctx, M, rows["1000"], 1000, True) if len(rows["1000"]) else None,
             "long_T": 2760,
             "long": matched.msd_series(ctx, M, rows["2760"], 2760, True) if len(rows["2760"]) else None,
             "phases": matched.phase_series(ctx, M, rows["phase"], True) if len(rows["phase"]) else None}
        out[cell] = (e, {name: counts_of(ctx.tracks, r) for name, r in rows.items()})
    return out


def report(cells):
    """The JSON of one level: per quantity the 8 Delta(C) with values and counts, the phi, Delta(F) and shares."""
    draws = {c: matched.draws(e) for c, (e, _) in cells.items()}
    out = {}
    for metric, name in QUANTITIES:
        if any(metric not in draws[c] for c in range(8)):
            continue
        pair = {c: tuple(None if x is None else x - y for x, y in zip(draws[c][metric]["group"], draws[c][metric]["control"]))
                for c in range(8)}
        both = [0, 1] if pair[0][1] is not None else [0]
        parts = {i: shapley({c: pair[c][i] for c in range(8)}) for i in both}
        quote = lambda f: twoway.quote(f(0), f(1) if 1 in parts else None)
        phi = {k: quote(lambda i, k=k: parts[i][k]) for k, _ in FACTORS}
        point = {k: float(parts[0][k][0]) for k, _ in FACTORS}
        d0 = float(pair[0][0][0])
        out[metric] = {
            "set": name,
            "cells": {cell_name(c): {"delta": twoway.quote(pair[c][0], pair[c][1]),
                                     "group": float(draws[c][metric]["group"][0][0]),
                                     "control": float(draws[c][metric]["control"][0][0]), **cells[c][1][name]}
                      for c in range(8)},
            "delta_none": quote(lambda i: pair[0][i]), "delta_F": quote(lambda i: pair[FULL][i]), "phi": phi,
            "shares": shares(d0, {**point, "F": float(pair[FULL][0][0])})}
    return out


def draw_decomposition(result, path):
    """One panel per quantity of FOREST: Delta(none), phi_W, phi_T, phi_R, Delta(F), each with its band."""
    rows = []
    for label, get in (("Δ(none)", lambda q: q["delta_none"]), ("φ_W", lambda q: q["phi"]["W"]), ("φ_T", lambda q: q["phi"]["T"]),
                       ("φ_R", lambda q: q["phi"]["R"]), ("Δ(F)", lambda q: q["delta_F"])):
        rows.append((label, ROW_COLORS[label], {m: _plain(get(q)) for m, q in result.items() if m in FOREST}))
    matched.draw_forest(rows, path, "Open paragliders: the raw gap of group time and its parts", stats=[m for m in FOREST if m in result],
                        width=8.2)


def _plain(quote):
    """A quote with arrays turned into floats and the years kept as [min, max]."""
    return {k: (float(v) if k == "value" else [float(x) for x in v]) for k, v in quote.items()}


def draw_curves(cells, folder, tag, tracks):
    """MSD and H_loc figures of G and of the controls of the 8 cells (level "all"), datasets `1000` and `2760`."""
    for key, name in (("main", "1000"), ("long", "2760")):
        drawn = []
        for c in range(8):
            ser = cells[c][0][key]
            if ser is None:
                continue
            if not drawn:
                g = matched.curve_of(ser, "group")
                lo, hi = g["H_band"]
                drawn.append(curves.series(g, f"G ({g['pieces']:,} pieces): H = {g['H']:.3f} [{lo:.3f}, {hi:.3f}]", COLORS["group"]))
            h = matched.curve_of(ser, "control")
            lo, hi = h["H_band"]
            k = bin(c).count("1")
            label = f"{cell_name(c)} ({cells[c][1][name]['controls']:,} controls): H = {h['H']:.3f} [{lo:.3f}, {hi:.3f}]"
            drawn.append(curves.series(h, label, RAMP[k], STYLES[sum(1 for d in range(c) if bin(d).count('1') == k)]))
        if drawn:
            T_ = drawn[0]["T"]
            title = f"decomposition, lags to {T_:,} s, pieces of at least {T_ / SUPPORT:g} s"
            curves.figures(drawn, folder / f"decomposition{tag}_{name}", title)
            left = [{**drawn[0], "label": "G"}] + [{**matched.curve_of(cells[c][0][key], "control"), "label": f"controls of {'∅' if c == 0 else cell_name(c)}",
                                                    "color": RAMP[bin(c).count("1")]} for c in (0, FULL) if cells[c][0][key]]
            right = []
            for c in range(8):
                if cells[c][0][key] is None:
                    continue
                k, ends = bin(c).count("1"), c in (0, FULL)
                d = matched.delta_curve(cells[c][0][key])
                right.append({**d, "band": d["band"] if ends else None, "label": f"Δ({'∅' if c == 0 else cell_name(c)})", "color": RAMP[k],
                              "ls": STYLES[sum(1 for e in range(c) if bin(e).count('1') == k)], "lw": 1.4 if ends else 0.9})
            curves.delta_figure(left, right, folder / f"decomposition{tag}_{name}_delta", title)


def main(sample):
    flights = with_takeoffs(read_flights())
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    tracks = route.read_tracks(flights, None)
    tag = f"_sample{sample}" if sample else ""
    table = pd.read_parquet(CACHE / "comparisons_matches.parquet")
    first = int(table.index.max()) + 1
    table = table[table["side"] == "G"]
    chosen = json.loads((OUT / "matched_counts.json").read_text())["chosen caliper"]["name"]
    month, hour, same = matched.CALIPERS[chosen]
    calipers = (month, hour)
    level, cls = isolated.levels(flights), isolated.wing_class(flights)
    label = cls.reindex(level.index) if same == "class" else level  # the classes of isolated.LEVELS only

    # the pieces with a control in F, per dataset
    sets = {d: np.unique(matched.select(table, d, chosen)["piece"].to_numpy()) for d in DATASETS}
    if sample:
        keep = np.sort(np.random.default_rng(SEED).choice(sets["1000"], min(sample, len(sets["1000"])), replace=False))
        sets = {d: np.intersect1d(p, keep) for d, p in sets.items()}
    union = sets["1000"]
    print({d: len(p) for d, p in sets.items()}, file=sys.stderr, flush=True)

    # controls: the match rows for the pieces, the windows of the cells without R
    base = table[(table["cover"] >= MAIN_COVER - 1e-9) & table["piece"].isin(union)
                 & (table["dur_piece"] >= 1000 / SUPPORT) & (table["dur_solo"] >= 1000 / SUPPORT)]
    win = window_table(tracks, flights, union, level, label, calipers, first)
    print("windows:", len(win), win.groupby("cell").size().to_dict(), file=sys.stderr, flush=True)
    ctx = matched.Context(tracks, flights, summary)
    M = matched.Measured(ctx, pd.concat([base[["piece", "solo", "i", "j"]], win[["piece", "solo", "i", "j"]]]))
    print("measured", file=sys.stderr, flush=True)
    rows_of = lambda cell, dataset: cell_rows(base, win, cell, dataset, calipers, same)

    # the same pieces in every cell: a control in F, and a window in every cell without R
    have = set(union)
    for cell in range(8):
        if not cell & R:
            have &= set(win.loc[win["cell"] == cell, "piece"])
    dropped = int(len(union) - len(have))
    kept = {d: np.array(sorted(set(p) & have)) for d, p in sets.items()}
    # the phase statistics: phase time and a phased control in every cell
    phased = set(kept["1000"][M.T_g[np.searchsorted(M.piece, kept["1000"])] > 0])
    for cell in range(8):
        r = rows_of(cell, "1000")
        r = r[r["piece"].isin(phased)]
        phased &= set(r["piece"][M.T_s[np.searchsorted(M.rows, r.index.to_numpy())] > 0])
    kept["phase"] = np.array(sorted(phased))
    print({d: len(p) for d, p in kept.items()}, "dropped for lack of windows:", dropped, file=sys.stderr, flush=True)

    result = {"caliper": chosen, "cover": MAIN_COVER, "rho": route.RHO, "K": K, "n_boot": N_BOOT, "years": ctx.years.tolist(),
              "cells": [cell_name(c) for c in range(8)], "pieces": {d: len(p) for d, p in kept.items()},
              "pieces dropped for lack of windows": dropped, "windows": win.groupby("cell").size().rename(cell_name).to_dict(),
              "levels": {}}
    figure_cells = None
    for lv in LEVEL_NAMES:
        def pick(cell, name):
            ds = "1000" if name == "phase" else name
            r = rows_of(cell, ds)
            r = r[r["piece"].isin(kept[name])]
            r = matched.of_level(r, lv)
            if name == "phase":
                r = r[M.T_s[np.searchsorted(M.rows, r.index.to_numpy())] > 0]
            return r
        cells = decompose(ctx, M, kept, pick)
        result["levels"][lv] = report(cells)
        if lv == "all":
            figure_cells = cells
        print("level", lv, "done", file=sys.stderr, flush=True)
    result["check"] = check_point2(ctx, M, table, chosen, kept, figure_cells)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"decomposition{tag}.json").write_text(json.dumps(matched.jsonable(result), indent=1))
    draw_decomposition(result["levels"]["all"], OUT / f"decomposition{tag}.pdf")
    draw_curves(figure_cells, OUT, tag, tracks)
    show(result)


def check_point2(ctx, M, table, chosen, kept, cells):
    """The cell F against matched.py's main selection on the same pieces: the largest absolute difference of
    the draws of Delta H_loc and Delta H (0 when the cell is exact)."""
    out = {}
    for ds, T_, key in (("1000", 1000, "main"), ("2760", 2760, "long")):
        sel = matched.select(table, ds, chosen)
        sel = sel[sel["piece"].isin(kept[ds])]
        ref = matched.msd_series(ctx, M, sel, T_, False)["boot"]["delta"]
        mine = cells[FULL][0][key]["boot"]["delta"]
        out[ds] = {"pieces": int(sel["piece"].nunique()),
                   "max |difference| over the draws": {f: float(np.nanmax(np.abs(ref[f] - mine[f]))) for f in ("H_loc", "H")}}
    return out


def show(result):
    print(json.dumps({k: v for k, v in result.items() if k not in ("levels",)}, indent=1))
    for lv, quantities in result["levels"].items():
        print(lv)
        for m, q in quantities.items():
            d = [f"{cell_name(c)}={q['cells'][cell_name(c)]['delta']['value']:.3f}" for c in range(8)]
            phi = {k: round(float(v["value"]), 3) for k, v in q["phi"].items()}
            print(f"  {m:34} n={q['cells']['none']['pieces']:5} " + " ".join(d), "| phi", phi,
                  "F", round(float(q["delta_F"]["value"]), 3), "shares", {k: None if v is None else round(v, 2) for k, v in q["shares"].items()})


if __name__ == "__main__":
    main(route.option("--sample", 0) or None)
