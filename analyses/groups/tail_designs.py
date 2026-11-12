"""The tail of the phase durations of paragliders in the three designs of the comparisons, on the open flights.

The estimator and the bands are those of tail.py: the Hill mu of P(X > x) ~ x^-mu above one threshold,
mu = n / sum ln(x / xmin), and the two-way band of twoway.py (draws over pilots, blocks and cells, the same rows
for every population, so that a contrast has the band of its own draws). The threshold of each phase is the xmin
of S in the raw design (the Kolmogorov-Smirnov choice of powerlaw.Fit, read from tail.json) and every population
of every design is read at it; the band does not include the uncertainty of the threshold, which the Hill plot
against xmin shows. Next to each band, the textbook standard error of independent data, mu / sqrt(n).

1. Raw: S, A, G as in tail.py, with the gaps G - S, A - S, G - A, and the xmin that minimises the KS distance in
   each population (`ks_curve`).
2. Same route: the weighted mu of G, S|G, A, S|A and of the contrasts, for the three phases, by the weights of
   matched.py (as tail.same_route, which it reproduces for the glides); and the weighted medians of the descriptors
   of the long glides in G, S|G, A, S|A with their bands.
3. Same flight: the runs of the open flights that have a group stretch, by the stretch that holds them: BEFORE
   (the solo time before the first group stretch), IN GROUP, AFTER (the solo time after the last), BETWEEN (solo
   time between two group stretches), SPANNING (a run that does not end in the stretch where it starts). mu of
   BEFORE, IN GROUP, AFTER, and IN GROUP - mean(BEFORE, AFTER), AFTER - BEFORE; variants: SPANNING runs given
   to the class of their start, and only the stretches of at least STRETCH_FACTOR xmin (a run longer than its
   stretch cannot exist, so a short stretch cannot hold a run well above the threshold).
4. What the long glides (at least xmin) are: the descriptors of tail.py and the height lost in total, as medians
   with two-way bands, in every population and class of the three designs. The descriptors of a glide are those
   of the cache of tail.py.

    uv run python tail_designs.py --open [--sample N]   # after tail.py --open (its JSON and its cache of glides)
    # --sample N: the flights of N random open flights; writes tail_designs_sampleN.json
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import isolated
import longlags
import matched
import raw
import route
import survival
import tail
import twoway
from blocks import with_takeoffs
from tamsd import CACHE, LABELS, OUT, SEED, folders, read_flights, stretches

PHASES = ("transition", "climb", "search")
STRETCH_FACTOR = 2  # variant: stretches of at least this many xmin, long enough for a run well above the threshold
KS_GRID = np.geomspace(300, 4000, 60)  # s: the xmin tried for the KS distance
MIN_TAIL = 50  # runs above an xmin for its KS distance to be taken
CLASSES = ("before", "in group", "after")
OTHER = ("between", "spanning")
TOLERANCES = (10, 30)  # s: a run may overshoot its stretch by this much (checked, not used)
LENGTH_QUANTILES = (10, 25, 50, 75, 90)
DESCRIPTORS = (*tail.DESCRIPTORS, "height lost, total (m)")
CHUNK = 500  # draws made together for the weights of the runs
LONG_CACHE = CACHE / "tail_glides.parquet"


# ---- one estimate and its contrasts -----------------------------------------------------------------------------

def estimate(x, cell, counts, xmin):
    """The Hill mu of the durations `x` (cell codes `cell`) above `xmin`: ({value, band, n, se_iid}, draws), the
    draws of the 3 N_BOOT rows of `counts`. se_iid = mu / sqrt(n), the error of n independent runs."""
    sums = tail.hill_sums(x, cell, counts.shape[1], [xmin])
    total = sums.sum(0)
    draws = np.vstack([tail.ratio(np.hstack([counts[a:a + tail.CHUNK] @ sums[:, :1], counts[a:a + tail.CHUNK] @ sums[:, 1:]]))
                       for a in range(0, len(counts), tail.CHUNK)])[:, 0]
    value, n = float(tail.ratio(total)[0]), int(total[0])
    return quote(value, draws, n=n, se=value / np.sqrt(n) if n else np.nan), draws


def quote(value, draws, n=None, se=np.nan):
    """The entry of a value and its two-way band (the stacked `draws`), with the runs above and the iid error."""
    lo, hi = twoway.band(value, draws)
    return {"value": float(value), "band": [float(lo), float(hi)], "n": n, "se_iid": float(se)}


def combine(parts):
    """The linear combination of estimates: `parts` = [(coefficient, (entry, draws)), ...] gives its entry (the iid
    error adds in quadrature) and its draws."""
    value = sum(c * e["value"] for c, (e, _) in parts)
    draws = sum(c * d for c, (_, d) in parts)
    se = np.sqrt(sum((c * e["se_iid"]) ** 2 for c, (e, _) in parts))
    return quote(value, draws, se=se), draws


def ks_curve(x, grid, minimum=MIN_TAIL):
    """Per xmin of `grid`: the Hill mu above it and the Kolmogorov-Smirnov distance between the empirical
    distribution of x >= xmin and the fitted Pareto, D = max |F_emp - F_fit| (Clauset, Shalizi, Newman 2009); NaN
    where fewer than `minimum` runs lie above. Returns ({xmin: ...}, the xmin of the smallest D)."""
    x = np.sort(np.asarray(x, float))
    mu, d = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    for j, m in enumerate(grid):
        t = x[np.searchsorted(x, m, "left"):]
        if len(t) < minimum:
            continue
        mu[j] = len(t) / np.log(t / m).sum()
        fit = 1 - (t / m) ** -mu[j]
        d[j] = max(np.abs(np.searchsorted(t, t, "right") / len(t) - fit).max(),
                   np.abs(np.searchsorted(t, t, "left") / len(t) - fit).max())
    best = float(grid[np.nanargmin(d)]) if np.isfinite(d).any() else None
    return {"xmin": list(map(float, grid)), "mu": mu.tolist(), "D": d.tolist()}, best


def curve_of(x, cell, counts, grid):
    """tail.mu_curve over `grid` and the KS curve."""
    k, best = ks_curve(x, grid_with(grid))
    return {"hill": tail.mu_curve(x, cell, counts, grid), "ks": k, "xmin by KS": best}


def grid_with(grid):
    return np.unique(np.r_[KS_GRID, grid])


# ---- descriptors of the long glides ------------------------------------------------------------------------------

def read_descriptors():
    """The cache of tail.py (one row per glide of S, A, G with its descriptors) with the height lost in total."""
    gl = pd.read_parquet(LONG_CACHE).drop(columns=["population", "n_fixes"], errors="ignore")
    gl["height lost, total (m)"] = -gl["vertical speed (m/s)"] * (gl["t_end"] - gl["t_start"])
    keys = ["flight", "segment_id", "t_start", "t_end"]
    return gl.astype({"flight": np.int64, "segment_id": np.int64}).drop_duplicates(keys).set_index(keys)[list(DESCRIPTORS)]


def with_descriptors(runs, cache):
    """The rows of `runs` (flight, segment_id, t_start, t_end, ...) with their DESCRIPTORS (NaN when the glide is not in
    the cache: a run whose middle is in no piece)."""
    keys = pd.MultiIndex.from_arrays([runs[c].to_numpy().astype(np.int64 if c in ("flight", "segment_id") else float)
                                      for c in ("flight", "segment_id", "t_start", "t_end")])
    got = cache.reindex(keys)
    return runs.assign(**{c: got[c].to_numpy() for c in DESCRIPTORS})


def weighted_median(v, weights):
    """The lower weighted median of the values `v` for each column of `weights` (len(v), k): NaN where a column sums to 0."""
    k = weights.shape[1]
    if len(v) == 0:
        return np.full(k, np.nan)
    o = np.argsort(v, kind="stable")
    v, cum = v[o], np.cumsum(weights[o], axis=0)
    out = v[(2 * cum >= cum[-1:]).argmax(axis=0)]
    out[cum[-1] <= 0] = np.nan
    return out


def describe(rows, cell, counts, weights=None):
    """Per descriptor, from the long glides `rows` (a frame with the DESCRIPTORS and, for a weighted population, the
    weights of its runs per draw: `weights`, (runs, 1 + draws), the point estimate first): the number of glides with a
    value, the quartiles, and the median with its two-way band. Returns ({descriptor: entry}, {descriptor: (median,
    draws)})."""
    out, med = {"runs": int(len(rows)), "last run of the flight (share)": float(rows["last"].mean()) if len(rows) else np.nan}, {}
    for name in DESCRIPTORS:
        v = rows[name].to_numpy(float)
        ok = np.isfinite(v)
        if weights is None:
            m = float(np.median(v[ok])) if ok.any() else np.nan
            d = tail.median_draws(v[ok], cell[ok], counts) if ok.any() else np.full(len(counts), np.nan)
        else:
            both = weighted_median(v[ok], weights[ok])
            m, d = both[0], both[1:]
        med[name] = (m, d)
        out[name] = {"n": int(ok.sum()), **tail.quartiles(v[ok]), "median": float(m),
                     "median band": [float(b) for b in twoway.band(m, d)] if ok.any() else [np.nan, np.nan]}
    return out, med


def gaps_of(meds, terms):
    """Per descriptor, the combination `terms` = [(coefficient, key of meds), ...] of the medians with its band."""
    out = {}
    for name in DESCRIPTORS:
        value = sum(c * meds[k][name][0] for c, k in terms)
        draws = sum(c * meds[k][name][1] for c, k in terms)
        out[name] = {"value": float(value), "band": [float(b) for b in twoway.band(value, draws)]} if np.isfinite(value) else None
    return out


# ---- design 1: raw ----------------------------------------------------------------------------------------------

def raw_design(pops, cell, counts, xmins, cache):
    """Per phase: per population S, A, G the estimate at the xmin of S, the Hill and KS curves and the gaps; for the
    glides also the descriptors of the long ones."""
    out = {}
    for phase in PHASES:
        data, est, res = {pop: tail.phase_data(pops[pop], phase, cell) for pop in raw.POPS}, {}, {}
        for pop, (x, c, rows) in data.items():
            short = raw.POPS[pop]
            est[short] = estimate(x, c, counts, xmins[phase])
            res[short] = {"runs": len(x), "flights": int(pops[pop]["flight"].nunique()), "at xmin of S": est[short][0],
                          **curve_of(x, c, counts, tail.XMIN_GRID)}
        gaps = {g: combine([(1, est[a]), (-1, est[b])])[0] for g, (a, b) in {"G - S": ("G", "S"), "A - S": ("A", "S"), "G - A": ("G", "A")}.items()}
        out[phase] = {"xmin of S": xmins[phase], "populations": res, "gaps": gaps}
        if phase == "transition":
            meds, desc = {}, {}
            for pop, (x, c, rows) in data.items():
                long = rows[rows["n_fixes"] >= xmins[phase]]
                desc[raw.POPS[pop]], meds[raw.POPS[pop]] = describe(with_descriptors(long, cache), c[(x >= xmins[phase])], counts)
            out[phase]["descriptors"] = desc
            out[phase]["descriptor gaps"] = {g: gaps_of(meds, [(1, a), (-1, b)]) for g, (a, b) in
                                             {"G - S": ("G", "S"), "A - S": ("A", "S"), "G - A": ("G", "A")}.items()}
    return out


# ---- design 3: the same flight ------------------------------------------------------------------------------------

def group_stretches(flights, membership, summary):
    """The stretches (flight clock) of the flights that have a group stretch, each with its class in its flight:
    "in group", "before" (solo, before the first group stretch), "after", "between"; and its length."""
    s = stretches(flights, membership, summary)
    s = s[s["flight"].isin(s.loc[s["population"] == "group", "flight"].unique())]
    s = s.sort_values(["flight", "a"], kind="stable").reset_index(drop=True)
    g, i = s["population"] == "group", s.groupby("flight").cumcount()
    first, last = (i.where(g).groupby(s["flight"]).transform(f) for f in ("min", "max"))
    s["cls"] = np.select([g, i < first, i > last], ["in group", "before", "after"], "between")
    s["length"] = s["b"] - s["a"]
    return s


def classify(stretch, runs, tol=0.0):
    """Per run (flight, t_start, t_end): `cls`, the class of the stretch that holds the whole run, "spanning" when
    the run ends after the stretch of its start (by more than `tol` s), "outside" when its start is in no stretch (before
    the departure); `start`, the class of the stretch of its start ("outside" if none); `length`, that stretch's length."""
    key = lambda f, t: np.asarray(f, float) * isolated.RANK + np.asarray(t, float)
    ka, kb = key(stretch["flight"], stretch["a"]), key(stretch["flight"], stretch["b"])
    flight = runs["flight"].to_numpy()
    start, end = key(flight, runs["t_start"]), key(flight, runs["t_end"])
    k = np.clip(np.searchsorted(ka, start + tol, "right") - 1, 0, len(ka) - 1)
    ok = (stretch["flight"].to_numpy()[k] == flight) & (start < kb[k])
    cls = stretch["cls"].to_numpy()[k]
    start_cls = np.where(ok, cls, "outside")
    whole = np.where(end <= kb[k] + tol, start_cls, "spanning")
    return runs.assign(cls=np.where(ok, whole, "outside"), start=start_cls,
                       length=np.where(ok, stretch["length"].to_numpy()[k], np.nan))


def length_table(stretch, xmin):
    """Per class: the stretches, their lengths at LENGTH_QUANTILES and the share that last at least xmin and at least
    STRETCH_FACTOR xmin."""
    out = {}
    for c in (*CLASSES, "between"):
        d = stretch.loc[stretch["cls"] == c, "length"].to_numpy(float)
        out[c] = {"stretches": int(len(d)), "flights": int(stretch.loc[stretch["cls"] == c, "flight"].nunique()),
                  "length quantiles (s)": dict(zip(map(str, LENGTH_QUANTILES), np.percentile(d, LENGTH_QUANTILES).tolist())),
                  "share >= xmin": float((d >= xmin).mean()), f"share >= {STRETCH_FACTOR} xmin": float((d >= STRETCH_FACTOR * xmin).mean()),
                  f"stretches >= {STRETCH_FACTOR} xmin": int((d >= STRETCH_FACTOR * xmin).sum())}
    return out


def flight_design(runs, stretch, cell, counts, xmins, cache, in_raw):
    """Per phase and variant the estimates of the classes and the two contrasts. `runs`: the counted runs of the
    flights of `stretch`, classified. `in_raw`: per run, the population of raw.POPS whose piece holds it (or None)."""
    out = {}
    for phase in PHASES:
        r, xmin = runs[runs["phase"] == phase], xmins[phase]
        r = r[tail.trimmed(r["n_fixes"].to_numpy(float))]  # the cut of the durations on the runs of all these flights
        out[phase] = {"xmin of S": xmin, "variants": {}, "curves": {}, "other": {}}
        variants = {"whole in stretch": (r["cls"], r["length"] >= 0),
                    "spanning by its start": (r["start"], r["length"] >= 0)}
        if phase == "transition":
            variants[f"stretches of {STRETCH_FACTOR} xmin"] = (r["cls"], r["length"] >= STRETCH_FACTOR * xmin)
        for name, (label, keep) in variants.items():
            est = {}
            for c in (*CLASSES, *OTHER):
                m = (label == c) & keep
                x = r.loc[m, "n_fixes"].to_numpy(float)
                est[c] = estimate(x, cell.reindex(r.loc[m, "flight"]).to_numpy().astype(int), counts, xmin)
            terms = {"in group - mean(before, after)": [(1, est["in group"]), (-0.5, est["before"]), (-0.5, est["after"])],
                     "after - before": [(1, est["after"]), (-1, est["before"])]}
            out[phase]["variants"][name] = {"classes": {c: e for c, (e, _) in est.items()},
                                            "contrasts": {k: combine(v)[0] for k, v in terms.items()}}
        for c in (*CLASSES, *OTHER):
            m = r["cls"] == c
            x, cc = r.loc[m, "n_fixes"].to_numpy(float), cell.reindex(r.loc[m, "flight"]).to_numpy().astype(int)
            out[phase]["curves"][c] = curve_of(x, cc, counts, tail.XMIN_GRID)
            out[phase]["other"][c] = {"runs": int(m.sum()), "runs above": int((x >= xmin).sum()),
                                      "flights above": int(r.loc[m & (r["n_fixes"] >= xmin), "flight"].nunique())}
        if phase == "transition":
            out[phase]["survival"] = {}
            for c in CLASSES:
                m = r["cls"] == c
                out[phase]["survival"][c] = {"runs": int(m.sum()), **tail_survival(r.loc[m], cell, counts)}
            long = r[r["n_fixes"] >= xmin]
            long = long.assign(raw=in_raw.reindex(long.index).fillna("none").to_numpy())
            out[phase]["long by class and raw population"] = {c: long.loc[long["cls"] == c, "raw"].value_counts().to_dict()
                                                             for c in (*CLASSES, *OTHER, "outside")}
            out[phase]["descriptors"], out[phase]["descriptor gaps"] = flight_descriptors(long, cell, counts, cache)
            out[phase]["tolerance"] = tolerance_counts(stretch, runs, phase, xmin)
    return out


def tail_survival(rows, cell, counts):
    """survival.survival_curve of the durations of `rows`, with the share band."""
    return survival.survival_curve(rows["n_fixes"].to_numpy(float), cell.reindex(rows["flight"]).to_numpy().astype(int), counts,
                                   band=twoway.share_band)


def flight_descriptors(long, cell, counts, cache):
    """The descriptors of the long glides of the classes, and the gaps of their medians."""
    desc, meds = {}, {}
    for c in (*CLASSES, *OTHER):
        rows = long[long["cls"] == c]
        desc[c], meds[c] = describe(with_descriptors(rows, cache), cell.reindex(rows["flight"]).to_numpy().astype(int), counts)
    gaps = {"in group - mean(before, after)": gaps_of(meds, [(1, "in group"), (-0.5, "before"), (-0.5, "after")]),
            "after - before": gaps_of(meds, [(1, "after"), (-1, "before")])}
    return desc, gaps


def tolerance_counts(stretch, runs, phase, xmin):
    """The glides above xmin by class when a run may overshoot its stretch by TOLERANCES s."""
    r = runs[(runs["phase"] == phase) & (runs["n_fixes"] >= xmin)]
    out = {"0 s": r["cls"].value_counts().to_dict()}
    for tol in TOLERANCES:
        out[f"{tol} s"] = classify(stretch, r, tol)["cls"].value_counts().to_dict()
    return out


# ---- design 2: the same route -------------------------------------------------------------------------------------

def route_sides(ctx, table, line):
    """Per side G, A: the rows of the match table the way tail.same_route takes them: (windows, selection with phase time,
    the positions of its pieces and of its rows in the windows)."""
    out = {}
    for side in ("G", "A"):
        main = matched.select(table[table["side"] == side], "1000", **line)
        w = tail.windows(ctx, main)
        sel = matched.phased(w, main)
        out[side] = (w, sel, np.searchsorted(w.piece, np.unique(sel["piece"].to_numpy())), np.searchsorted(w.rows, sel.index.to_numpy()))
    return out


def phase_sums(ctx, win, xmins, phase):
    """tail.glide_sums for any phase: per window, hill_sums of its trimmed runs of the phase."""
    unit, idx = matched.window_runs(ctx, *win)
    x, ph = ctx.runs["n_fixes"].to_numpy(float)[idx], ctx.runs["phase"].to_numpy()[idx] == phase
    unit, x = unit[ph], x[ph]
    keep = tail.trimmed(x)
    return tail.hill_sums(x[keep], unit[keep], len(win[0]), xmins)


def route_design(ctx, sides, xmins, cache):
    """Per phase: the weighted mu of G, S|G, A, S|A over the Hill grid and at the xmin of S, with the contrasts; for the
    glides the descriptors of the long ones."""
    out, k = {}, len(tail.XMIN_GRID) + 1
    for phase in PHASES:
        grid = [*tail.XMIN_GRID, xmins[phase]]
        boots, est = {}, {}
        for side, own, control in (("G", "G", "S|G"), ("A", "A", "S|A")):
            w, sel, pos, rows = sides[side]
            a = phase_sums(ctx, tuple(v[pos] for v in w.win_g), grid, phase)
            b = phase_sums(ctx, tuple(v[rows] for v in w.win_s), grid, phase)
            (group, ctrl, _), _ = matched.run_engine(ctx, sel, a, b, False)
            boots[own], boots[control] = tail.ratio(group.T), tail.ratio(ctrl.T)
            n = {own: a[:, :k].sum(0), control: ctrl[:k, 0]}  # runs above; the controls weighted
            for name, label in ((own, own), (control, control)):
                v, d = boots[name][0], boots[name][1:]
                est[name] = (quote(v[-1], d[:, -1], n=float(n[name][-1]), se=v[-1] / np.sqrt(n[name][-1]) if n[name][-1] > 0 else np.nan), d[:, -1])
        terms = {"G - S|G": [(1, est["G"]), (-1, est["S|G"])], "A - S|A": [(1, est["A"]), (-1, est["S|A"])]}
        terms["(G - S|G) - (A - S|A)"] = terms["G - S|G"] + [(-c, e) for c, e in terms["A - S|A"]]
        curves = {s: {"xmin": list(map(float, grid[:-1])), "value": boots[s][0][:-1].tolist(),
                      "band": [b[:-1] for b in twoway.band(boots[s][0], boots[s][1:])]} for s in boots}
        out[phase] = {"xmin of S": xmins[phase], "populations": {s: e for s, (e, _) in est.items()},
                      "contrasts": {name: combine(t)[0] for name, t in terms.items()}, "curves": curves,
                      "pieces": {s: int(len(sides[s][2])) for s in sides}, "controls": {s: int(len(sides[s][1])) for s in sides}}
        if phase == "transition":
            out[phase]["descriptors"], out[phase]["descriptor gaps"] = route_descriptors(ctx, sides, xmins[phase], cache)
    return out


def route_weights(ctx, sel, columns):
    """The weights of the runs of the pieces of `sel` and of their controls for the draws `columns` (0 the point
    estimate), as isolated.engine gives them: (the weight of each piece, (pieces, columns); the weight of each match
    in the control side, (matches, columns)): the weight w_g of a piece shared among its controls by the weights of
    their flights, zero where the piece or its controls weigh 0."""
    pieces, p = np.unique(sel["piece"].to_numpy(), return_inverse=True)
    of = ctx.boot[0]
    wp = of(ctx.row(ctx.tracks["flight"][pieces]), columns)
    wf = of(ctx.row(ctx.tracks["flight"][sel["solo"].to_numpy()]), columns)
    mass = csr_matrix((np.ones(len(p)), (p, np.arange(len(p)))), shape=(len(pieces), len(p))) @ wf
    ok = (wp > 0) & (mass > 0)
    share = np.divide(wp, mass, out=np.zeros_like(wp), where=ok)
    return np.where(ok, wp, 0.0), share[p] * wf


def route_descriptors(ctx, sides, xmin, cache):
    """The descriptors of the glides of at least xmin (trimmed as in the estimates) in the pieces of G and A and in
    their controls, by the weights of the engine (route_weights), and the gaps of the medians."""
    n_draws = ctx.boot[1]
    desc, meds = {}, {}
    for side, own, control in (("G", "G", "S|G"), ("A", "A", "S|A")):
        w, sel, pos, rows = sides[side]
        picked = []
        for win in (tuple(v[pos] for v in w.win_g), tuple(v[rows] for v in w.win_s)):
            unit, idx = matched.window_runs(ctx, *win)
            r = ctx.runs.iloc[idx]
            glide, x = r["phase"].to_numpy() == "transition", r["n_fixes"].to_numpy(float)
            keep = np.zeros(len(r), bool)
            keep[glide] = tail.trimmed(x[glide])
            picked.append((unit[keep & (x >= xmin)], r[keep & (x >= xmin)]))
        (ug, rg), (uc, rc) = picked
        wg, wc = np.zeros((len(ug), n_draws)), np.zeros((len(uc), n_draws))
        for a in range(0, n_draws, CHUNK):
            cols = np.arange(a, min(a + CHUNK, n_draws))
            piece_weight, match_weight = route_weights(ctx, sel, cols)
            wg[:, cols], wc[:, cols] = piece_weight[ug], match_weight[uc]
        for name, rows_, wt in ((own, rg, wg), (control, rc, wc)):
            desc[name], meds[name] = describe(with_descriptors(rows_, cache), None, None, wt)
    gaps = {"G - S|G": gaps_of(meds, [(1, "G"), (-1, "S|G")]), "A - S|A": gaps_of(meds, [(1, "A"), (-1, "S|A")])}
    gaps["(G - S|G) - (A - S|A)"] = gaps_of(meds, [(1, "G"), (-1, "S|G"), (-1, "A"), (1, "S|A")])
    return desc, gaps


# ---- driver ------------------------------------------------------------------------------------------------------

def compare_with_tail(result, old):
    """The largest difference between the numbers of the raw design here and in tail.json (same estimator and draws)."""
    worst = 0.0
    for phase in PHASES:
        for s, r in old[phase]["populations"].items():
            q, mine = r["at xmin of S"], result["raw"][phase]["populations"][s]["at xmin of S"]
            worst = max(worst, abs(q["mu"] - mine["value"]), *(abs(a - b) for a, b in zip(q["band"], mine["band"])))
    for s, q in old["same route"]["mu"].items():
        mine = result["same route"]["transition"]
        mine = mine["populations"].get(s) or mine["contrasts"][s]
        worst = max(worst, abs(q["value"][-1] - mine["value"]), *(abs(q["band"][i][-1] - mine["band"][i]) for i in (0, 1)))
    return worst


if __name__ == "__main__":
    assert "--open" in sys.argv, "run with --open"
    sample = route.option("--sample", 0)
    folders()
    old = json.loads((OUT / "tail.json").read_text())
    xmins = {phase: old[phase]["xmin of S"] for phase in PHASES}
    flights = with_takeoffs(read_flights())
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    cell, pilot, block = twoway.cells(flights, summary)
    counts = twoway.counts(pilot, block, np.random.default_rng(SEED))
    p = dict(np.load(longlags.PIECES))
    wanted = set(flights["flight"])
    if sample:
        wanted = set(np.random.default_rng(1).choice(flights["flight"].to_numpy(), sample, replace=False))
    every, counted = tail.read_runs(flights)
    counted = counted.assign(last=counted["t_end"] == counted.groupby("flight")["t_end"].transform("max"))  # the last counted run
    pops = tail.population_runs(counted, p, wanted)
    cache = read_descriptors()
    result = {"n_boot": twoway.N_BOOT, "cells": int(len(pilot)), "sample": sample or None, "xmin of S": xmins,
              "stretch factor": STRETCH_FACTOR}
    result["raw"] = raw_design(pops, cell, counts, xmins, cache)
    print("raw done", flush=True)
    # same flight
    stretch = group_stretches(flights, membership, summary)
    stretch = stretch[stretch["flight"].isin(wanted)]
    runs = counted[counted["flight"].isin(stretch["flight"].unique())]
    runs = classify(stretch, runs)
    in_raw = pd.Series(None, index=runs.index, dtype=object)
    for pop, short in raw.POPS.items():
        in_raw[raw.in_pieces(runs, *raw.pieces_of(p, pop))] = short
    result["same flight"] = flight_design(runs, stretch, cell, counts, xmins, cache, in_raw)
    result["same flight"]["stretch lengths"] = length_table(stretch, xmins["transition"])
    result["same flight"]["flights"] = int(stretch["flight"].nunique())
    result["same flight"]["classes of the runs"] = runs.loc[runs["phase"] == "transition", "cls"].value_counts().to_dict()
    print("same flight done", flush=True)
    # same route
    ctx = tail.context(flights, summary, route.TRACKS, every, counted)
    table = pd.read_parquet(CACHE / "comparisons_matches.parquet")
    if sample:
        table = table[np.isin(ctx.tracks["flight"][table["piece"]], list(wanted))]
    chosen = json.loads((OUT / "matched_counts.json").read_text())["chosen caliper"]["name"]
    line = {"caliper": chosen, "cover": matched.MAIN_COVER, "isolation": False}
    result["same route"] = route_design(ctx, route_sides(ctx, table, line), xmins, cache)
    result["line"] = line
    if not sample:
        result["check against tail.json"] = {"largest difference": compare_with_tail(result, old)}
        print("largest difference with tail.json:", result["check against tail.json"])
    (OUT / f"tail_designs{'_sample' + str(sample) if sample else ''}.json").write_text(json.dumps(matched.jsonable(result), indent=1))
    for design in ("raw", "same route"):
        for s, r in result[design]["transition"]["populations"].items():
            q = r.get("at xmin of S", r)
            print(design, s, round(q["value"], 2), [round(b, 2) for b in q["band"]], q["n"], round(q["se_iid"], 2))
    for variant, v in result["same flight"]["transition"]["variants"].items():
        print(variant, {c: (round(e["value"], 2), e["n"]) for c, e in v["classes"].items()}, {k: round(e["value"], 2) for k, e in v["contrasts"].items()})
