"""The tail of the time to find a thermal, with the failed searches censored: the censored Hill exponent per population S, A, G
and ground class, and the Kaplan-Meier survival on a grid for the tail.

Every search of search_km.py is an attempt: found (event) or failed (right-censored at its duration). For a threshold u the
censored Hill estimator (Beirlant, Guillou, Dierckx and Fils-Villa 2007; Einmahl, Fils-Villa and Guillou 2008) takes the k
searches with d > u, found or failed, and
  H = (1/k) sum ln(d_i / u),   p = share of found among them,   gamma_c = H / p,   mu_c = 1 / gamma_c = p / H.
With p = 1 it is the Hill exponent of tail_ks.py. The threshold u of a class is the x_min that tail_ks.py chose for the
searches that end in a climb of that class (phases_refit.json, tails[class]["search"]["xmin"]["common"], the point
estimate); it is the same in every draw. Each search weighs the count of its cell in a draw of the shared two-way draws
(row 0, the point estimate, weighs 1), so k, p and mu_c are weighted; the bands are those of twoway.band, and the gap
mu_c,G - mu_c,A is taken draw by draw.

Written into search_km.json (to be run after search_km.py, which rewrites that file):
  "tail censored"  per class: u, and for S, A, G: k (searches above u), p, mu_c; the gap mu_c,G - mu_c,A and A - S;
  "km tail"        S_km of S, A, G on every flight on TAIL_POINTS log-spaced seconds from TAIL_FROM to the largest search
                   duration, with bands, and "at risk": per population and grid point, the number of searches with a
                   duration >= t in the point estimate.

    uv run python tail_censored.py --declared-open --dt5 [--draws N]
"""

import json

import numpy as np

import longlags
import phase_gaps
import phase_terms
import search_km
from phase_gaps import CLASS_KEYS, POPS, rows_of, stratum
from search_km import GAPS, attempts, curve_rows, curves

TAIL_FROM, TAIL_POINTS = 10, 70  # s, number of grid points up to the largest search duration


def censored_hill(d, event, cell, weights, u):
    """k, p and mu_c of the searches (durations `d`, event, cell code `cell`) above `u`, one value per row of `weights`
    (rows, cells): three arrays over the rows."""
    above = d > u
    n = weights.shape[1]
    k = weights @ np.bincount(cell[above], minlength=n)
    found = weights @ np.bincount(cell[above], event[above] * 1.0, minlength=n)
    logs = weights @ np.bincount(cell[above], np.log(d[above] / u), minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, hill = found / k, logs / k
        return k, p, p / hill


def statistics(runs, code, counts, xmin):
    """Per class: u, k, p and mu_c of S, A, G and the gaps (G - A, A - S) of mu_c, with bands. `xmin`: {class: u}."""
    w = np.vstack([np.ones((1, counts.shape[1])), counts]).astype(float)
    out = {}
    for key in CLASS_KEYS:
        mine = stratum(runs, key)
        data = {POPS[k]: attempts(mine[mine["pop"] == i]) for i, k in enumerate(POPS)}
        got = {}
        for pop, (d, event, flight) in data.items():
            k, p, mu = censored_hill(d, event, code.reindex(flight).to_numpy(), w, xmin[key])
            got[pop] = {"k": k, "p": p, "mu": mu}
        out[key] = {"u": xmin[key],
                    **{pop: {"k": int(g["k"][0]), "p": float(g["p"][0]), "mu_c": rows_of(g["mu"])} for pop, g in got.items()},
                    "gaps": {gap: rows_of(got[a]["mu"] - got[b]["mu"]) for gap, (a, b) in GAPS.items()}}
        print(key, round(xmin[key], 1), {pop: out[key][pop]["k"] for pop in data}, flush=True)
    return out


def survival(runs, code, counts):
    """S_km of S, A, G on every flight on the log grid of the module docstring, with bands, and the number at risk."""
    data = {POPS[k]: attempts(runs[runs["pop"] == i]) for i, k in enumerate(POPS)}
    top = max(d.max() for d, _, _ in data.values())
    grid = np.unique(np.rint(np.geomspace(TAIL_FROM, top, TAIL_POINTS)).astype(int))
    w = np.vstack([np.ones((1, counts.shape[1])), counts])
    out = {pop: curve_rows(curves(d, event, code.reindex(flight).to_numpy(), w, grid)["S_km"]) for pop, (d, event, flight) in data.items()}
    risk = {pop: [int(np.sum(w[0][code.reindex(flight).to_numpy()][d >= t])) for t in grid] for pop, (d, _, flight) in data.items()}
    return {"grid": grid.tolist(), "survival": out, "at risk": risk}


if __name__ == "__main__":
    refit = json.loads((longlags.RESULTS / "phases_refit.json").read_text())["tails"]
    xmin = {c: refit[c]["search"]["xmin"]["common"] for c in CLASS_KEYS}
    p, code, counts, classes = phase_gaps.context()
    runs = phase_gaps.assign(phase_terms.table(phase_terms.SET, phase_terms.kept_flights()), p, classes)
    path = longlags.RESULTS / f"{search_km.NAME}.json"
    result = json.loads(path.read_text())
    result["tail censored"] = statistics(runs, code, counts, xmin)
    result["km tail"] = survival(runs, code, counts)
    path.write_text(json.dumps(result, indent=1))
    print(f"{'class':15s}{'u (s)':>7s}  " + "".join(f"{x:>22s}" for x in ("k_A  k_G", "p_A    p_G", "mu_c A", "mu_c G", "mu_c G - A")))
    for c in CLASS_KEYS:
        r = result["tail censored"][c]
        q = lambda v: f"{v['value']:.3f} [{v['band'][0]:.3f}, {v['band'][1]:.3f}]"
        print(f"{c:15s}{r['u']:7.0f}  {r['A']['k']:>9d}{r['G']['k']:>9d}   {r['A']['p']:.3f} {r['G']['p']:.3f}   "
              f"{q(r['A']['mu_c'])}  {q(r['G']['mu_c'])}  {q(r['gaps']['G - A'])}")
