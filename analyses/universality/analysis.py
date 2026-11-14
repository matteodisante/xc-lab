"""Is H universal between paragliders and hang gliders? The curves, the exponents and their bands.

Flights: the declared open flights with an opening above openflights.MIN_OPENING (measure.py), both disciplines,
whole pieces on the 10-s grid. Terrain: terrain.py, the class of the median ground elevation under the flight;
the raw comparison takes every flight (unknown terrain included), the stratified one every class but "unknown".

Curves (equal weight per flight, 90% two-way bootstrap of pilots and site-days, shared by all populations):
- A, fixed population: a curve to the lag T uses only the pieces of at least T / SUPPORT s, so the flights do not
  change along the curve. T = TAU_FIXED (1000 s) for every cell ("fixed"), and the longest lag of LONG_LAGS at
  which para and hang of the stratum both keep MIN_FLIGHTS flights ("longest").
- B, all flights: the lag tau uses every piece of at least tau / SUPPORT s, so the flights fall with the lag. Its
  lags go up to the "longest" T of the stratum (at least TAU_FIXED), and H is fitted over the lags up to 1000 s and
  up to T.
H: half the slope of log MSD against log tau from 10 s to the longest lag of the design; alpha(tau) = 2 H_loc.
A comparison is skipped when one of its two populations has fewer than MIN_CELL flights (they stay in the counts).
Populations compared: para against hang in each terrain stratum and in each region (regions.py), and the regions
against each other within a discipline.

    uv run python analysis.py [--sample N]    # after measure.py; writes out/<full | sample N>/results.json
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import measure
import openflights
import regions
import tamsd
import terrain
import twoway

MIN_FLIGHTS = 100  # flights every cell keeps at the longest lag of design A
MIN_CELL = 30  # flights a population needs to be fitted
STRATA = ("all",) + terrain.CLASSES
DISCIPLINES = ("para", "hang")
SEED = 20261114
BIN_M, MAX_M = 50, 4800  # histogram of the median ground elevation


def selection(table, p):
    """The flights of the analysis and their pieces. F: one row per open flight (discipline, pilot, block, cell,
    class, longest piece); the pieces `owner`, `duration`, `sums`, `origins` with the row of F of each piece."""
    m = table[table["measured"]].copy()
    m = m[openflights.is_open(m["flight_type"], m["ends_km"], m["path_km"])]
    F = m.assign(cls=terrain.classify(m["median_elev"], m["n_points"], m["n_on_dem"]))
    F["pilot_code"] = np.unique(F["pilot"].to_numpy(str), return_inverse=True)[1]
    F["block"] = twoway.sites(F["lat0"], F["lon0"], F["date"])
    cell, pilot_of_cell, block_of_cell = twoway.cell_codes(F["pilot_code"].to_numpy(), F["block"].to_numpy())
    F["cell"] = cell
    F["region"] = regions.region_of(F["lat0"], F["lon0"], F["cls"])
    pos = pd.Series(np.arange(len(F)), index=F.index)
    keep = np.isin(p["row"], F.index)
    q = {"owner": pos.reindex(p["row"][keep]).to_numpy(), "duration": p["duration"][keep],
         "sums": p["sums"][keep], "origins": p["origins"][keep]}
    longest = np.zeros(len(F))
    np.maximum.at(longest, q["owner"], q["duration"])
    F["longest"] = longest
    return F.reset_index(drop=True), q, (pilot_of_cell, block_of_cell)


def funnel(table):
    """Flights per discipline at each step of the selection."""
    out = {}
    for d in DISCIPLINES:
        t = table[table["discipline"] == d]
        m = t[t["measured"]]
        o = m[openflights.is_open(m["flight_type"], m["ends_km"], m["path_km"])]
        known = terrain.classify(o["median_elev"], o["n_points"], o["n_on_dem"]) != terrain.UNKNOWN
        out[d] = {"run table": len(t), "eligible": int(t["eligible"].sum()), "declared open": int(len(m)),
                  "open": int(len(o)), "known terrain": int(known.sum()), "pilots": int(o["pilot"].nunique())}
    both = set(table.loc[table["measured"] & (table["discipline"] == "para"), "pilot"]) & \
        set(table.loc[table["measured"] & (table["discipline"] == "hang"), "pilot"])
    out["pilots in both"] = len(both)
    return out


def counts(F):
    """Flights and pilots per discipline and class (plus all and unknown), and the histogram of the median
    ground elevation per discipline."""
    out = {"cells": {}, "histogram": {"edges": np.arange(0, MAX_M + BIN_M, BIN_M).tolist(), "n": {}}}
    for d in DISCIPLINES:
        f = F[F["discipline"] == d]
        out["cells"][d] = {s: {"flights": int((f["cls"] == s).sum()), "pilots": int(f.loc[f["cls"] == s, "pilot_code"].nunique())}
                           for s in terrain.CLASSES + (terrain.UNKNOWN,)}
        out["cells"][d]["all"] = {"flights": len(f), "pilots": int(f["pilot_code"].nunique())}
        out["regions"] = out.get("regions", {})
        out["regions"][d] = {r: {"flights": int((f["region"] == r).sum()), "pilots": int(f.loc[f["region"] == r, "pilot_code"].nunique())}
                             for r in regions.BOXES}
        out["cells"][d]["median"] = float(f["median_elev"].median())
        out["histogram"]["n"][d] = np.histogram(f["median_elev"].dropna().clip(upper=MAX_M - 1), out["histogram"]["edges"])[0].tolist()
    return out


def population(F, d, stratum):
    return ((F["discipline"] == d) & ((F["cls"] == stratum) if stratum != "all" else True)).to_numpy()


def available(F, mask):
    """Flights of a population with a piece long enough for each lag of LONG_LAGS."""
    return (F.loc[mask, "longest"].to_numpy()[:, None] >= tamsd.LONG_LAGS / tamsd.SUPPORT).sum(axis=0)


def longest_lag(F, masks):
    """The longest lag of LONG_LAGS at which every population of `masks` keeps MIN_FLIGHTS flights."""
    n = np.min([available(F, m) for m in masks.values()], axis=0)
    ok = tamsd.LONG_LAGS[n >= MIN_FLIGHTS]
    return int(ok.max()) if len(ok) else None


def curve(F, q, mask, top, stacked, fixed):
    """MSD of the population `mask` over the lags up to `top`, with its bootstrap draws. `fixed`: the pieces of at
    least top / SUPPORT s at every lag (design A); else those of at least lag / SUPPORT s (design B)."""
    lags = tamsd.LONG_LAGS[tamsd.LONG_LAGS <= top]
    rows = np.flatnonzero(mask)
    local = np.full(len(F), -1)
    local[rows] = np.arange(len(rows))
    keep = mask[q["owner"]]
    need = np.full(len(lags), top / tamsd.SUPPORT) if fixed else lags / tamsd.SUPPORT
    msd = tamsd.msd_matrix(local[q["owner"][keep]], q["duration"][keep], q["sums"][keep][:, :len(lags)],
                           q["origins"][keep][:, :len(lags)], len(rows), need)
    point, draws = twoway.mean_draws(msd, F["cell"].to_numpy()[rows], stacked)
    ok = np.isfinite(msd)
    return {"lags": lags, "point": point, "draws": draws, "n": ok.sum(axis=0),
            "pilots": np.array([F["pilot_code"].to_numpy()[rows][ok[:, j]].size and len(np.unique(F["pilot_code"].to_numpy()[rows][ok[:, j]]))
                                for j in range(len(lags))])}


def summary(c, upto):
    """alpha(tau) and H of the curve `c` cut at the lag `upto`: values, draws (rows) and the counts."""
    k = c["lags"] <= upto
    lags = c["lags"][k]
    a, ad = tamsd.alpha(lags, c["point"][k]), tamsd.alpha(lags, c["draws"][:, k])
    h, hd = tamsd.fit_h(lags, c["point"][k]), tamsd.fit_h(lags, c["draws"][:, k])
    return {"lags": lags, "alpha": a, "alpha_draws": ad, "H": h, "H_draws": hd, "msd": c["point"][k], "msd_draws": c["draws"][:, k],
            "n": c["n"][k], "pilots": c["pilots"][k]}


def quote(s):
    """The JSON of a summary: values, bands and counts."""
    return {"lags": s["lags"].tolist(), "msd": s["msd"].tolist(), "msd_band": twoway.band(s["msd"], s["msd_draws"]),
            "alpha": s["alpha"].tolist(),
            "alpha_band": twoway.band(s["alpha"], s["alpha_draws"]), "H": float(s["H"]),
            "H_band": twoway.band(s["H"], s["H_draws"]), "flights": s["n"].tolist(), "pilots": s["pilots"].tolist()}


def compare(sa, sb, names):
    """The pair of summaries of one design, under `names`, and the difference first - second with its band (the
    draws are paired)."""
    da, dh = sa["alpha"] - sb["alpha"], sa["H"] - sb["H"]
    return {names[0]: quote(sa), names[1]: quote(sb), "T": int(sa["lags"][-1]),
            "d_alpha": {"value": da.tolist(), "band": twoway.band(da, sa["alpha_draws"] - sb["alpha_draws"])},
            "d_H": {"value": float(dh), "band": twoway.band(dh, sa["H_draws"] - sb["H_draws"])}}


def pair(F, q, masks, stacked):
    """All the designs for two populations (`masks`: name -> mask over the flights of F)."""
    names = tuple(masks)
    T = longest_lag(F, masks)
    out = {"flights": {k: int(m.sum()) for k, m in masks.items()}, "T_longest": T,
           "available": {k: {"lags": tamsd.LONG_LAGS.tolist(), "flights": available(F, m).tolist()} for k, m in masks.items()}}
    if min(out["flights"].values()) < MIN_CELL:
        return {**out, "skipped": f"fewer than {MIN_CELL} flights"}
    designs = {"A_fixed": tamsd.TAU_FIXED} | ({"A_longest": T} if T else {})
    for key, top in designs.items():
        c = {k: curve(F, q, m, top, stacked, True) for k, m in masks.items()}
        for k in names:
            assert len(set(c[k]["n"])) == 1, "the fixed population changes with the lag"
        out[key] = compare(*(summary(c[k], top) for k in names), names)
    top = max(T or 0, tamsd.TAU_FIXED)
    c = {k: curve(F, q, m, top, stacked, False) for k, m in masks.items()}
    out["B"] = compare(*(summary(c[k], top) for k in names), names)
    out["B_to_1000"] = compare(*(summary(c[k], tamsd.TAU_FIXED) for k in names), names)
    if T:
        out["B_to_longest"] = compare(*(summary(c[k], T) for k in names), names)
    return out


def region_mask(F, d, r):
    return ((F["discipline"] == d) & (F["region"] == r)).to_numpy()


def report(name, r):
    if "skipped" in r:
        print(name, "skipped:", r["skipped"], r["flights"], flush=True)
    else:
        print(name, r["flights"], "T", r["T_longest"], {k: round(v["d_H"]["value"], 3) for k, v in r.items()
                                                      if isinstance(v, dict) and "d_H" in v}, flush=True)


def run(path, out):
    table = pd.read_parquet(path / "flights.parquet")
    F, q, (pilot_of_cell, block_of_cell) = selection(table, dict(np.load(path / "pieces.npz")))
    stacked = twoway.counts(pilot_of_cell, block_of_cell, np.random.default_rng(SEED))
    np.savez(out / "takeoffs.npz", discipline=F["discipline"].to_numpy(str), lat=F["lat0"].to_numpy(), lon=F["lon0"].to_numpy(),
             cls=F["cls"].to_numpy(str), region=F["region"].to_numpy(str))
    result = {"funnel": funnel(table), "terrain": counts(F), "strata": {},
              "constants": {"min_flights": MIN_FLIGHTS, "min_cell": MIN_CELL, "edges_m": terrain.EDGES_M,
                            "min_on_dem": terrain.MIN_ON_DEM, "min_opening": openflights.MIN_OPENING,
                            "tau_fixed": tamsd.TAU_FIXED, "support": tamsd.SUPPORT, "n_boot": twoway.N_BOOT,
                            "site_m": twoway.SITE_M, "blocks": int(F["block"].nunique()), "cells": int(F["cell"].nunique())}}
    for name in STRATA:
        result["strata"][name] = pair(F, q, {d: population(F, d, name) for d in DISCIPLINES}, stacked)
        report(name, result["strata"][name])
    result["regions"] = {"para_vs_hang": {}, "between": {d: {} for d in DISCIPLINES}}
    for r in regions.BOXES:
        result["regions"]["para_vs_hang"][r] = pair(F, q, {d: region_mask(F, d, r) for d in DISCIPLINES}, stacked)
        report(r, result["regions"]["para_vs_hang"][r])
    for d in DISCIPLINES:
        for a, b in itertools.combinations(regions.BOXES, 2):
            result["regions"]["between"][d][f"{a} | {b}"] = pair(F, q, {a: region_mask(F, d, a), b: region_mask(F, d, b)}, stacked)
            report(f"{d}: {a} | {b}", result["regions"]["between"][d][f"{a} | {b}"])
    return result


if __name__ == "__main__":
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 0
    out = Path(__file__).with_name("out") / (f"sample{sample}" if sample else "full")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(run(measure.CACHE / (f"sample{sample}" if sample else ""), out), indent=1))
