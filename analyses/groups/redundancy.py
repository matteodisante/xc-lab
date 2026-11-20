"""Which of the statistics of budget.py and performance.py tell the same thing.

Unit: one population (solo, group time, solo time of the group flights) on one site-day with at
least MIN_FLIGHTS flights in both tables; each statistic is the ratio of sums of the unit, as in
the figures. A statistic is redundant when the others explain most of it across the units: R2 of
its linear regression on them, all standardised. Backward elimination drops the statistic most
explained by the others until every R2 is at most LIMIT (the four shares of time sum to 100%
exactly, so one of them always goes first, the first of STATS on a tie). The statistics left are
those drawn (budget.PLOTTED, performance.PLOTTED); the others stay in the JSON and the tables.

    uv run python redundancy.py [--run NAME]    # from this folder, after budget.py and performance.py

With --phases, the same on the statistics of phase_gaps.py (the S, A and G runs of the label set `--labels`, the ground
classes pooled), written to phases_redundancy_<labels>.json in longlags.RESULTS; the statistics left are those drawn in
phases_composition.pdf and phases_performance.pdf.

    uv run python redundancy.py --phases --declared-open --dt5 [--labels NAME]
"""

import json
import sys

import numpy as np
import pandas as pd

import budget
import performance
from blocks import sites, with_takeoffs
from tamsd import CACHE, LABELS, OUT, read_flights, stretches

MIN_FLIGHTS, LIMIT = 25, 0.5
PHASES_MIN_FLIGHTS = 10  # the declared open flights up to 5 s are fewer per site-day: 25 leaves 6 units, 10 leaves 236
STATS = {**budget.STATS, **performance.STATS}


def unit_stats(table, stats, site, minimum=MIN_FLIGHTS):
    """The `stats` of `table` (rows with flight, population and their terms) for every
    (site-day, population) with at least `minimum` flights; `site` maps flights to site-days."""
    table = table.assign(site=site.reindex(table["flight"]).to_numpy()).dropna(subset=["site"])
    keys = ["site", "population"]
    flights = table.groupby(keys)["flight"].nunique()
    out = pd.DataFrame(budget.ratios(table.groupby(keys).sum(numeric_only=True), stats))
    return out[flights.reindex(out.index) >= minimum]


def r2(z, target, others):
    """R2 of the linear regression of the column `target` of z on the columns `others`."""
    x, y = z[others].to_numpy(), z[target].to_numpy()
    return 1 - (y - x @ np.linalg.lstsq(x, y, rcond=None)[0]).var() / y.var()


def eliminate(units, limit=LIMIT):
    """The statistics (columns of `units`) left once the one most explained by the others has
    been dropped, over and over, until every R2 is at most `limit`; the standardised units; and
    the statistics dropped, in order, with their R2 at the time."""
    z = (units - units.mean()) / units.std()
    kept, dropped = list(z.columns), []
    while len(kept) > 2:
        r = {c: round(r2(z, c, [k for k in kept if k != c]), 6) for c in kept}
        worst = max(r, key=r.get)
        if r[worst] <= limit:
            break
        dropped.append((worst, r[worst]))
        kept.remove(worst)
    return kept, z, dropped


def report(units):
    """The result of eliminate() on `units` as a dict: the units, the statistics kept and dropped (in order, with their
    R2 at the time), and the R2 of each statistic on all the others."""
    kept, z, dropped = eliminate(units)
    return {"units": len(units), "populations": units.index.get_level_values("population").value_counts().to_dict(),
            "kept": kept, "dropped": [{"statistic": c, "r2_at_the_time": r} for c, r in dropped],
            "r2_on_all_others": {c: r2(z, c, [k for k in z.columns if k != c]) for c in z.columns},
            "r2_of_the_dropped_on_the_kept": {c: r2(z, c, kept) for c in units.columns if c not in kept},
            "r2_of_the_kept_on_the_kept": {c: r2(z, c, [k for k in kept if k != c]) for c in kept}}


def phases():
    """The statistics of phase_gaps.py on site-days (unit_stats), both label sets' own runs."""
    import longlags
    import phase_gaps
    import phase_terms

    flights = with_takeoffs(read_flights())
    p, _, _, classes = phase_gaps.context()
    runs = phase_gaps.assign(phase_terms.table(phase_gaps.LABELS_NAME, phase_terms.kept_flights()), p, classes)
    runs = runs.assign(population=np.array(list(phase_gaps.POPS))[runs["pop"].to_numpy()]).drop(columns=["pop", "cls", "phase"])
    site = pd.Series(sites(flights["lat0"], flights["lon0"], flights["date"]), index=flights["flight"])
    units = unit_stats(runs, phase_gaps.STATS, site, PHASES_MIN_FLIGHTS)
    out = report(units) | {"labels": phase_gaps.LABELS_NAME, "min_flights": PHASES_MIN_FLIGHTS, "limit": LIMIT}
    (longlags.RESULTS / f"phases_redundancy_{phase_gaps.LABELS_NAME}{phase_gaps.PLAIN}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("units", "populations", "kept", "dropped")}, indent=1))


if __name__ == "__main__":
    if "--phases" in sys.argv:
        phases()
        sys.exit()
    flights = with_takeoffs(read_flights())
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary)
    site = pd.Series(sites(flights["lat0"], flights["lon0"], flights["date"]), index=flights["flight"])
    units = unit_stats(budget.phase_runs(flights, spans), budget.STATS, site).join(
        unit_stats(pd.read_parquet(CACHE / "performance_terms.parquet"), performance.STATS, site), how="inner")
    kept, z, dropped = eliminate(units)
    out = {"units": len(units), "populations": units.index.get_level_values("population").value_counts().to_dict(),
           "kept": kept, "dropped": [{"statistic": c, "r2_at_the_time": r} for c, r in dropped],
           "kept_by_population": {pop: eliminate(units.xs(pop, level="population"))[0]
                                  for pop in ("group", "group_solo")},  # the 11 solo units are too few
           "r2_of_the_dropped_on_the_kept": {c: r2(z, c, kept) for c in units.columns if c not in kept},
           "r2_of_the_kept_on_the_kept": {c: r2(z, c, [k for k in kept if k != c]) for c in kept}}
    (OUT / "redundancy.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
