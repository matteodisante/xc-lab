"""Which of the statistics of budget.py and performance.py tell the same thing.

Unit: one population (solo, group time, solo time of the group flights) on one site-day with at
least MIN_FLIGHTS flights in both tables; each statistic is the ratio of sums of the unit, as in
the figures. A statistic is redundant when the others explain most of it across the units: R2 of
its linear regression on them, all standardised. Backward elimination drops the statistic most
explained by the others until every R2 is at most LIMIT (the four shares of time sum to 100%
exactly, so one of them always goes first, the first of STATS on a tie). The statistics left are
those drawn (budget.PLOTTED, performance.PLOTTED); the others stay in the JSON and the tables.

    uv run python redundancy.py [--run NAME]    # from this folder, after budget.py and performance.py
"""

import json

import numpy as np
import pandas as pd

import budget
import performance
from blocks import sites, with_takeoffs
from tamsd import CACHE, LABELS, OUT, stretches

MIN_FLIGHTS, LIMIT = 25, 0.5
STATS = {**budget.STATS, **performance.STATS}


def unit_stats(table, stats, site):
    """The `stats` of `table` (rows with flight, population and their terms) for every
    (site-day, population) with at least MIN_FLIGHTS flights; `site` maps flights to site-days."""
    table = table.assign(site=site.reindex(table["flight"]).to_numpy()).dropna(subset=["site"])
    keys = ["site", "population"]
    flights = table.groupby(keys)["flight"].nunique()
    out = pd.DataFrame(budget.ratios(table.groupby(keys).sum(numeric_only=True), stats))
    return out[flights.reindex(out.index) >= MIN_FLIGHTS]


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


if __name__ == "__main__":
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
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
