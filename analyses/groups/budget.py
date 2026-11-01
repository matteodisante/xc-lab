"""How paragliders share their time among the phases, solo and in a group.

Vilpellet's phase runs, each whole in the population of the stretch that holds its middle
(survival.assign). Per population, ratios summed over its runs: the share of time in
transition, in climb, in a search that ends in a climb and in one that does not (a search in
vain); climbs per hour; the share of climbs that start after a search; the share of searches
that end in a climb. The run before and after are read on the whole flight. Bands: 5-95% of
N_BOOT draws of the dependency blocks, each summing numerator and denominator over its blocks.

    uv run python budget.py [--run NAME]    # from this folder; writes out/<run>/budget.*
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_msd import RUNS
from survival import assign
from tamsd import COLORS, LABELS, N_BOOT, OUT, POPULATIONS, SEED, read_flights, save, stretches

# name: numerator, denominator (columns of `columns`), scale, decimals shown
STATS = {
    "time in transition (%)": ("transition_s", "s", 100, 1),
    "time in climb (%)": ("climb_s", "s", 100, 1),
    "time in a search that ends in a climb (%)": ("found_s", "s", 100, 1),
    "time in a search in vain (%)": ("vain_s", "s", 100, 1),
    "climbs per hour": ("climbs", "hours", 1, 2),
    "climbs that start after a search (%)": ("after_search", "climbs", 100, 1),
    "searches that end in a climb (%)": ("found", "searches", 100, 1),
}
# The statistics drawn: the others are explained by them and stay in the JSON and the tables
# (redundancy.py). The four shares of time sum to 100%, and "climbs that start after a search"
# counts the same search-then-climb events as "searches that end in a climb".
PLOTTED = ("time in climb (%)", "searches that end in a climb (%)")


def columns(runs):
    """Per run the terms of STATS; runs sorted within each flight, neighbours on the whole flight."""
    runs = runs.sort_values(["flight_id", "segment_id", "t_start"], kind="stable")
    phase = runs.groupby("flight_id", sort=False)["phase"]
    before = phase.shift(1).eq("search").fillna(False).to_numpy(bool)
    after = phase.shift(-1).eq("climb").fillna(False).to_numpy(bool)
    kind = runs["phase"].to_numpy(object)
    s = runs["n_fixes"].to_numpy(float)
    search, climb = kind == "search", kind == "climb"
    return runs.assign(s=s, hours=s / 3600, transition_s=s * (kind == "transition"), climb_s=s * climb,
                       found_s=s * (search & after), vain_s=s * (search & ~after), climbs=climb * 1.0,
                       after_search=(climb & before) * 1.0, found=(search & after) * 1.0, searches=search * 1.0)


def phase_runs(flights, spans):
    """Vilpellet's runs of the paragliders with the terms of STATS, each in the population of the
    stretch of `spans` that holds its middle."""
    numbers = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]]
    return assign(columns(pd.read_parquet(RUNS["para"]).merge(numbers, on="flight_id")), spans)


def ratios(sums, stats=STATS):
    """`stats` from summed columns (a Series, or a frame of draws)."""
    return {name: scale * sums[num] / sums[den] for name, (num, den, scale, _) in stats.items()}


def unit_rates(rows, unit, time, stats=STATS):
    """Average first, ratio after: per unit u of `time` (T_u, a Series by unit), the rates r_u = n_u / T_u
    and q_u = d_u / T_u of every statistic of `stats` (name: numerator, denominator, scale, ...), with n_u and
    d_u the columns summed over the `rows` of the unit (`unit`: the unit of each row; a unit with no row
    gets 0). Returns the frames (r, q), one column per statistic."""
    terms = sorted({c for num, den, *_ in stats.values() for c in (num, den)})
    sums = rows[terms].groupby(np.asarray(unit)).sum().reindex(time.index, fill_value=0.0)
    r = pd.DataFrame({name: sums[num] / time for name, (num, *_) in stats.items()})
    q = pd.DataFrame({name: sums[den] / time for name, (_, den, *_) in stats.items()})
    return r, q


def mean_ratios(r, q, block, counts, stats=STATS):
    """stat = scale * sum_u w_u r_u / sum_u w_u q_u for the frames of unit_rates: all weights 1 (the value, a
    Series) and the weights of each draw, the count of the block of the unit (`block`: a code per unit, the
    columns of `counts`, N_BOOT rows): the draws, an array (N_BOOT, statistics)."""
    codes, n = np.asarray(block), counts.shape[1]
    by_block = [np.zeros((n, len(stats))) for _ in range(2)]
    for total, frame in zip(by_block, (r, q)):
        np.add.at(total, codes, frame.to_numpy(float))
    scale = np.array([s[2] for s in stats.values()], float)
    value = pd.Series(scale * by_block[0].sum(axis=0) / by_block[1].sum(axis=0), index=list(stats))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = [scale * (counts[a:a + 100] @ by_block[0]) / (counts[a:a + 100] @ by_block[1])
                 for a in range(0, len(counts), 100)]
    return value, np.concatenate(draws)


def budget(runs, block, rng, stats=STATS, populations=POPULATIONS):
    """Per population of `populations` (default the three of tamsd.py) in `runs` (rows with
    flight, population and the terms of `stats`): flights, `stats` and their bands."""
    terms = sorted({c for num, den, *_ in stats.values() for c in (num, den)})
    out = {}
    for pop in populations:
        r = runs[runs["population"] == pop]
        by_block = r.groupby(block.reindex(r["flight"]).to_numpy())[terms].sum()
        draws = []
        for start in range(0, N_BOOT, 50):
            w = rng.multinomial(len(by_block), np.full(len(by_block), 1 / len(by_block)), size=min(50, N_BOOT - start))
            draws.append(pd.DataFrame(w @ by_block.to_numpy(), columns=terms))
        draws = ratios(pd.concat(draws, ignore_index=True), stats)
        value = ratios(by_block.sum(), stats)
        out[pop] = {"flights": int(r["flight"].nunique()),
                    **{name: {"value": float(value[name]), "band": np.percentile(draws[name], [5, 95]).tolist()}
                       for name in stats}}
    return out


def plot(result, path, stats=STATS, title="Paragliders: how the time is spent, solo and in a group",
         populations=POPULATIONS, colors=COLORS):
    """One panel per statistic of `stats` (a row of up to three, else two rows), one line per
    population of `populations` (key: label), coloured by `colors`."""
    rows = 1 if len(stats) <= 3 else 2
    columns = -(-len(stats) // rows)
    pops = list(populations)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns + 1.6, rows * (0.9 + 0.32 * len(pops)) + 0.6),
                             squeeze=False)
    for ax, name in zip(axes.flat, stats):
        for y, pop in enumerate(pops):
            v = result[pop][name]
            lo, hi = v["band"]
            ax.errorbar(v["value"], y, xerr=[[v["value"] - lo], [hi - v["value"]]], fmt="o", ms=7,
                        color=colors[pop], elinewidth=1.5, capsize=3)
            ax.annotate(f"{v['value']:.{stats[name][3]}f}", (hi, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#333333")
        lows = [result[p][name]["band"][0] for p in pops]
        highs = [result[p][name]["band"][1] for p in pops]
        pad = 0.25 * (max(highs) - min(lows))
        ax.set_xlim(min(lows) - pad, max(highs) + 2 * pad)
        ax.set_ylim(-0.7, len(pops) - 0.3)
        ax.invert_yaxis()
        ax.set_yticks(range(len(pops)), [populations[p] for p in pops] if ax in axes[:, 0] else [])
        ax.set_title(name, fontsize=9)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(stats):]:
        ax.axis("off")
    fig.suptitle(f"{title} (5-95% bands)", fontsize=10)
    save(fig, path)


if __name__ == "__main__":
    flights = read_flights()
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary)
    runs = phase_runs(flights, spans)
    result = budget(runs, summary.set_index("flight")["block"], np.random.default_rng(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "budget.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "budget.pdf", {name: STATS[name] for name in PLOTTED})
    for pop, r in result.items():
        print(pop, r["flights"], " | ".join(f"{n}: {r[n]['value']:.2f} {np.round(r[n]['band'], 2).tolist()}" for n in STATS))
