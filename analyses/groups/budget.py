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
from tamsd import COLORS, LABELS, N_BOOT, OUT, POPULATIONS, SEED, save, stretches

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


def ratios(sums, stats=STATS):
    """`stats` from summed columns (a Series, or a frame of draws)."""
    return {name: scale * sums[num] / sums[den] for name, (num, den, scale, _) in stats.items()}


def budget(runs, block, rng, stats=STATS):
    """Per population of `runs` (rows with flight, population and the terms of `stats`):
    flights, `stats` and their bands."""
    terms = sorted({c for num, den, *_ in stats.values() for c in (num, den)})
    out = {}
    for pop in POPULATIONS:
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


def plot(result, path, stats=STATS, title="Paragliders: how the time is spent, solo and in a group"):
    columns = -(-len(stats) // 2)
    fig, axes = plt.subplots(2, columns, figsize=(3 * columns, 4.6), squeeze=False)
    pops = list(POPULATIONS)
    for ax, name in zip(axes.flat, stats):
        for y, pop in enumerate(pops):
            v = result[pop][name]
            lo, hi = v["band"]
            ax.errorbar(v["value"], y, xerr=[[v["value"] - lo], [hi - v["value"]]], fmt="o", ms=7,
                        color=COLORS[pop], elinewidth=1.5, capsize=3)
            ax.annotate(f"{v['value']:.{stats[name][3]}f}", (hi, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#333333")
        lows = [result[p][name]["band"][0] for p in pops]
        highs = [result[p][name]["band"][1] for p in pops]
        pad = 0.25 * (max(highs) - min(lows))
        ax.set_xlim(min(lows) - pad, max(highs) + 2 * pad)
        ax.set_ylim(-0.7, len(pops) - 0.3)
        ax.invert_yaxis()
        ax.set_yticks(range(len(pops)), [POPULATIONS[p] for p in pops] if ax in axes[:, 0] else [])
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
    flights = pd.read_parquet(LABELS / "flights.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary)
    numbers = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]]
    runs = assign(columns(pd.read_parquet(RUNS["para"]).merge(numbers, on="flight_id")), spans)
    result = budget(runs, summary.set_index("flight")["block"], np.random.default_rng(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "budget.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "budget.pdf")
    for pop, r in result.items():
        print(pop, r["flights"], " | ".join(f"{n}: {r[n]['value']:.2f} {np.round(r[n]['band'], 2).tolist()}" for n in STATS))
