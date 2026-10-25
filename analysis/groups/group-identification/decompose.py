"""The difference between group and solo flights taken apart: group effect, selection, day, site and hour.

Populations of tamsd.py: S solo flights, A the solo time of the group flights ("same flights
alone"), G their group time. Each contrast is a difference P - Q of one statistic:

    raw                        G - S, all flights pooled
    same site-day-hour         G - S within strata of site-day and local hour
    selection                  A - S within site-day and hour: which flights fly in groups, net
                               of the day, the site and the hour
    group, same flight         G - A within each flight: pilot, glider, logger, day and site fixed
    group, same site-day-hour  G - A within site-day and hour: the hour fixed as well

A site-day holds the flights whose takeoffs lie within 3 km on one day (blocks.sites); the hour
is the local hour (Europe/Paris) of the middle of a piece or a run. In a stratum each population
gives the statistic as the other scripts do (ensemble MSD with equal weight per flight; ratio of
sums), D_s = stat_P - stat_Q, and the strata with both populations are averaged with weights
w_s = n_P n_Q / (n_P + n_Q), n the flights of each population there (1/2 for a flight). For the
TA-MSD, D_s is the difference of log MSD at every lag; Delta H is half the slope of its average
over the lags, Delta H_loc the same within +/- 0.25 decades. Bands: 5-95% of N_BOOT draws of
blocks that join the episodes of the note and the site-days (blocks.merged), the same draws for
every statistic and contrast, so that every stratum stays whole.

    uv run python decompose.py [--run NAME] [--no-size | --only-size]    # after tamsd.py, performance.py and size.py
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget
import performance
import size
from blocks import merged, sites, with_takeoffs
from tamsd import CACHE, LABELS, LAGS, OUT, SEED, SIZES, fit_h, local_h, read_pieces, save, stretches

N_BOOT = 200
CONTRASTS = {  # name: (P, Q, stratum)
    "raw": ("group", "solo", None),
    "same site-day-hour": ("group", "solo", "site_hour"),
    "selection": ("group_solo", "solo", "site_hour"),
    "group, same flight": ("group", "group_solo", "flight"),
    "group, same site-day-hour": ("group", "group_solo", "site_hour"),
}
LABEL = {"raw": "G - S, pooled", "same site-day-hour": "G - S, same site-day-hour",
         "selection": "A - S, same site-day-hour (selection)", "group, same flight": "G - A, same flight (group)",
         "group, same site-day-hour": "G - A, same site-day-hour (group)"}
COLORS = {"raw": "#555555", "same site-day-hour": "#2a78d6", "selection": "#1baf7a", "group, same flight": "#eb6834",
          "group, same site-day-hour": "#c24a1d"}
AT = (100, 1000)  # lags of the Delta H_loc quoted, s
XC = "cross-country speed over 30 min (km/h)"
# The glide ratio (distance / height lost) has a denominator that can vanish in a small stratum (a
# glide in rising air); its inverse, the height lost per kilometre of glide, never does.
SINK = "height lost per km of glide (m/km)"
BUDGET = {k: budget.STATS[k] for k in budget.PLOTTED}
PERFORMANCE = {"climb rate (m/s)": performance.STATS["climb rate (m/s)"],
               "glide speed (km/h)": performance.STATS["glide speed (km/h)"], SINK: ("loss_m", "chord_m", 1000, 0),
               XC: performance.STATS[XC]}
DRAWN = ("H_loc(100 s)", "H_loc(1000 s)", *budget.PLOTTED, "climb rate (m/s)", "glide speed (km/h)", SINK)


class Units:
    """Rows to compare: population, stratum, block, and the values (one column per lag or per
    statistic) with, for ratios, their denominators."""

    def __init__(self, population, stratum, block, values, den=None):
        self.population, self.stratum, self.block = population, stratum, block
        self.values, self.den = values, den


def local_hour(utc):
    """Local hour (Europe/Paris) of UTC seconds."""
    return pd.to_datetime(np.asarray(utc, float), unit="s", utc=True).tz_convert("Europe/Paris").hour.to_numpy()


def strata(table, key, info):
    """Stratum of every row of `table` (flight, t_mid on the flight clock) for `key`: None (one
    stratum), "site_hour" (site-day and local hour), or a column of the table such as "flight";
    `info`: utc0 and site by flight."""
    if key is None:
        return np.zeros(len(table), np.int64)
    if key != "site_hour":
        return pd.factorize(table[key])[0]
    rows = info.reindex(table["flight"].to_numpy())
    hour = local_hour(rows["utc0"].to_numpy() + table["t_mid"].to_numpy(float))
    return rows["site"].to_numpy(np.int64) * 24 + hour


def msd_units(p, key, info, block):
    """TA-MSD units: one per (flight, stratum, population), its MSD at LAGS pooling the origins of
    its pieces there; `p` may carry extra columns (one value per piece) to stratify by."""
    table = pd.DataFrame({"flight": p["flight"], "population": p["population"], "t_mid": p["start"] + p["duration"] / 2,
                          **{k: v for k, v in p.items() if np.ndim(v) == 1 and k not in ("flight", "population")}})
    table["stratum"] = strata(table, key, info)
    code, keys = pd.factorize(pd.MultiIndex.from_frame(table[["flight", "stratum", "population"]]))
    sums, origins = np.zeros((len(keys), len(LAGS))), np.zeros((len(keys), len(LAGS)))
    np.add.at(sums, code, p["sums"])
    np.add.at(origins, code, p["origins"])
    keys = pd.DataFrame(keys.tolist(), columns=["flight", "stratum", "population"])
    return Units(keys["population"].to_numpy(), keys["stratum"].to_numpy(), block.reindex(keys["flight"]).to_numpy(),
                 sums / origins)


def ratio_units(table, key, info, block, stats):
    """Ratio units: one per (flight, stratum, population), with the summed numerators (scaled) and
    denominators of `stats` (name: num, den, scale, digits)."""
    terms = sorted({c for num, den, *_ in stats.values() for c in (num, den)})
    table = table.assign(stratum=strata(table, key, info))
    sums = table.groupby(["flight", "stratum", "population"], sort=False)[terms].sum().reset_index()
    num = np.column_stack([sums[s[0]].to_numpy(float) * s[2] for s in stats.values()])
    den = np.column_stack([sums[s[1]].to_numpy(float) for s in stats.values()])
    return Units(sums["population"].to_numpy(), sums["stratum"].to_numpy(), block.reindex(sums["flight"]).to_numpy(),
                 num, den)


def effect(units, P, Q, counts):
    """The stratified mean of stat_P - stat_Q, one value per column of the units, for the point
    estimate (every unit once) and for each row of `counts` (the count of every block in a draw)."""
    s = pd.factorize(units.stratum)[0]
    n_s = s.max() + 1
    out = []
    for d in range(len(counts) + 1):
        w = np.ones(len(s)) if d == 0 else counts[d - 1, units.block].astype(float)
        stat, n = {}, {}
        for pop in (P, Q):
            m = units.population == pop
            n[pop] = np.bincount(s[m], weights=w[m], minlength=n_s)
            num = np.column_stack([np.bincount(s[m], weights=w[m] * v, minlength=n_s) for v in units.values[m].T])
            with np.errstate(divide="ignore", invalid="ignore"):
                if units.den is None:  # the mean MSD over the flights of the stratum, in log
                    stat[pop] = np.log(num / n[pop][:, None])
                else:
                    den = np.column_stack([np.bincount(s[m], weights=w[m] * v, minlength=n_s) for v in units.den[m].T])
                    stat[pop] = num / den
        with np.errstate(divide="ignore", invalid="ignore"):
            ws = n[P] * n[Q] / (n[P] + n[Q])
        diff = stat[P] - stat[Q]
        ok = np.isfinite(diff) & (ws[:, None] > 0)
        out.append(np.where(ok, diff * ws[:, None], 0).sum(axis=0) / np.where(ok, ws[:, None], 0).sum(axis=0))
    return np.array(out)


def msd_statistics(values):
    """Delta H over all the lags and Delta H_loc at AT, from the averaged log-MSD differences."""
    k = [int(np.searchsorted(LAGS, x)) for x in AT]
    out = {"Delta H": [fit_h(LAGS, np.exp(v)) for v in values]}
    for x, i in zip(AT, k):
        out[f"H_loc({x} s)"] = [local_h(LAGS, np.exp(v))[i] for v in values]
    return out


def described(values):
    """Point estimate (the first value) and the 5-95% band of the others."""
    v = np.asarray(values, float)
    return {"value": float(v[0]), "band": np.nanpercentile(v[1:], [5, 95]).tolist()}


def decompose(pieces, runs, terms, info, block, counts, contrasts=CONTRASTS):
    """Every contrast of `contrasts` (name: P, Q, stratum) for the TA-MSD statistics and the
    ratios of BUDGET (from `runs`) and PERFORMANCE (from `terms`)."""
    out = {}
    for name, (P, Q, key) in contrasts.items():
        out[name] = {stat: described(v) for stat, v in
                     msd_statistics(effect(msd_units(pieces, key, info, block), P, Q, counts)).items()}
        for table, stats in ((runs, BUDGET), (terms, PERFORMANCE)):
            values = effect(ratio_units(table, key, info, block, stats), P, Q, counts)
            out[name].update({stat: described(values[:, j]) for j, stat in enumerate(stats)})
        print(name, {k: round(v["value"], 4) for k, v in out[name].items()}, flush=True)
    return out


def forest(result, path, stats=DRAWN, contrasts=CONTRASTS, labels=LABEL, colors=COLORS,
           title="Paragliders: the differences between group and solo flights taken apart", per_row=None):
    """One panel per statistic, in rows of `per_row` (all in one row by default): every contrast as
    a dot with its band, zero dotted."""
    ncols = per_row or len(stats)
    nrows = -(-len(stats) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols + 2.8, nrows * (0.42 * len(contrasts) + 1.3) + 0.3),
                             squeeze=False)
    for ax in axes.flat[len(stats):]:
        ax.axis("off")
    for ax, stat in zip(axes.flat, stats):
        for y, name in enumerate(contrasts):
            v = result[name][stat]
            lo, hi = v["band"]
            ax.errorbar(v["value"], y, xerr=[[v["value"] - lo], [hi - v["value"]]], fmt="o", ms=6, color=colors[name],
                        elinewidth=1.5, capsize=2)
        ax.axvline(0, color="#999999", lw=0.8, ls=":")
        ax.set_ylim(-0.6, len(contrasts) - 0.4)
        ax.invert_yaxis()
        ax.set_yticks(range(len(contrasts)), [labels[c] for c in contrasts] if ax in axes[:, 0] else [])
        ax.set_title("Δ " + stat.replace("(%)", "(points)"), fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{title} (5-95% bands)", fontsize=10)
    save(fig, path)


def size_contrasts():
    """By the number of pilots: each size against the same flights alone (same flight), and each
    size against groups of 2 pilots (same site-day-hour)."""
    return ({f"{n} - A, same flight": (n, "group_solo", "flight") for n in SIZES}
            | {f"{n} - n2, same site-day-hour": (n, "n2", "site_hour") for n in list(SIZES)[1:]})


def size_figure(result, path, stats=DRAWN, per_row=4):
    """One panel per statistic, `per_row` to a row: against the size of the group, the group
    effect within the flight (each size minus the same flights alone) and, within site-day-hour,
    each size minus 2 pilots."""
    sizes = list(SIZES)
    nrows = -(-len(stats) // per_row)
    fig, axes = plt.subplots(nrows, per_row, figsize=(3.2 * per_row + 0.6, 3.0 * nrows + 0.4), squeeze=False)
    for ax in axes.flat[len(stats):]:
        ax.axis("off")
    for ax, stat in zip(axes.flat, stats):
        for name, color, offset, label in (("{} - A, same flight", "#eb6834", -0.08, "size minus same flights alone, same flight"),
                                           ("{} - n2, same site-day-hour", "#2a78d6", 0.08, "size minus 2 pilots, same site-day-hour")):
            x = [k for k, n in enumerate(sizes) if name.format(n) in result]
            v = np.array([result[name.format(sizes[k])][stat]["value"] for k in x])
            band = np.array([result[name.format(sizes[k])][stat]["band"] for k in x])
            ax.errorbar(np.array(x) + offset, v, yerr=[v - band[:, 0], band[:, 1] - v], fmt="o-", ms=5, lw=1, color=color,
                        capsize=2, label=label)
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.set_xticks(range(len(sizes)), [n[1:] for n in sizes])
        ax.set(title="Δ " + stat.replace("(%)", "(points)"), xlabel="pilots in the group")
        ax.title.set_fontsize(8.5)
        ax.tick_params(labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Paragliders: the group effect by the number of pilots (5-95% bands)", fontsize=10)
    save(fig, path)


def flight_table(flights, summary):
    """Per flight: utc0 and site-day; the block that joins its episodes and its site-day; the
    number of such blocks."""
    site = sites(flights["lat0"], flights["lon0"], flights["date"])
    joined = merged(summary.set_index("flight")["block"].reindex(flights["flight"]).to_numpy(), site)
    info = pd.DataFrame({"utc0": flights["utc0"].to_numpy(), "site": site}, index=flights["flight"].to_numpy())
    return info, pd.Series(joined, index=flights["flight"].to_numpy()), int(joined.max()) + 1


def block_counts(n_blocks, rng):
    """N_BOOT draws of the blocks with replacement: how many times each block is drawn."""
    return rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=N_BOOT).astype(np.int32)


if __name__ == "__main__":
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary)
    info, block, n_blocks = flight_table(flights, summary)
    pieces = read_pieces(CACHE / "pieces.npz")["para"]
    runs = budget.phase_runs(flights, spans).assign(t_mid=lambda r: (r["t_start"] + r["t_end"]) / 2)
    terms = pd.read_parquet(CACHE / "performance_terms.parquet")
    counts = block_counts(n_blocks, np.random.default_rng(SEED))
    if "--only-size" not in sys.argv:
        result = decompose(pieces, runs, terms, info, block, counts)
        (OUT / "decompose.json").write_text(json.dumps(result, indent=1))
        forest(result, OUT / "decompose.pdf", per_row=4)
    if "--no-size" not in sys.argv:
        by_size = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary, by_size=True)
        runs = budget.phase_runs(flights, by_size).assign(t_mid=lambda r: (r["t_start"] + r["t_end"]) / 2)
        result = decompose(size.tamsd_pieces("para"), runs, size.terms(), info, block, counts, size_contrasts())
        (OUT / "decompose_size.json").write_text(json.dumps(result, indent=1))
        size_figure(result, OUT / "decompose_size.pdf")
