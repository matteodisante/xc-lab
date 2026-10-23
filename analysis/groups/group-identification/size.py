"""Group flights by the number of pilots in their configuration, pooled and on chosen site-days.

The group time of tamsd.py is cut where the configuration changes (tamsd.stretches by_size) and
each stretch goes to the key of SIZES of its configuration: 2, 3, 4, 5-6 and 7+ pilots. A flight
can have stretches of several sizes, one population each, with the equal weight per flight of the
other scripts. The solo and the solo-time stretches are those of tamsd.py, phase_msd.py and
performance.py, so only the group stretches are measured again (SIZE_CACHE, beside their caches).

Populations drawn: solo flights, the sizes and, in the pooled figures, "same flights alone": the
solo stretches of the flights that have the size, the control of the other scripts. A population
enters only with MIN_FLIGHTS flights in at least MIN_BLOCKS blocks of the note.

Site-days (flights whose takeoffs lie within 3 km on one day, blocks.sites) hold the day and the
site fixed while the size changes. SITEDAYS of them are drawn: the ones with sizes of at least
MIN_FLIGHTS flights in most sizes, then most flights, at most PER_SITE per takeoff site; or the
ones given by --sitedays "Name:YYYY-MM-DD,...". The curves and ratios are redone on each, with
the bands of the blocks of the note; the survival of the phase durations needs the pooled runs.

    uv run python size.py [--run NAME] [--recompute | --recompute-pieces] [--sitedays "Name:YYYY-MM-DD,..."]
    # from this folder, after the other scripts; writes out/<run>/size_* and siteday_*
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget
import performance
import phase_msd
import survival
import tamsd
from blocks import sites, with_takeoffs
from phase_msd import PHASES, RUNS
from tamsd import CACHE, COLORS, LABELS, NAMES, OUT, SEED, SIZES, read_pieces, save, stretches

SIZE_CACHE = {"pieces": CACHE / "size_pieces.npz", "phase": CACHE / "size_phase_pieces.npz",
              "terms": CACHE / "size_performance_terms.parquet"}
MIN_FLIGHTS, MIN_BLOCKS = 15, 5
MIN_TAIL = 50  # runs above the common threshold for a mu to be shown
SITEDAYS, PER_SITE = 6, 2
RAMP = ("#f7c59f", "#f2a065", "#eb6834", "#c24a1d", "#8c2f0f")  # orange, light to dark: more pilots
SIZE_COLORS = {"solo": COLORS["solo"], **dict(zip(SIZES, RAMP))}
MARKERS = "osv^DP"
GREY = "#8a8a86"


def measure_sizes(flights, membership, summary, pieces_only=False):
    """The three passes over the fixes of the group stretches by size, saved in SIZE_CACHE (only the
    TA-MSD pieces with `pieces_only`)."""
    spans = stretches(flights, membership, summary, by_size=True)
    group = spans[spans["population"].isin(SIZES)]
    tamsd.save_pieces(SIZE_CACHE["pieces"], flights, group)
    if not pieces_only:
        phase_msd.save_pieces(SIZE_CACHE["phase"], flights, group)
        performance.save_terms(SIZE_CACHE["terms"], flights, group)


def tamsd_pieces(discipline):
    """The TA-MSD pieces of one discipline: solo and solo-time of tamsd.py, the sizes of SIZE_CACHE."""
    old, new = read_pieces(CACHE / "pieces.npz")[discipline], read_pieces(SIZE_CACHE["pieces"])[discipline]
    keep = old["population"] != "group"
    if not len(new["flight"]):  # no group stretch of this discipline
        return {k: v[keep] for k, v in old.items()}
    return {k: np.concatenate([v[keep], new[k]]) for k, v in old.items()}


def phase_pieces(discipline):
    """The phase pieces of one discipline and their MSDs: those of phase_msd.py outside the group, and the sizes."""
    (old, v_old), (new, v_new) = (phase_msd.read_pieces(path, discipline) for path in
                                  (CACHE / "phase_pieces.npz", SIZE_CACHE["phase"]))
    keep = (old["population"] != "group").to_numpy()
    return pd.concat([old[keep], new], ignore_index=True), np.concatenate([v_old[keep], v_new])


def terms():
    """The terms of performance.STATS: those of performance.py outside the group, and the sizes."""
    old = pd.read_parquet(CACHE / "performance_terms.parquet")
    return pd.concat([old[old["population"] != "group"], pd.read_parquet(SIZE_CACHE["terms"])], ignore_index=True)


def labelled(controls):
    """The populations drawn, key: label, and their colours; with `controls`, also each size's
    "same flights alone" (key "<size>/alone", the colour of the size)."""
    pops, colors = {"solo": "solo flights", **SIZES}, dict(SIZE_COLORS)
    if controls:
        for s in SIZES:
            pops[f"{s}/alone"] = f"{SIZES[s]}, same flights alone"
            colors[f"{s}/alone"] = colors[s]
    return pops, colors


def select(population, flight, block, flights=None, controls=True):
    """The rows of a table (arrays over its rows) to analyse, as row indices and population labels.

    Rows of `flights` only (all when None). With `controls`, the solo-time rows of the flights
    that have each size are added again with the label "<size>/alone". The group flights' own
    solo time is dropped, and so is every population with fewer than MIN_FLIGHTS flights or
    MIN_BLOCKS blocks (`block`: Series flight -> block)."""
    rows = np.arange(len(population)) if flights is None else np.flatnonzero(np.isin(flight, flights))
    labels = population[rows]
    alone, more, names = rows[labels == "group_solo"], [], []
    for s in SIZES if controls else ():
        mine = np.isin(flight[alone], np.unique(flight[rows[labels == s]]))
        more.append(alone[mine])
        names.append(np.full(mine.sum(), f"{s}/alone"))
    rows, labels = np.concatenate([rows, *more]), np.concatenate([labels, *names])
    frame =pd.DataFrame({"label": labels, "flight": flight[rows], "block": block.reindex(flight[rows]).to_numpy()})
    count = frame.groupby("label").agg(flights=("flight", "nunique"), blocks=("block", "nunique"))
    enough = count.index[(count["flights"] >= MIN_FLIGHTS) & (count["blocks"] >= MIN_BLOCKS)]
    keep = np.isin(labels, enough) & (labels != "group_solo")
    return rows[keep], labels[keep]


def present(pops, labels):
    """The populations of `pops` that have rows among `labels`, in order."""
    return {k: v for k, v in pops.items() if k in set(labels)}


def choose_sitedays(sized, flights, site, spec=None):
    """The site-days drawn, each as (label, flights): those of `spec` ("Name:YYYY-MM-DD" items,
    the takeoff name as the catalog writes it), else the SITEDAYS best by the number of sizes with
    MIN_FLIGHTS flights (at least two), then by flights, with at most PER_SITE per takeoff name.
    `sized`: the TA-MSD pieces of the paragliders by size; `flights`: the flight table with the
    catalog name of the takeoff site; `site`: Series flight -> site-day."""
    info = flights.assign(site=site.reindex(flights["flight"]).to_numpy()).groupby("site").agg(
        date=("date", "first"), name=("site_name", lambda s: s.mode().iat[0] if s.notna().any() else "?"))
    seen = pd.DataFrame({"flight": sized["flight"], "size": sized["population"]}).drop_duplicates()
    seen["site"] = site.reindex(seen["flight"]).to_numpy()
    counts = seen.groupby(["site", "size"])["flight"].nunique().unstack(fill_value=0)
    table = info.join(pd.DataFrame({"big": (counts >= MIN_FLIGHTS).sum(axis=1), "total": counts.sum(axis=1)}),
                      how="inner")
    if spec:
        chosen = []
        for n, d in (item.rsplit(":", 1) for item in spec):
            found = table.index[(table["name"] == n) & (table["date"] == d)]
            if not len(found):
                raise SystemExit(f"no site-day {n} on {d} with group flights; choices are:\n{table.sort_values('total')}")
            chosen.append(found[0])
    else:
        chosen, per = [], {}
        for s, row in table[table["big"] >= 2].sort_values(["big", "total"], ascending=False).iterrows():
            if per.get(row["name"], 0) < PER_SITE and len(chosen) < SITEDAYS:
                chosen.append(s)
                per[row["name"]] = per.get(row["name"], 0) + 1
    return [(f"{table.loc[s, 'name']}, {table.loc[s, 'date']}", flights.loc[site.reindex(flights["flight"]).to_numpy() == s,
                                                                         "flight"].to_numpy()) for s in chosen]


def tamsd_results(p, summary, rng, flights, pops):
    """TA-MSD of `p` (dict of arrays) for the populations `pops` on `flights`."""
    block = summary.set_index("flight")["block"]
    rows, labels = select(p["population"], p["flight"], block, flights, controls=any("/" in k for k in pops))
    q = {k: v[rows] for k, v in p.items()} | {"population": labels}
    return tamsd.summarize(q, summary, rng, present(pops, labels))


def phase_results(pieces, values, block, rng, flights, pops, slopes=False):
    """MSD per phase (and, with `slopes`, their fits with bands) for `pops` on `flights`."""
    rows, labels = select(pieces["population"].to_numpy(), pieces["flight"].to_numpy(), block, flights,
                          controls=any("/" in k for k in pops))
    q, v, here = pieces.iloc[rows].assign(population=labels).reset_index(drop=True), values[rows], present(pops, labels)
    results = phase_msd.medians(q, v, populations=here)
    if not slopes:
        return results, None
    lags = {pop: {ph: phase_msd.drawn(results[pop][ph])[0] for ph in phase_msd.FITS} for pop in here}
    return results, {"slope": phase_msd.fitted(results, lags),
                     "band": phase_msd.bands(q, v, block.reindex(q["flight"]).to_numpy(), lags, rng, here)}


def ratio_results(table, block, rng, flights, stats, pops):
    """`stats` per population of `pops` for the rows of `table` (a frame with population, flight, terms)."""
    rows, labels = select(table["population"].to_numpy(), table["flight"].to_numpy(), block, flights,
                          controls=any("/" in k for k in pops))
    return budget.budget(table.iloc[rows].assign(population=labels), block, rng, stats, present(pops, labels))


def draw_curves(axes, results, pops, colors):
    """MSD on axes[0] and H_loc on axes[1] of every population of `results`: bands for solo and the
    sizes, dashes for "same flights alone"."""
    for pop, r in results.items():
        alone, lags = pop.endswith("/alone"), np.array(r["lags"])
        style = dict(color=colors[pop], lw=1, ls="--" if alone else "-", marker="" if alone else "o", ms=3)
        if not alone:
            for ax, band in zip(axes, (r["msd_band"], r["H_loc_band"])):
                ax.fill_between(lags, *np.array(band), color=colors[pop], alpha=0.2, lw=0)
        axes[0].plot(lags, r["msd"], label=None if alone else f"{pops[pop]} ({r['flights']}): H = {r['H']:.3f}", **style)
        axes[1].plot(lags, r["H_loc"], **style)
    if any(pop.endswith("/alone") for pop in results):
        axes[0].plot([], [], color="#555555", ls="--", lw=1, label="dashed: same flights alone")
    axes[0].set(xscale="log", yscale="log")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    for y in (0.5, 1):
        axes[1].axhline(y, color="#999999", lw=0.6, ls=":")
    axes[1].set(xscale="log", xlabel=r"$\tau$ (s)")


def curves_figure(columns, path, pops, colors, per_row=None):
    """The MSD above the H_loc of each (title, results) of `columns`, side by side or, with
    `per_row`, in rows of that many; the MSDs share one scale and so do the H_loc."""
    ncols = per_row or len(columns)
    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(4.4 * ncols, 7 * nrows), sharex=True, squeeze=False)
    for k, (title, results) in enumerate(columns):
        row, col = divmod(k, ncols)
        pair = axes[2 * row:2 * row + 2, col]
        draw_curves(pair, results, pops, colors)
        pair[0].set_title(title)
        pair[0].sharey(axes[0, 0])
        pair[1].sharey(axes[1, 0])
    for k in range(len(columns), nrows * ncols):
        for ax in axes[2 * (k // ncols):2 * (k // ncols) + 2, k % ncols]:
            ax.axis("off")
    for row in range(nrows):
        axes[2 * row, 0].set_ylabel(r"MSD (m$^2$)")
        axes[2 * row + 1, 0].set_ylabel(r"$H_{\rm loc}(\tau)$")
    save(fig, path)


def hloc_figure(results, path, pops, colors):
    """H_loc(tau) of the paragliders by the size of the configuration (bands for solo and the sizes),
    the same flights alone dashed."""
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for pop, r in results.items():
        alone, lags = pop.endswith("/alone"), np.array(r["lags"])
        if not alone:
            ax.fill_between(lags, *np.array(r["H_loc_band"]), color=colors[pop], alpha=0.2, lw=0)
        ax.plot(lags, r["H_loc"], color=colors[pop], ls="--" if alone else "-", marker="" if alone else "o", ms=3, lw=1.2,
                label=None if alone else f"{pops[pop]}: H = {r['H']:.3f}")
    ax.plot([], [], color="#555555", ls="--", lw=1, label="dashed: the same flights alone")
    ax.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$", title="Paragliders: H_loc by the size of the configuration")
    ax.title.set_fontsize(9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, path)


def phase_figure(results, path, pops, colors):
    """Columns: phases; rows: MSD of the populations of `results`, and its local exponent."""
    fig, axes = plt.subplots(2, len(PHASES), figsize=(4.2 * len(PHASES), 6.5), sharex=True, squeeze=False)
    for col, phase in enumerate(PHASES):
        for pop, r in results.items():
            x, m = phase_msd.drawn(r[phase])
            if not x.size:
                continue
            style = dict(color=colors[pop], lw=1, ls="--" if pop.endswith("/alone") else "-")
            label = None if pop.endswith("/alone") else f"{pops[pop]} ({r[phase]['flights']})"
            axes[0, col].plot(x, m, label=label, **style)
            axes[1, col].plot(x, phase_msd.local_exponent(x, m), **style)
        axes[0, col].set(xscale="log", yscale="log", title=phase)
        axes[0, col].legend(frameon=False, fontsize=6.5)
        for y in (2, 1):
            axes[1, col].axhline(y, color="#999999", lw=0.6, ls=":")
        axes[1, col].set(xscale="log", xlabel=r"$\Delta$ (s)", ylim=(0, 2.3))
    axes[0, 0].set_ylabel(r"MSD (m$^2$)")
    axes[1, 0].set_ylabel(r"$d\,\log\mathrm{MSD} / d\,\log\Delta$")
    save(fig, path)


def trend_figure(pooled, items, stats, path, title):
    """One panel per statistic: its value against the number of pilots (solo first). The pooled
    populations are the bold line with its band; each site-day of `items` (label, result) a thin grey
    line with its own marker and its bands."""
    cats = ["solo", *SIZES]
    fig, axes = plt.subplots(1, len(stats), figsize=(3.6 * len(stats) + 1.8, 3.6), squeeze=False)
    for ax, name in zip(axes[0], stats):
        for k, (label, result) in enumerate([("all site-days", pooled), *items]):
            x = np.array([i for i, c in enumerate(cats) if c in result], float)
            v, band = (np.array([result[cats[int(i)]][name][key] for i in x]) for key in ("value", "band"))
            bold = k == 0
            ax.errorbar(x + (0 if bold else 0.06 * (k - 3)), v, yerr=[v - band[:, 0], band[:, 1] - v],
                        color="#c24a1d" if bold else GREY, marker="o" if bold else MARKERS[k - 1], ms=6 if bold else 4,
                        lw=2 if bold else 0.8, elinewidth=1.5 if bold else 0.8, capsize=0, label=label, zorder=3 if bold else 2)
        ax.set_xticks(range(len(cats)), ["solo", *(s[1:] for s in SIZES)])
        ax.set(title=name, xlabel="pilots in the configuration")
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, -1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{title} (5-95% bands)", fontsize=10)
    save(fig, path)


def to_json(path, value):
    path.write_text(json.dumps(value, indent=1))


if __name__ == "__main__":
    tamsd.folders()
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    if "--recompute" in sys.argv or "--recompute-pieces" in sys.argv or not all(p.exists() for p in SIZE_CACHE.values()):
        measure_sizes(flights, membership, summary, pieces_only="--recompute-pieces" in sys.argv)
    block = summary.set_index("flight")["block"]
    # On a site-day a large episode is one block of the note: there the bands resample the flights.
    by_flight = summary.assign(block=summary["flight"])
    day_block = by_flight.set_index("flight")["block"]
    spans = stretches(flights, membership, summary, by_size=True)
    site = pd.Series(sites(flights["lat0"], flights["lon0"], flights["date"]), index=flights["flight"])
    names = pd.read_csv(tamsd.FIXES["para"].parent.parent / "catalog/catalog.csv", usecols=["flight_id", "takeoff"],
                        dtype=str).rename(columns={"takeoff": "site_name"})
    spec = sys.argv[sys.argv.index("--sitedays") + 1].split(",") if "--sitedays" in sys.argv else None
    days = choose_sitedays(read_pieces(SIZE_CACHE["pieces"])["para"], flights.merge(names, how="left", on="flight_id"),
                           site, spec)
    to_json(OUT / "siteday_chosen.json", [{"site_day": label, "flights": len(f)} for label, f in days])
    print("site-days:", [label for label, _ in days], flush=True)
    pops, colors = labelled(True)
    plain, plain_colors = labelled(False)
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None

    def stage(name):
        """Whether to run `name`: every stage but "slopes" unless --only names some; "slopes" only when named.
        Each stage has its own generator, so a stage gives the same numbers alone or with the others."""
        return name in only if only is not None else name != "slopes"

    rng = np.random.default_rng(SEED)
    if stage("hloc"):  # from the JSON of the tamsd stage
        hloc_figure(json.loads((OUT / "size_tamsd.json").read_text())["para"], OUT / "size_hloc.pdf", pops, colors)

    if stage("tamsd"):
        pooled = {d: tamsd_results(tamsd_pieces(d), summary, rng, None, pops) for d in tamsd.FIXES}
        to_json(OUT / "size_tamsd.json", pooled)
        curves_figure([(NAMES[d], pooled[d]) for d in pooled], OUT / "size_tamsd.pdf", pops, colors)
        para = tamsd_pieces("para")
        on_days = [(label, tamsd_results(para, by_flight, rng, f, plain)) for label, f in days]
        to_json(OUT / "siteday_tamsd.json", dict(on_days))
        curves_figure([(f"{label} ({len(f)} flights)", r) for (label, r), (_, f) in zip(on_days, days)],
                      OUT / "siteday_tamsd.pdf", plain, plain_colors, per_row=3)

    for name, table, stats, plotted, title in (
            ("budget", None, budget.STATS, budget.PLOTTED, "Paragliders: how the time is spent, by the number of pilots"),
            ("performance", None, performance.STATS, performance.PLOTTED,
             "Paragliders: climb rate, glide speed and glide ratio, by the number of pilots")):
        if not stage(name):
            continue
        rng = np.random.default_rng(SEED)
        table = budget.phase_runs(flights, spans) if name == "budget" else terms()
        result = ratio_results(table, block, rng, None, stats, pops)
        to_json(OUT / f"size_{name}.json", result)
        budget.plot(result, OUT / f"size_{name}.pdf", {k: stats[k] for k in plotted}, title, present(pops, result), colors)
        items = [(label, ratio_results(table, day_block, rng, f, stats, plain)) for label, f in days]
        to_json(OUT / f"siteday_{name}.json", dict(items))
        trend_figure(result, items, {k: stats[k] for k in plotted}, OUT / f"siteday_{name}.pdf", title)

    if stage("phase"):
        phase, values = phase_pieces("para")
        results, _ = phase_results(phase, values, block, None, None, pops)
        to_json(OUT / "size_phase_msd.json", results)
        phase_figure(results, OUT / "size_phase_msd.pdf", pops, colors)
        for label, f in days:  # how many pieces a site-day offers: too few to draw its lags
            r, _ = phase_results(phase, values, day_block, None, f, plain)
            print(label, {pop: {ph: int(max(r[pop][ph]["runs"])) for ph in PHASES} for pop in r}, flush=True)

    if stage("slopes"):  # about an hour: the bootstrap of the slopes of phase_msd.py on more populations
        phase, values = phase_pieces("para")
        _, slopes = phase_results(phase, values, block, np.random.default_rng(SEED), None, pops, slopes=True)
        to_json(OUT / "size_phase_slopes.json", slopes)

    if stage("survival"):
        rng = np.random.default_rng(SEED)
        paragliders = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]]
        counted = survival.assign(survival.counted(pd.read_parquet(RUNS["para"]).merge(paragliders, on="flight_id")),
                                  spans)
        rows, labels = select(counted["population"].to_numpy(), counted["flight"].to_numpy(), block, controls=False)
        data, fits = survival.survival(counted.iloc[rows].assign(population=labels), block, rng, present(plain, labels))
        to_json(OUT / "size_survival.json", fits)
        survival.plot(data, fits, OUT / "size_survival.pdf", present(plain, labels), plain_colors, MIN_TAIL)
