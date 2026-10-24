"""The figures of the report, from the JSON of the analyses: out/<run>/report_*.pdf.

All are drawn at the width of the text (6.6 in), so that their 8-pt labels print at 8 pt. Colour
names the reference of a comparison (S blue, A green, the same route alone purple, the other
controls grey, the group itself orange); a filled point is an estimate, an open one a control or
a variant. Estimates are points with their 5-95% bands on one axis, curves over every lag carry
their bands.

    uv run python report/figures.py    # from analysis/groups/group-identification, after report/tables.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from tables import MAIN, OUTS, SINK, load

OUT = OUTS / MAIN
WIDTH = 6.6  # in, the text width of the report
COLOR = {"S": "#2a78d6", "G": "#eb6834", "A": "#1baf7a", "R": "#7b5cc4"}
INK, GREY, MUTED, LIGHT = "#0b0b0b", "#76746f", "#9a9893", "#e6e5e1"
STRATA = ("zone", "zone x month", "zone x hour", "site-day x hour")
STATS = {"H_loc(100 s)": r"$\Delta H_{\rm loc}(100~{\rm s})$", "H_loc(1000 s)": r"$\Delta H_{\rm loc}(1000~{\rm s})$",
         "time in climb (%)": "$\\Delta$ time in climb\n(points)",
         "searches that end in a climb (%)": "$\\Delta$ searches ending\nin a climb (points)",
         "climb rate (m/s)": r"$\Delta$ climb rate (m/s)", "glide speed (km/h)": r"$\Delta$ glide speed (km/h)",
         SINK: "$\\Delta$ height lost\nper km (m/km)"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": "#52514e", "ytick.color": INK, "axes.titlesize": 8, "legend.frameon": False})


def at(e, lag, key="H_loc", band="H_loc_band"):
    """Value and band of a curve at `lag`."""
    k = int(np.searchsorted(np.array(e["lags"]), lag))
    return e[key][k], (e[band][0][k], e[band][1][k])


def described(v):
    return v["value"], tuple(v["band"])


def forest(ax, rows, title, zero=True, breaks=(), ticks=4):
    """Rows of (label, value, (low, high), color, filled), top to bottom, on one axis; a thin line
    after each row index in `breaks` separates the blocks."""
    y = np.arange(len(rows))[::-1]
    for i in breaks:
        ax.axhline(len(rows) - 1.5 - i, color=LIGHT, lw=0.8)
    for yi, (_, v, (lo, hi), color, filled) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.6, solid_capstyle="round")
        ax.plot(v, yi, "o", ms=4.5, color=color, mfc=color if filled else "white", mew=1.3)
    ax.set_yticks(y, [r[0] for r in rows])
    if zero:
        ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    ax.grid(axis="x", color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(ticks))
    ax.margins(x=0.1)


def panels(rows_of, titles, shape, height, name, breaks=()):
    """Small multiples of forest(): one panel per title, the row labels on the first column only."""
    fig, axes = plt.subplots(*shape, figsize=(WIDTH, height), constrained_layout=True)
    for ax, title in zip(axes.flat, titles):
        forest(ax, rows_of(title), title, breaks=breaks, ticks=3)
        ax.tick_params(axis="x", labelsize=7)
        if ax not in axes[:, 0]:
            ax.set_yticklabels([])
    for ax in axes.flat[len(titles):]:
        ax.axis("off")
    save(fig, name)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def raw():
    """Pooled S, G and A for the two exponents and the five drawn phase statistics."""
    t, b, p = load(MAIN, "tamsd")["para"], load(MAIN, "budget"), load(MAIN, "performance")
    pops = (("solo", "S", "S, solo flights"), ("group", "G", "G, group time"), ("group_solo", "A", "A, same flights alone"))
    get = {r"$H_{\rm loc}(100~{\rm s})$": lambda k: at(t[k], 100), r"$H_{\rm loc}(1000~{\rm s})$": lambda k: at(t[k], 1000)}
    for src, s, title in ((b, "time in climb (%)", "time in climb (%)"),
                          (b, "searches that end in a climb (%)", "searches ending\nin a climb (%)"),
                          (p, "climb rate (m/s)", "climb rate (m/s)"), (p, "glide speed (km/h)", "glide speed (km/h)"),
                          (p, "glide ratio", "glide ratio")):
        get[title] = lambda k, src=src, s=s: described(src[k][s])
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 2.9), constrained_layout=True)
    for ax, (title, g) in zip(axes.flat, get.items()):
        forest(ax, [(name, *g(k), COLOR[c], True) for k, c, name in pops], title, zero=False, ticks=3)
        ax.tick_params(axis="x", labelsize=7)
        if ax not in axes[:, 0]:
            ax.set_yticklabels([])
    axes.flat[-1].axis("off")
    save(fig, "report_raw")


def contrasts():
    """The five stratified contrasts of decompose.py, one panel per statistic."""
    d = load(MAIN, "decompose")
    rows = [("G − S, pooled", "raw", INK, True), ("G − S, site-day and hour", "same site-day-hour", COLOR["S"], True),
            ("A − S, site-day and hour", "selection", GREY, False), ("G − A, same flight", "group, same flight", COLOR["A"], True),
            ("G − A, site-day and hour", "group, same site-day-hour", COLOR["A"], False)]
    stat = {v: k for k, v in STATS.items()}
    panels(lambda title: [(label, *described(d[c][stat[title]]), color, filled) for label, c, color, filled in rows],
           list(stat), (2, 4), 3.4, "report_contrasts")


def zones_lags():
    """Delta H_loc over every lag, G - S and G - A, the other population reweighted to the strata of G,
    from coarse to fine strata (light to dark)."""
    z = load(MAIN, "zones")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), constrained_layout=True, sharey=True)
    for ax, (c, title) in zip(axes, (("S", "G − S, S reweighted to the strata of G"),
                                     ("A", "G − A, A reweighted to the strata of G"))):
        shades = plt.matplotlib.colors.LinearSegmentedColormap.from_list("", ["#d9d7d2", COLOR[c]])
        for i, name in enumerate(("none", *STRATA)):
            e = z[name][c]
            lags = np.array(e["lags"])
            color = GREY if name == "none" else shades(0.35 + 0.65 * i / len(STRATA))
            ax.fill_between(lags, *np.array(e["dH_loc_band"]), color=color, alpha=0.18, lw=0)
            ax.plot(lags, e["dH_loc"], color=color, lw=1.4,
                    label="no strata" if name == "none" else name.replace(" x ", " × "))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=title)
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel(r"$\Delta H_{\rm loc}(\tau)$")
    save(fig, "report_zones_lags")


def zones_forest():
    """Delta H_loc(1000 s) by stratification: G - S and G - A, reweighted and as is."""
    z = load(MAIN, "zones")
    rows = []
    for name in STRATA:
        S, A = z[name]["S"], z[name]["A"]
        k = int(np.searchsorted(np.array(S["lags"]), 1000))
        band = lambda e, key: (e[key + "_band"][0][k], e[key + "_band"][1][k])
        label = f"{name.replace(' x ', ' × ')}, {100 * S['share of G kept']:.0f}% kept: G − S reweighted"
        rows += [(label, S["dH_loc"][k], band(S, "dH_loc"), COLOR["S"], True),
                 ("G − S as is", S["dH_loc_support"][k], band(S, "dH_loc_support"), COLOR["S"], False),
                 ("G − A reweighted", A["dH_loc"][k], band(A, "dH_loc"), COLOR["A"], True),
                 ("G − A as is", A["dH_loc_support"][k], band(A, "dH_loc_support"), COLOR["A"], False)]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6), constrained_layout=True)
    forest(ax, rows, r"$\Delta H_{\rm loc}(1000~{\rm s})$, by stratification", breaks=range(3, len(rows) - 1, 4))
    save(fig, "report_zones")


def route_curves():
    """The matched pieces of group time and the same routes flown alone: H_loc over every lag, and
    their difference, any date and same season and hour."""
    r = load(MAIN, "route_sample10000")
    lags = np.array(load(MAIN, "tamsd")["para"]["group"]["lags"])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), constrained_layout=True)
    a = r["any date"]
    axes[0].plot(lags, a["H_loc"]["group"], color=COLOR["G"], lw=1.4, label="G, matched pieces")
    axes[0].plot(lags, a["H_loc"]["route"], color=COLOR["R"], lw=1.4, label="same route alone, another day")
    axes[0].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$")
    axes[0].legend(loc="lower left", fontsize=7)
    for key, alpha, label in (("any date", 1.0, "any date"), ("same season and hour", 0.45, "same season and hour")):
        e = r[key]
        axes[1].fill_between(lags, *np.array(e["Delta H_loc band"]), color=COLOR["R"], alpha=0.15 * alpha + 0.05, lw=0)
        axes[1].plot(lags, e["Delta H_loc"], color=COLOR["R"], lw=1.4, alpha=alpha, label=f"{label}: {e['pieces']:,} pieces".replace(",", " "))
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\Delta H_{\rm loc}(\tau)$, G $-$ same route alone")
    axes[1].legend(loc="upper left", fontsize=7)
    save(fig, "report_route_curves")


def route_forest():
    """Delta H_loc(1000 s) of the matched pieces for every variant: against the solo flights of their
    site-day and hour, and against the same route alone (any date, same season and hour)."""
    variants = [("10 000 pieces, ρ 500 m, cover 80%", "route_sample10000"), ("2000 pieces, ρ 500 m, cover 80%", "route_sample2000"),
                ("2000 pieces, ρ 300 m, cover 80%", "route_sample2000_rho300_cover0.8"),
                ("2000 pieces, ρ 1000 m, cover 80%", "route_sample2000_rho1000_cover0.8"),
                ("2000 pieces, ρ 500 m, cover 70%", "route_sample2000_rho500_cover0.7"),
                ("2000 pieces, ρ 500 m, cover 90%", "route_sample2000_rho500_cover0.9")]
    rows = []
    for label, name in variants:
        r = load(MAIN, name)
        a, s = r["any date"], r["same season and hour"]
        rows += [(f"{label}: G − S, site-day and hour", *described(a["baseline"]["H_loc(1000 s)"]), COLOR["S"], True),
                 ("G − same route alone", *described(a["H_loc(1000 s)"]), COLOR["R"], True),
                 ("G − same route, season and hour", *described(s["H_loc(1000 s)"]), COLOR["R"], False)]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2), constrained_layout=True)
    forest(ax, rows, r"$\Delta H_{\rm loc}(1000~{\rm s})$ of the matched pieces", breaks=range(2, len(rows) - 1, 3))
    save(fig, "report_route")


def prepost_forest():
    """Group minus alone on entering and on leaving, each above the drift it is to be read against
    (later minus earlier for entering, the opposite for leaving), and after minus before."""
    r = load(MAIN, "prepost_3000")
    ps = r["phase statistics"]
    flip = lambda v, band: (-v, (-band[1], -band[0]))

    def get(name, stat):
        if stat in ("H_loc(1000 s)", "H_loc(100 s)"):
            return at(r[name], 1000 if stat == "H_loc(1000 s)" else 100, "dH_loc", "dH_loc_band")
        if stat == "H":
            return r[name]["dH"], tuple(r[name]["dH_band"])
        return described(ps[name][stat])

    rows = [("entering: start − before", "entry", False, COLOR["G"], True),
            ("drift A, later − earlier", "control, group flights alone", False, COLOR["A"], False),
            ("drift S, later − earlier", "control, solo flights", False, COLOR["S"], False),
            ("leaving: end − after", "exit", False, COLOR["G"], True),
            ("drift A, earlier − later", "control, group flights alone", True, COLOR["A"], False),
            ("drift S, earlier − later", "control, solo flights", True, COLOR["S"], False),
            ("after − before", "after minus before", False, GREY, False)]
    titles = {**{STATS[s]: s for s in ("H_loc(1000 s)", "H_loc(100 s)")}, r"$\Delta H$ (fitted)": "H",
              **{STATS[s]: s for s in ("glide speed (km/h)", SINK, "time in climb (%)", "searches that end in a climb (%)",
                                       "climb rate (m/s)")}}

    def rows_of(title):
        out = []
        for label, name, reverse, color, filled in rows:
            v, band = get(name, titles[title])
            out.append((label, *(flip(v, band) if reverse else (v, band)), color, filled))
        return out

    panels(rows_of, list(titles), (2, 4), 4.2, "report_prepost", breaks=(2, 5))


def size():
    """The contrasts by size of decompose.py: G_n - A within the flight and G_n - G_2 at the same
    site-day and hour, one panel per statistic."""
    d = load(MAIN, "decompose_size")
    sizes = ("n2", "n3", "n4", "n5-6", "n7+")
    names = {"n2": "2", "n3": "3", "n4": "4", "n5-6": "5–6", "n7+": "7+"}
    stats = ("H_loc(1000 s)", "time in climb (%)", "glide speed (km/h)", SINK, "climb rate (m/s)",
             "searches that end in a climb (%)")
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.6), constrained_layout=True)
    x = np.arange(len(sizes))
    for ax, stat in zip(axes.flat, stats):
        for dx, key, color, filled, label in ((-0.12, "{} - A, same flight", COLOR["A"], True, r"G$_n$ − A, same flight"),
                                             (0.12, "{} - n2, same site-day-hour", GREY, False,
                                              r"G$_n$ − G$_2$, site-day and hour")):
            for i, n in enumerate(sizes):
                e = d.get(key.format(n))
                if e is None:
                    continue
                v, (lo, hi) = described(e[stat])
                ax.plot([i + dx] * 2, [lo, hi], color=color, lw=1.6, solid_capstyle="round")
                ax.plot(i + dx, v, "o", ms=4.5, color=color, mfc=color if filled else "white", mew=1.3,
                        label=label if i == 1 else None)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        ax.set_xticks(x, [names[n] for n in sizes])
        ax.set_title(STATS[stat], loc="left")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="y", color=LIGHT, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("pilots in the configuration")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, fontsize=7)
    save(fig, "report_size")


def estimates():
    """Every estimate of Delta H_loc(1000 s), from the raw one to the four tests."""
    d, z = load(MAIN, "decompose"), load(MAIN, "zones")
    r, pp = load(MAIN, "route_sample10000"), load(MAIN, "prepost_3000")
    dec = lambda c: described(d[c]["H_loc(1000 s)"])
    sdh = z["site-day x hour"]
    k = int(np.searchsorted(np.array(sdh["S"]["lags"]), 1000))
    zone = lambda e: (e["dH_loc"][k], (e["dH_loc_band"][0][k], e["dH_loc_band"][1][k]))
    moment = lambda name: at(pp[name], 1000, "dH_loc", "dH_loc_band")
    rows = [("raw: G − S pooled", *dec("raw"), INK, True),
            ("strata: G − S, site-day and hour", *dec("same site-day-hour"), COLOR["S"], True),
            ("A − S, site-day and hour (selection)", *dec("selection"), GREY, False),
            ("G − A, same flight", *dec("group, same flight"), COLOR["A"], True),
            ("G − A, site-day and hour", *dec("group, same site-day-hour"), COLOR["A"], False),
            ("place: G − S reweighted, site-day × hour", *zone(sdh["S"]), COLOR["S"], True),
            ("G − A reweighted, site-day × hour", *zone(sdh["A"]), COLOR["A"], True),
            ("route: matched G − S, site-day and hour", *described(r["any date"]["baseline"]["H_loc(1000 s)"]), COLOR["S"], True),
            ("matched G − same route alone", *described(r["any date"]["H_loc(1000 s)"]), COLOR["R"], True),
            ("the same, same season and hour", *described(r["same season and hour"]["H_loc(1000 s)"]), COLOR["R"], False),
            ("moment: entering, start − before", *moment("entry"), COLOR["G"], True),
            ("leaving, end − after", *moment("exit"), COLOR["G"], True),
            ("after − before", *moment("after minus before"), GREY, False)]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5), constrained_layout=True)
    forest(ax, rows, r"$\Delta H_{\rm loc}(1000~{\rm s})$: every estimate", breaks=(0, 4, 6, 9))
    save(fig, "report_estimates")


if __name__ == "__main__":
    raw()
    contrasts()
    zones_lags()
    zones_forest()
    route_curves()
    route_forest()
    prepost_forest()
    size()
    estimates()
