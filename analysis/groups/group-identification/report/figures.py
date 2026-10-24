"""The summary figures of the report, from the JSON of the analyses: out/<run>/report_*.pdf.

Each figure replaces a table of the main text (the tables stay in the appendices): estimates as
points with their 5-95% bands on one axis, the populations or comparisons named on the axis.

    uv run python report/figures.py    # from analysis/groups/group-identification, after report/tables.py
"""

import matplotlib.pyplot as plt
import numpy as np

from tables import MAIN, OUTS, SINK, load

OUT = OUTS / MAIN
COLOR = {"S": "#2a78d6", "G": "#eb6834", "A": "#1baf7a"}
INK, MUTED, LIGHT = "#0b0b0b", "#9a9893", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": "#52514e", "ytick.color": INK})


def at(e, lag, key="H_loc", band="H_loc_band"):
    """Value and band of a curve at `lag`."""
    k = int(np.searchsorted(np.array(e["lags"]), lag))
    return e[key][k], (e[band][0][k], e[band][1][k])


def forest(ax, rows, title, zero=True, breaks=()):
    """Rows of (label, value, (low, high), color, filled), top to bottom, on one axis; a thin line
    after each row index in `breaks` separates the blocks."""
    y = np.arange(len(rows))[::-1]
    for i in breaks:
        ax.axhline(len(rows) - 1.5 - i, color=LIGHT, lw=0.8)
    for yi, (_, v, (lo, hi), color, filled) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.6, solid_capstyle="round")
        ax.plot(v, yi, "o", ms=5, color=color, mfc=color if filled else "white", mew=1.4)
    ax.set_yticks(y, [r[0] for r in rows])
    if zero:
        ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    ax.grid(axis="x", color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8, loc="left")
    ax.tick_params(axis="y", length=0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def raw():
    """Pooled S, G and A for the two exponents and the five drawn phase statistics."""
    t, b, p = load(MAIN, "tamsd")["para"], load(MAIN, "budget"), load(MAIN, "performance")
    pops = (("solo", "S", "solo flights"), ("group", "G", "group time"), ("group_solo", "A", "same flights alone"))
    panels = [(r"$H_{\rm loc}(100~{\rm s})$", lambda k: (at(t[k], 100)[0], at(t[k], 100)[1])),
              (r"$H_{\rm loc}(1000~{\rm s})$", lambda k: (at(t[k], 1000)[0], at(t[k], 1000)[1]))]
    for src, s, label in ((b, "time in climb (%)", "time in climb (%)"),
                          (b, "searches that end in a climb (%)", "searches ending in a climb (%)"),
                          (p, "climb rate (m/s)", "climb rate (m/s)"), (p, "glide speed (km/h)", "glide speed (km/h)"),
                          (p, "glide ratio", "glide ratio")):
        panels.append((label, lambda k, src=src, s=s: (src[k][s]["value"], tuple(src[k][s]["band"]))))
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 3.4), constrained_layout=True)
    for ax, (title, get) in zip(axes.flat, panels):
        forest(ax, [(name, *get(k), COLOR[c], True) for k, c, name in pops], title, zero=False)
        if ax not in axes[:, 0]:
            ax.set_yticklabels([])
    axes.flat[-1].axis("off")
    save(fig, "report_raw")


def zones_forest():
    """Delta H_loc(1000 s) by stratification: G - S reweighted and as is, G - A reweighted."""
    z = load(MAIN, "zones")
    rows = []
    for name, r in z.items():
        S, A = r["S"], r["A"]
        k = int(np.searchsorted(np.array(S["lags"]), 1000))
        label = f"{name.replace(' x ', ' × ')}, {100 * S['share of G kept']:.0f}% kept"
        band = lambda e, key: (e[key + "_band"][0][k], e[key + "_band"][1][k])
        rows.append([(f"{label}: G − S reweighted", S["dH_loc"][k], band(S, "dH_loc"), COLOR["S"], True),
                     ("G − S as is", S["dH_loc_support"][k], (S["dH_loc_support"][k],) * 2, COLOR["S"], False),
                     ("G − A reweighted", A["dH_loc"][k], band(A, "dH_loc"), COLOR["A"], True)])
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    forest(ax, [r for group in rows for r in group], r"$\Delta H_{\rm loc}(1000~{\rm s})$, by stratification",
           breaks=range(2, 3 * len(rows) - 1, 3))
    save(fig, "report_zones")


def route_forest():
    """Delta H_loc(1000 s) of the matched pieces: against the same route alone (any date, same season
    and hour) and against the solo flights of their site-day and hour, for every variant."""
    variants = [("10 000 pieces, 500 m, 80%", "route_sample10000"), ("2000, 500 m, 80%", "route_sample2000"),
                ("2000, 300 m, 80%", "route_sample2000_rho300_cover0.8"),
                ("2000, 1000 m, 80%", "route_sample2000_rho1000_cover0.8"),
                ("2000, 500 m, 70%", "route_sample2000_rho500_cover0.7"),
                ("2000, 500 m, 90%", "route_sample2000_rho500_cover0.9")]
    rows = []
    for label, name in variants:
        r = load(MAIN, name)
        a, s = r["any date"], r["same season and hour"]
        get = lambda e: (e["H_loc(1000 s)"]["value"], tuple(e["H_loc(1000 s)"]["band"]))
        rows += [(f"{label}: G − S, site-day and hour", *get(a["baseline"]), COLOR["S"], True),
                 ("G − same route alone", *get(a), COLOR["G"], True),
                 ("G − same route, season and hour", *get(s), COLOR["G"], False)]
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    forest(ax, rows, r"$\Delta H_{\rm loc}(1000~{\rm s})$ of the matched pieces", breaks=range(2, len(rows) - 1, 3))
    save(fig, "report_route")


def prepost_forest():
    """Group minus alone on entering and on leaving, each above the drift it is to be read against
    (later minus earlier for entering, the opposite for leaving), and after minus before."""
    r = load(MAIN, "prepost_3000")
    ps = r["phase statistics"]
    flip = lambda v, band: (-v, (-band[1], -band[0]))

    def get(name, stat):
        if stat in ("H_loc(1000)", "H_loc(100)"):
            v, band = at(r[name], 1000 if stat == "H_loc(1000)" else 100, "dH_loc", "dH_loc_band")
        elif stat == "H":
            v, band = r[name]["dH"], tuple(r[name]["dH_band"])
        else:
            v, band = ps[name][stat]["value"], tuple(ps[name][stat]["band"])
        return v, band

    rows = [("entering: start − before", "entry", False, COLOR["G"], True),
            ("drift A, later − earlier", "control, group flights alone", False, COLOR["A"], False),
            ("drift S, later − earlier", "control, solo flights", False, COLOR["S"], False),
            ("leaving: end − after", "exit", False, COLOR["G"], True),
            ("drift A, earlier − later", "control, group flights alone", True, COLOR["A"], False),
            ("drift S, earlier − later", "control, solo flights", True, COLOR["S"], False),
            ("after − before", "after minus before", False, INK, True)]
    panels = [(r"$\Delta H_{\rm loc}(1000~{\rm s})$", "H_loc(1000)"), (r"$\Delta H_{\rm loc}(100~{\rm s})$", "H_loc(100)"),
              (r"$\Delta H$ (fitted)", "H"), ("glide speed (km/h)", "glide speed (km/h)"),
              ("height lost per km (m/km)", SINK), ("time in climb (points)", "time in climb (%)"),
              ("successful searches (points)", "searches that end in a climb (%)"), ("climb rate (m/s)", "climb rate (m/s)")]
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.6), constrained_layout=True)
    for ax, (title, stat) in zip(axes.flat, panels):
        data = []
        for label, name, reverse, color, filled in rows:
            v, band = get(name, stat)
            data.append((label, *(flip(v, band) if reverse else (v, band)), color, filled))
        forest(ax, data, title, breaks=(2, 5))
        if ax not in axes[:, 0]:
            ax.set_yticklabels([])
    save(fig, "report_prepost")


def estimates():
    """Every estimate of Delta H_loc(1000 s), from the raw one to the four tests."""
    d, z = load(MAIN, "decompose"), load(MAIN, "zones")
    r, pp = load(MAIN, "route_sample10000"), load(MAIN, "prepost_3000")
    dec = lambda c: (d[c]["H_loc(1000 s)"]["value"], tuple(d[c]["H_loc(1000 s)"]["band"]))
    sdh = z["site-day x hour"]
    k = int(np.searchsorted(np.array(sdh["S"]["lags"]), 1000))
    zone = lambda e: (e["dH_loc"][k], (e["dH_loc_band"][0][k], e["dH_loc_band"][1][k]))
    route = lambda e: (e["H_loc(1000 s)"]["value"], tuple(e["H_loc(1000 s)"]["band"]))
    moment = lambda name: at(pp[name], 1000, "dH_loc", "dH_loc_band")
    rows = [("raw: G − S pooled", *dec("raw"), INK, True),
            ("pilots: G − S, same site-day and hour", *dec("same site-day-hour"), COLOR["S"], True),
            ("A − S, same site-day and hour (selection)", *dec("selection"), MUTED, False),
            ("G − A, same flight", *dec("group, same flight"), COLOR["A"], True),
            ("G − A, same site-day and hour", *dec("group, same site-day-hour"), COLOR["A"], True),
            ("place: G − S reweighted, site-day × hour", *zone(sdh["S"]), COLOR["S"], True),
            ("G − A reweighted, site-day × hour", *zone(sdh["A"]), COLOR["A"], True),
            ("route: matched G − S, site-day and hour", *route(r["any date"]["baseline"]), COLOR["S"], True),
            ("matched G − same route alone", *route(r["any date"]), COLOR["G"], True),
            ("the same, same season and hour", *route(r["same season and hour"]), COLOR["G"], False),
            ("moment: entering, start − before", *moment("entry"), COLOR["G"], True),
            ("leaving, end − after", *moment("exit"), COLOR["G"], True),
            ("after − before", *moment("after minus before"), MUTED, False)]
    fig, ax = plt.subplots(figsize=(6.2, 3.9), constrained_layout=True)
    forest(ax, rows, r"$\Delta H_{\rm loc}(1000~{\rm s})$: every estimate", breaks=(0, 4, 6, 9))
    save(fig, "report_estimates")


if __name__ == "__main__":
    raw()
    zones_forest()
    route_forest()
    prepost_forest()
    estimates()
