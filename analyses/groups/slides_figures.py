"""Figures of the slides of meetings/2026-11/group_vs_solo that show a few panels or rows of the results of the
report, drawn from the JSON of the analyses; nothing is recomputed.

    uv run python slides_figures.py --open
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import curves
import decompose
import decomposition
import matched
import prepost
from tamsd import COLORS, OUT, SUPPORT, save

PHASE_STATS = ("time in climb (%)", "time in search (%)", "searches that end in a climb (%)", "climbs per hour",
               "climb rate (m/s)", "glide speed (km/h)", decompose.SINK)
MAIN = ("H_loc(1840 s)", "time in climb (%)", "glide speed (km/h)")


def read(name):
    return json.loads((OUT / f"{name}.json").read_text())


def quote(q):
    return {"value": float(q["value"]), "band": [float(x) for x in q["band"]]}


def raw_phases():
    """H_loc and the seven phase statistics of S, A, G, on a 3 x 3 grid sized for the slide: `slide_raw_phases`."""
    raw = read("raw")
    stats = raw["phase statistics"]
    hloc = {"$H_{\\mathrm{loc}}(1000\\,\\mathrm{s})$": ("1000", "H_loc(1000 s)"),
            "$H_{\\mathrm{loc}}(1840\\,\\mathrm{s})$": ("2760", "H_loc(1840 s)")}
    rows = [(label, COLORS[pop], {**{s: quote(stats[short]["average first"][s]) for s in PHASE_STATS},
                                  **{name: quote(raw["datasets"][ds]["populations"][short][key]) for name, (ds, key) in hloc.items()}})
            for label, pop, short in (("S", "solo", "S"), ("A", "group_solo", "A"), ("G", "group", "G"))]
    short = {"time in climb (%)": "climb time (%)", "time in search (%)": "search time (%)",
             "searches that end in a climb (%)": "searches → climb (%)", decompose.SINK: "height lost (m/km)"}
    rows = [(label, color, {short.get(k, k): v for k, v in quotes.items()}) for label, color, quotes in rows]
    panels = [*hloc, "climb time (%)", "search time (%)", "searches → climb (%)", "climbs per hour",
              "climb rate (m/s)", "glide speed (km/h)", "height lost (m/km)"]
    number = {name: f"{i}  {name}" for i, name in enumerate(panels, 1)}
    rows = [(label, color, {number[k]: v for k, v in quotes.items()}) for label, color, quotes in rows]
    matched.draw_forest(rows, OUT / "slide_raw_phases.pdf", None, stats=list(number.values()), per_row=3, width=4.4,
                        prefix="", zero=False, row_h=1.08)


def raw_delta():
    """H_loc of S, A, G with their flights in the legend, beside the differences, on each dataset of the raw comparison:
    `slide_raw_delta` (1000 s), `slide_raw_2760_delta`, `slide_raw_longest_delta`."""
    raw = read("raw")
    colors = {"G - S": COLORS["group"], "A - S": COLORS["group_solo"], "G - A": "#8c2f0f"}
    for name, out in (("1000", "slide_raw_delta"), ("2760", "slide_raw_2760_delta"), ("longest", "slide_raw_longest_delta")):
        d = raw["datasets"][name]
        left = [{**d["populations"][short]["curve"], "label": f"{short} ({d['populations'][short]['flights']:,} flights)", "color": COLORS[pop]}
                for pop, short in (("solo", "S"), ("group_solo", "A"), ("group", "G"))]
        gaps = raw["gaps"][name]["H_loc"]
        right = [{**gaps[k], "label": k, "color": colors[k]} for k in colors]
        g = gaps["G - S"]
        below = [i for i, v in enumerate(g["value"]) if v <= 0]
        turn = below[-1] + 1 if below and below[-1] + 1 < len(g["lags"]) else None
        curves.delta_figure(left, right, OUT / out,
                            vline=None if turn is None else (g["lags"][turn], f"G - S > 0 from {g['lags'][turn]:g} s"))


def raw_msd_fits():
    """MSD against the lag (log-log) of S, A, G on each dataset of the raw comparison, with the line of the fitted H
    (least squares of log MSD on log lag, equal weight per lag); below, H_loc against the lag with the fitted H as a
    dashed line. `slide_raw_msd_fits`."""
    raw = read("raw")["datasets"]
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 3.9), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]},
                             constrained_layout=True)
    for j, name in enumerate(("1000", "2760", "longest")):
        top, bottom = axes[0, j], axes[1, j]
        for pop, short in (("solo", "S"), ("group_solo", "A"), ("group", "G")):
            q = raw[name]["populations"][short]
            c = q["curve"]
            lags, msd = np.array(c["lags"]), np.array(c["msd"])
            slope, icpt = np.polyfit(np.log10(lags), np.log10(msd), 1)
            top.plot(lags, msd, "o", ms=2.5, color=COLORS[pop], label=f"{short}: H = {slope / 2:.3f} ({q['flights']:,})")
            top.plot(lags, 10 ** icpt * lags ** slope, color=COLORS[pop], lw=0.9, ls="--")
            bottom.plot(lags, c["H_loc"], color=COLORS[pop], lw=1.2)
            bottom.axhline(slope / 2, color=COLORS[pop], lw=0.8, ls="--")
        top.set(xscale="log", yscale="log", title=f"to {raw[name]['tau_max']:,} s")
        top.title.set_fontsize(9)
        top.legend(frameon=False, fontsize=6.5, loc="upper left")
        bottom.set(xscale="log", xlabel=r"$\tau$ (s)")
        for ax in (top, bottom):
            ax.grid(color="#eeeeee", lw=0.5)
            ax.tick_params(labelsize=7)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel(r"MSD (m$^2$)")
    axes[1, 0].set_ylabel(r"$H_{\rm loc}(\tau)$")
    axes[0, 2].annotate("stronger\nselection bias", xy=(0.40, 0.72), xytext=(0.04, 0.38), xycoords="axes fraction", ha="left",
                        fontsize=7, color="#a02020", arrowprops={"arrowstyle": "->", "color": "#a02020", "lw": 1.1})
    fig.savefig(OUT / "slide_raw_msd_fits.pdf", bbox_inches="tight")
    fig.savefig(OUT / "slide_raw_msd_fits.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pools_delta():
    """The figure of slide 5 on two other populations (pools.py): `slide_pool_variable`, the pieces long enough for
    each lag, and `slide_pool_short`, the pieces of 1000 to 1250 s. The flights are in the legend."""
    pools = read("pools")
    colors = {"G - S": COLORS["group"], "A - S": COLORS["group_solo"], "G - A": "#8c2f0f"}
    for key, out in (("variable pool", "slide_pool_variable"), ("short pieces", "slide_pool_short")):
        d = pools[key]
        left = []
        for pop, short in (("solo", "S"), ("group_solo", "A"), ("group", "G")):
            q = d["populations"][short]
            n = q["flights"]
            label = f"{short} ({n[0]:,} to {n[-1]:,} flights)" if n[0] != n[-1] else f"{short} ({n[0]:,} flights)"
            left.append({"lags": d["lags"], "H_loc": q["H_loc"], "label": label, "color": COLORS[pop]})
        right = [{**d["gaps"][k], "label": k, "color": colors[k]} for k in colors]
        g = d["gaps"]["G - S"]
        below = [i for i, v in enumerate(g["value"]) if v <= 0]
        turn = below[-1] + 1 if below and below[-1] + 1 < len(g["lags"]) else None
        curves.delta_figure(left, right, OUT / out,
                            vline=None if turn is None else (g["lags"][turn], f"G - S > 0 from {g['lags'][turn]:g} s"))


def matched_deltas():
    """G - S|G and A - S|A against the lag, with their bands: pieces of the main series (left) and of the long
    series (right) on one y scale, `slide_matched_deltas`."""
    d = read("matched")["levels"]["all"]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2), sharey=True, constrained_layout=True)
    for ax, key in zip(axes, ("main", "long")):
        T = d["delta curves"][matched.CONTRASTS[0]][key]["lags"][-1]
        for c in matched.CONTRASTS[:2]:
            q = d["delta curves"][c][key]
            ax.fill_between(q["lags"], *q["band"], color=matched.CONTRAST_COLORS[c], alpha=0.18, lw=0)
            ax.plot(q["lags"], q["value"], color=matched.CONTRAST_COLORS[c], lw=1.4, label=c)
        ax.axhline(0, color="#999999", lw=0.8)
        g = d["delta curves"][matched.CONTRASTS[0]][key]
        below = [i for i, v in enumerate(g["value"]) if v <= 0]
        if below and below[-1] + 1 < len(g["lags"]):  # the lag from which G - S|G stays above 0
            turn = g["lags"][below[-1] + 1]
            ax.axvline(turn, color=matched.CONTRAST_COLORS[matched.CONTRASTS[0]], lw=0.9, ls="--")
            ax.annotate(f"{turn:g} s", (turn, 1), xycoords=("data", "axes fraction"), xytext=(3, -9),
                        textcoords="offset points", fontsize=7, color=matched.CONTRAST_COLORS[matched.CONTRASTS[0]])
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=f"pieces of at least {T / SUPPORT:g} s")
        ax.title.set_fontsize(9)
        ax.grid(color="#eeeeee", lw=0.5)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"$\Delta H_{\rm loc}(\tau)$")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.savefig(OUT / "slide_matched_deltas.pdf", bbox_inches="tight")
    fig.savefig(OUT / "slide_matched_deltas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def matched_forest():
    """The nine quantities of the same-route comparison at level "all": G - S|G, A - S|A and their difference,
    `slide_matched_forest`."""
    d = read("matched")["levels"]["all"]["deltas"]
    labels = {matched.CONTRASTS[0]: "G − S|G", matched.CONTRASTS[1]: "A − S|A", matched.CONTRASTS[2]: "difference"}
    rows = [(labels[c], matched.CONTRAST_COLORS[c], {m: quote(d[c][m]) for m in matched.FOREST}) for c in matched.CONTRASTS]
    matched.draw_forest(rows, OUT / "slide_matched_forest.pdf", None, stats=list(matched.FOREST), per_row=3, width=6.0,
                        wrap=22, row_h=1.6)


def prepost_delta():
    """H_loc of the four windows of the same flight beside D_in, D_out, g, d w and the two controls, `slide_prepost_delta`.
    Outside the group (pre, post): green and dashed; in the group (start, end): orange and solid."""
    d = read("prepost_2400")
    style = {"pre": (COLORS["group_solo"], "--"), "start": (COLORS["group"], "-"), "end": ("#8c2f0f", "-"),
             "post": ("#0f6e4c", "--")}
    left = [{**d["curves"]["stages"][w], "label": w, "color": style[w][0], "ls": style[w][1]} for w in style]
    keys = ("D_in", "D_out", "g", "d w", "ctrlA2 - ctrlA1", "ctrlS2 - ctrlS1")
    right = [{"lags": d["lags"][:len(d["differences"][k]["dH_loc"]["value"])], **d["differences"][k]["dH_loc"],
              "label": prepost.FOREST_LABELS[k], "color": prepost.FOREST[k], "ls": "-" if k in keys[:3] else "--",
              "lw": 1.4 if k == "g" else 0.9, **({} if k == "g" else {"band": None})} for k in keys]
    curves.delta_figure(left, right, OUT / "slide_prepost_delta", f"{d['curves']['stages']['events']:,} events, windows of 2400 s")


def prepost_forest():
    """H_loc(1000 s), H_loc(1840 s) and the seven phase statistics of the same flight (windows of 2400 s): D_in, D_out,
    g and the two controls, `slide_prepost_forest`."""
    r = read("prepost_2400")
    d, lags = r["differences"], r["lags"]
    i = lags.index(1840)
    at_1840 = lambda q: {"value": q["value"][i], "band": [q["band"][0][i], q["band"][1][i]]}
    rows_of = (("D_in", "D_in"), ("D_out", "D_out"), ("g", "g"), ("ctrlA2 - ctrlA1", "control, A"), ("ctrlS2 - ctrlS1", "control, solo"))
    short = {"H_loc(1000 s)": "H_loc(1000 s)", "time in climb (%)": "climb time (pp)", "time in search (%)": "search time (pp)",
             "searches that end in a climb (%)": "searches → climb (pp)", "climbs per hour": "climbs per hour",
             "climb rate (m/s)": "climb rate (m/s)", "glide speed (km/h)": "glide speed (km/h)",
             decompose.SINK: "height lost (m/km)"}
    rows = [(label, prepost.FOREST[k], {"H_loc(1840 s)": at_1840(d[k]["dH_loc"]), **{short[m]: quote(d[k][m]) for m in short}})
            for k, label in rows_of]
    stats = ["H_loc(1000 s)", "H_loc(1840 s)", *[v for k, v in short.items() if k != "H_loc(1000 s)"]]
    matched.draw_forest(rows, OUT / "slide_prepost_forest.pdf", None, stats=stats, per_row=3, width=6.4, wrap=22, row_h=1.75)


def factors():
    """G against three of the controls of the decomposition (no factor, route, all fixed), on the nine
    quantities, `slide_factors`."""
    d = read("decomposition")["levels"]["all"]
    rows = [(label, color, {m: quote(d[m]["cells"][cell]["delta"]) for m in matched.FOREST})
            for label, color, cell in (("∅ (no factor)", "#555555", "none"), ("R (route)", "#16427a", "R"),
                                       ("R+T+W (S|G)", "#eb6834", "R+T+W"))]
    matched.draw_forest(rows, OUT / "slide_factors.pdf", None, stats=list(matched.FOREST), per_row=3, width=6.0,
                        wrap=22, row_h=1.6)


def shapley():
    """Delta(none), the Shapley values of the three factors and Delta(F) on the nine quantities, `slide_shapley`."""
    d = read("decomposition")["levels"]["all"]
    rows = [(label, decomposition.ROW_COLORS[colour], {m: quote(get(d[m])) for m in matched.FOREST})
            for label, colour, get in (("Δ(∅)", "Δ(none)", lambda q: q["delta_none"]), ("φ_W", "φ_W", lambda q: q["phi"]["W"]),
                                       ("φ_T", "φ_T", lambda q: q["phi"]["T"]), ("φ_R", "φ_R", lambda q: q["phi"]["R"]),
                                       ("Δ(F)", "Δ(F)", lambda q: q["delta_F"]))]
    matched.draw_forest(rows, OUT / "slide_shapley.pdf", None, stats=list(matched.FOREST), per_row=3, width=6.4,
                        wrap=22, row_h=1.75)


def _finish(fig, ax, name):
    ax.grid(color="#eeeeee", lw=0.5)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, OUT / name)


def _bar(ax, y, q, color, **kw):
    ax.errorbar(q["value"], y, xerr=[[q["value"] - q["band"][0]], [q["band"][1] - q["value"]]], fmt="o", ms=5, color=color,
                elinewidth=1.8, capsize=2, **kw)


def fitted_h():
    """The fitted H (10 s to the end of the lags), one figure per design: of S, A, G to three end lags, each on its
    own population (fitted_h.py); the differences of the same route; g of the same flight against the window."""
    by_end = read("fitted_h_by_end")
    fig, ax = plt.subplots(figsize=(5.2, 2.6), constrained_layout=True)
    for k, (pop, short) in enumerate((("solo", "S"), ("group_solo", "A"), ("group", "G"))):
        for y, end in enumerate(by_end["ends"]):
            _bar(ax, y + 0.2 * (k - 1), quote(by_end["H"][short][str(end)]), COLORS[pop], label=short if y == 0 else None)
    ax.set(yticks=range(3), yticklabels=[f"to {end:,} s\n" + " · ".join(f"{k} {by_end['flights'][str(end)][k]:,}" for k in "SAG")
                                         for end in by_end["ends"]], ylim=(2.6, -0.6), xlabel="$H$, fitted from 10 s")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, "slide_fitted_h_raw.pdf")
    deltas = read("matched")["levels"]["all"]["deltas"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6), constrained_layout=True)
    for y, (k, m) in enumerate((k, m) for m in ("H (10 s to 1000 s)", "H (10 s to T)") for k in ("G - S|G", "A - S|A")):
        _bar(ax, y, quote(deltas[k][m]), "#eb6834" if k.startswith("G") else "#1baf7a")
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set(yticks=range(4), yticklabels=["G - S|G, to 1000 s", "A - S|A, to 1000 s", "G - S|G, to 2760 s", "A - S|A, to 2760 s"],
           ylim=(3.6, -0.6), xlabel="$\\Delta H$, fitted from 10 s")
    _finish(fig, ax, "slide_fitted_h_route.pdf")
    fig, ax = plt.subplots(figsize=(5.2, 2.6), constrained_layout=True)
    for y, w in enumerate((1260, 1800, 2400, 3000)):
        _bar(ax, y, quote(read(f"prepost_{w}")["differences"]["g"]["Delta H"]), "#8c2f0f")
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set(yticks=range(4), yticklabels=[f"window {w} s" for w in (1260, 1800, 2400, 3000)], ylim=(3.6, -0.6),
           xlabel="$\\Delta H$ of $g$, fitted from 10 s to 0.8 of the window")
    _finish(fig, ax, "slide_fitted_h_flight.pdf")


if __name__ == "__main__":
    raw_phases()
    raw_delta()
    raw_msd_fits()
    pools_delta()
    matched_deltas()
    matched_forest()
    prepost_delta()
    prepost_forest()
    factors()
    shapley()
    fitted_h()
