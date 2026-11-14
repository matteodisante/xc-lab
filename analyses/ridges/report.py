"""Figures, tables and macros of the flight part of the crest report, from decompose.py's decomposition.json.

Writes in crests/: figures/flights_{phases,msd,alpha,decomposition}.{pdf,png}, generated/flights_macros.tex (every
number the text quotes) and generated/flights_table_*.tex (the tabulars of the tables).

    uv run python report.py [--cache DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import decompose as dc
import figstyle as fs
import pauses as pa

OUT = Path(__file__).parent / "crests"
GROUPS = [(r, d) for r in dc.REGIONS for d in dc.DISCS]
RNAME, DNAME = {"alps": "Alps", "pyrenees": "Pyr"}, {"para": "Para", "hang": "Hang"}
RLONG, DLONG = {"alps": "Alps", "pyrenees": "Pyrenees"}, {"para": "paragliders", "hang": "hang gliders"}
COLOR, LABEL = fs.KIND_COLOR, fs.KIND_LABEL
ALL = dc.CURVES + dc.NULLS
fs.apply()


def gkey(r, d):
    return f"{r}/{d}"


def gname(r, d):
    return RNAME[r] + DNAME[d]


class Macros:
    def __init__(self):
        self.lines = []

    def add(self, name, value):
        assert name.isalpha(), name
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def ci(self, name, x, f, scale=1.0):
        v, lo, hi = (np.asarray(x, float) * scale).tolist() if x is not None else (np.nan,) * 3
        for suffix, x in (("", v), ("Lo", lo), ("Hi", hi)):
            self.add(name + suffix, "--" if np.isnan(x) else num(f, x))


def num(f, x):
    """Formatted number with a true minus sign. A Lomax fit at the bound of the search (mu near 3000, tau0 above
    1e5 s) is the exponential limit: shown as infinity."""
    if f == "{:.2f}" and x >= 1000:
        return r"$\infty$"
    if f == "{:.0f}" and x >= 1e5:
        return r"$\infty$"
    s = f.format(x)
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


def fmt_ci(x, f="{:.3f}", scale=1.0):
    if x is None or np.isnan(x[0]):
        return "--"
    v, lo, hi = np.asarray(x, float) * scale
    if num(f, v) == r"$\infty$":
        return num(f, v)
    return f"{num(f, v)} ({num(f, lo)}{' to ' if lo < 0 else '--'}{num(f, hi)})"


def save(fig, name):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# figures
GROUP_LABEL = {(r, d): f"{RLONG[r]}, {DLONG[d]}" for r, d in GROUPS}
SHOWN = [(r, d) for r, d in GROUPS if (r, d) != ("pyrenees", "hang")]  # the Pyrenees hang gliders are too few to draw


def fig_phases(res):
    """Durations of glides and climbs: the survival of each group with the Lomax and the exponential fit."""
    fig, axs = plt.subplots(2, 3, figsize=(fs.TEXT_WIDTH_IN, 3.6), sharey=True)
    for j, (r, d) in enumerate(SHOWN):
        for i, kind in enumerate((pa.GLIDE, pa.PAUSE)):
            ax = axs[i, j]
            f = res["phases"].get(gkey(r, d), {}).get(kind)
            ax.set(xscale="log", yscale="log", ylim=(3e-4, 1.6), xlim=(1, 6e3))
            fs.letter(ax, "abcdef"[3 * i + j])
            ax.text(0.03, 0.03, f"{GROUP_LABEL[(r, d)]}\n{'glides' if i == 0 else 'climbs'}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
            t, s = (np.array(a) for a in f["survival"])
            keep = np.unique(np.round(np.geomspace(1, len(t), 60)).astype(int) - 1)
            g = np.geomspace(t.min(), t.max(), 200)
            ax.plot(t[keep], s[keep], ".", color="#8C8C8C", ms=3, label="data", zorder=1)
            ax.plot(g, pa.lomax_survival(g - 0.5, f["mu"], f["tau0"], f["x0"]), color="black", lw=1.2, label="Lomax", zorder=2)
            ax.plot(g, pa.exp_survival(g - 0.5, f["theta"], f["x0"]), color="black", lw=1.2, ls=":", label="exponential", zorder=3)
            if i == 1:
                ax.set_xlabel("duration $t$ [s]")
            if j == 0:
                ax.set_ylabel("$P(\\mathrm{duration}\\geq t)$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "flights_phases")


SHOW_BEYOND = 1.15  # curves are drawn up to this multiple of the last lag of the fit


def shade_after(ax, tau_b, right):
    """Grey beyond the last lag of the fit."""
    ax.axvspan(tau_b, right, color="#EDEDED", lw=0, zorder=0)


def curves_figure(res, what, name):
    """MSD over the ballistic value against the lag, one panel per group; the fourth panel holds the legend."""
    right = SHOW_BEYOND * max(res["groups"][gkey(r, d)]["tau_max"] for r, d in SHOWN)
    fig, axs = plt.subplots(2, 2, figsize=(fs.TEXT_WIDTH_IN, 5.0), sharex=True)
    for k_, (ax, (r, d)) in enumerate(zip(axs.flat, SHOWN)):
        g = res["groups"].get(gkey(r, d))
        ax.set_xscale("log")
        ax.set_yscale("log")
        fs.letter(ax, "abc"[k_])
        ax.text(0.03, 0.03, GROUP_LABEL[(r, d)], transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        shade_after(ax, g["tau_max"], right)
        ax.axhline(1, color=fs.GUIDE, lw=0.6, zorder=0)
        ax.set_ylim(0.1, 1.3)
        for k in ALL:
            c = g["curves"][k]
            tau = np.array(c["tau"])
            ball = (g["v"] * tau) ** 2
            y, (lo, hi) = np.exp(c["lnm"]) / ball, np.exp(np.array(c["lnm_band"])) / ball
            keep = tau <= SHOW_BEYOND * g["tau_max"]
            tau, y, lo, hi = tau[keep], y[keep], lo[keep], hi[keep]
            ax.fill_between(tau, lo, hi, color=COLOR[k], alpha=0.15, lw=0, zorder=1)
            ax.plot(tau, y, color=COLOR[k], lw=fs.LW, zorder=2)
        ax.set_xlim(1, right)
    for ax in (axs[0, 1], axs[1, 0]):
        ax.set_xlabel("lag $\\tau$ [s]")
        ax.tick_params(labelbottom=True)
    for ax in axs[:, 0]:
        ax.set_ylabel("MSD$\\,/\\,(v\\tau)^2$")
    leg = axs[1, 1]
    leg.axis("off")
    handles = [plt.Line2D([], [], color=COLOR[k], lw=fs.LW, label=LABEL[k]) for k in ALL]
    handles += [plt.Line2D([], [], color=fs.GUIDE, lw=0.8, label="ballistic motion at $v$")]
    leg.legend(handles=handles, loc="center", frameon=False)
    fig.tight_layout()
    save(fig, name)


ROWS = [("comparison", ("geo", "model", "dyn", "route"), (0.75, 1.02), (1.0,)),
        ("null models", ("model", "dyn", "straight", "random"), (0.45, 1.02), (1.0, 0.5))]


def alpha_figure(res, name="flights_alpha"):
    """Local slope against the lag: a row for the comparison of crests, model and flights, a row for the null models."""
    right = SHOW_BEYOND * max(res["groups"][gkey(r, d)]["tau_max"] for r, d in SHOWN)
    fig, axs = plt.subplots(2, 3, figsize=(fs.TEXT_WIDTH_IN, 4.4), sharex=True, sharey="row")
    for i, (_, kinds, ylim, guides) in enumerate(ROWS):
        for j, (r, d) in enumerate(SHOWN):
            ax, g = axs[i, j], res["groups"][gkey(r, d)]
            ax.set_xscale("log")
            fs.letter(ax, "abcdef"[3 * i + j])
            shade_after(ax, g["tau_max"], right)
            for level in guides:
                ax.axhline(level, color=fs.GUIDE, lw=0.6, zorder=0)
            for k in kinds:
                c = g["curves"][k]
                tau = np.array(c["tau"])
                y, (lo, hi) = np.array(c["alpha"]), np.array(c["alpha_band"])
                keep = tau <= SHOW_BEYOND * g["tau_max"]
                tau, y, lo, hi = tau[keep], y[keep], lo[keep], hi[keep]
                ax.fill_between(tau, lo, hi, color=COLOR[k], alpha=0.15, lw=0, zorder=1)
                ax.plot(tau, y, color=COLOR[k], lw=fs.LW, zorder=2)
                s = g["start"][k][0]
                if np.isfinite(s):
                    ax.plot([s, g["tau_max"]], [g["H"][k][0]] * 2, color=COLOR[k], lw=1.0, ls=fs.FIT_DASH, zorder=3)
            ax.set_xlim(1, right)
            ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(GROUP_LABEL[(r, d)], fontsize=8)
            if i == 1:
                ax.set_xlabel("lag $\\tau$ [s]")
            if j == 0:
                ax.set_ylabel("local slope $\\alpha(\\tau)$")
    handles = [plt.Line2D([], [], color=COLOR[k], lw=fs.LW, label=LABEL[k]) for k in ALL]
    handles += [plt.Line2D([], [], color="black", lw=1.0, ls=fs.FIT_DASH, label="fitted $H$ (same colour)"),
                plt.Line2D([], [], color=fs.GUIDE, lw=0.8, label="$\\alpha = 1$ and $0.5$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, name)


def fig_decomposition(res):
    """1 - H down to H_dyn in three steps, per group, with each step as a percentage of 1 - H_dyn."""
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, 3.2))
    steps = [("geometry", "geo", "geometry"), ("intermittency", "model", "intermittency"), ("other", None, "remainder")]
    color = {"geo": COLOR["geo"], "model": COLOR["model"], None: "#F0E442"}  # yellow: no curve has it
    labels = []
    for i, (r, d) in enumerate(SHOWN):
        g = res["groups"][gkey(r, d)]
        x0 = 5.4 * i
        levels = [1.0, g["H"]["geo"][0], g["H"]["model"][0], g["H"]["dyn"][0]]
        for j, (term, kind, word) in enumerate(steps):
            a, b = levels[j], levels[j + 1]
            ax.add_patch(plt.Rectangle((x0 + 1.4 * j, min(a, b)), 1.0, abs(a - b), facecolor=color[kind], edgecolor="black" if kind is None else "none", lw=0.5))
            ax.plot([x0 + 1.4 * j - 0.2 if j else x0 - 0.2, x0 + 1.4 * j + 1.6], [a, a], color=fs.GUIDE, lw=0.6, zorder=0)
            pct = f"{g['decomposition'][term][0]:.0f}%".replace("-", "\u2212")
            below = j == 1  # the label of the middle step goes under its bar, clear of the neighbours
            ax.text(x0 + 1.4 * j + 0.5, (min(a, b) - 0.002) if below else (max(a, b) + 0.002), f"{word}\n{pct}", ha="center",
                    va="top" if below else "bottom", fontsize=8, zorder=5)
        ax.plot([x0 + 3.9, x0 + 4.6], [levels[3]] * 2, color=COLOR["dyn"], lw=1.4)
        labels.append((x0 + 2.0, GROUP_LABEL[(r, d)].replace(", ", "\n") + f"\n$1-H_{{dyn}}$ = {g['decomposition']['total'][0]:.3f}"))
    ax.axhline(1.0, color=fs.GUIDE, lw=0.6, zorder=0)
    ax.set_xticks([x for x, _ in labels], [t for _, t in labels])
    ax.set_xlim(-0.6, 5.4 * len(SHOWN) - 0.4)
    ax.set_ylim(0.85, 1.03)
    ax.set_ylabel("$H$")
    fig.tight_layout()
    save(fig, "flights_decomposition")


def fig_models():
    """The four models on the same sequence of glides and pauses (speed 1, lengths in glide time)."""
    glides, pauses = [2.4, 1.6, 3.0, 2.0], [1.4, 0.7, 1.9]
    arcs = np.r_[0, np.cumsum(glides)]
    xy = np.zeros((4000, 2))
    xy[:, 0] = np.linspace(0, 6.2, 4000)
    xy[:, 1] = 1.0 * np.sin(1.0 * xy[:, 0]) + 0.45 * np.sin(2.6 * xy[:, 0] + 1)
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    xy, arc = xy * arcs[-1] / arc[-1], arc * arcs[-1] / arc[-1]
    on_crest = lambda a: np.c_[np.interp(a, arc, xy[:, 0]), np.interp(a, arc, xy[:, 1])]
    angles = np.array([0.5, 2.1, -0.4, 1.3])
    panels = [("crests", "geo", "keeps: geometry"),
              ("crest model", "model", "keeps: geometry and intermittency"),
              ("straight", "straight", "keeps: intermittency only (no turning)"),
              ("random", "random", "keeps: intermittency and random turning")]
    fig, axs = plt.subplots(2, 2, figsize=(fs.TEXT_WIDTH_IN, 3.9))
    for ax, (title, kind, keeps) in zip(axs.flat, panels):
        pos = np.zeros(2) if kind != "geo" and kind != "model" else on_crest(0.0)[0]
        start = pos.copy()
        for i, gl in enumerate(glides):
            if kind in ("geo", "model"):
                seg = on_crest(np.linspace(arcs[i], arcs[i + 1], 60))
            elif kind == "straight":
                seg = np.c_[np.linspace(arcs[i], arcs[i + 1], 60), np.zeros(60)]
            else:
                seg = pos + np.linspace(0, 1, 60)[:, None] * gl * np.array([np.cos(angles[i]), np.sin(angles[i])])
            ax.plot(seg[:, 0], seg[:, 1], color=COLOR[kind], lw=1.8, solid_capstyle="round", zorder=2)
            pos = seg[-1]
            if kind != "geo" and i < len(pauses):
                ax.scatter(*pos, s=60 * pauses[i], color=COLOR[kind], edgecolor="white", lw=0.8, zorder=3)
        if kind == "geo":  # constant speed: a dot at every equal step of time
            tick = on_crest(np.arange(0.5, arcs[-1] - 1e-9, 0.5))
            ax.scatter(tick[:, 0], tick[:, 1], s=9, color="white", edgecolor=COLOR[kind], lw=0.7, zorder=3)
        ax.scatter(*start, s=26, marker="s", color="black", zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-0.8, 9.6)
        ax.set_ylim(-2.4, 4.4)
        ax.axis("off")
        ax.text(0.0, 1.0, title, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
        ax.text(0.0, 0.0, keeps, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    for ax, s_ in zip(axs.flat, "abcd"):
        fs.letter(ax, s_, x=-0.02, y=1.0)
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    save(fig, "flights_models")


# macros and tables
def macros(res, cache):
    m = Macros()
    m.add("flAlphaEnd", f"{dc.ALPHA_END:g}")
    m.add("flFloorM", f"{dc.FLOOR_M:g}")
    m.add("flNBoot", f"{dc.N_BOOT}")
    m.add("flNReal", f"{dc.N_REAL}")
    m.add("flMinFlights", f"{dc.MIN_FLIGHTS}")
    m.add("flMinRows", f"{dc.MIN_ROWS}")
    t = pd.read_csv(cache / "flights.csv", dtype={"flight": str})
    for disc, D in (("paragliders", "Para"), ("hang_gliders", "Hang")):
        s = t[t["discipline"] == disc]
        m.add("flCand" + D, f"{len(s):,}".replace(",", r"\,"))
        for status, S in (("ok", "Ok"), ("dt", "Dt"), ("opening", "Opening"), ("region", "Region")):
            m.add(f"flStat{S}{D}", f"{(s['status'] == status).sum():,}".replace(",", r"\,"))
        for r, N in (("alps", "Alps"), ("pyrenees", "Pyr")):
            m.add(f"flSel{N}{D}", f"{((s['status'] == 'ok') & (s['region'] == r)).sum():,}".replace(",", r"\,"))
    for r, d in GROUPS:
        n, key = gname(r, d), gkey(r, d)
        ph, sp, g = res["phases"].get(key, {}), res["speed"].get(key), res["groups"].get(key)
        m.add("flNOneHz" + n, str(ph.get("n_flights_1hz", 0)))
        m.add("flNPhaseFlights" + n, str(ph.get("n_flights_with_phases", 0)))
        if sp:
            m.ci("flVGlide" + n, [sp["v"], *sp["v_ci"]], "{:.1f}")
            m.ci("flVGlideKmh" + n, [sp["v"], *sp["v_ci"]], "{:.0f}", 3.6)
        for kind, K in ((pa.GLIDE, "Glide"), (pa.PAUSE, "Pause")):
            f = ph.get(kind)
            if f is None:
                continue
            m.add(f"fl{K}N" + n, f"{f['n']:,}".replace(",", r"\,"))
            m.add(f"fl{K}Mean" + n, f"{f['mean']:.0f}")
            m.ci(f"fl{K}Mu" + n, [f["mu"], *f["mu_ci"]], "{:.2f}")
            m.ci(f"fl{K}Tau" + n, [f["tau0"], *f["tau0_ci"]], "{:.0f}")
            m.ci(f"fl{K}Theta" + n, [f["theta"], *f["theta_ci"]], "{:.0f}")
            m.ci(f"fl{K}MeanFit" + n, [f["mean_lomax"], *f["mean_lomax_ci"]], "{:.0f}")
            m.add(f"fl{K}Aic" + n, num("{:.0f}", f["aic_exp"] - f["aic_lomax"]))
            m.add(f"fl{K}Better" + n, f"{100 * f['lomax_better']:.0f}")
        if g:
            m.add("flN" + n, str(g["n_flights"]))
            m.add("flNPaths" + n, str(g["n_paths"]))
            m.add("flTauMax" + n, f"{g['tau_max']:.0f}")
            m.add("flTauFloor" + n, f"{g['tau_floor']:.0f}")
            for k, K in (("geo", "Geo"), ("model", "Mod"), ("dyn", "Dyn"), ("route", "Route")):
                m.ci("flH" + K + n, g["H"][k], "{:.3f}")
                m.ci("flStart" + K + n, g["start"][k], "{:.0f}")
            m.ci("flHTen" + n, g["h10"]["H"], "{:.3f}")
            m.add("flHTenLast" + n, f"{g['h10']['last_lag']:.0f}")
            for k, K in (("straight", "Straight"), ("random", "Random")):
                m.ci("flH" + K + n, g["H"][k], "{:.3f}")
                m.ci(f"flNullDH{K}" + n, g["null"][k]["dH"], "{:.3f}")
                m.ci(f"flNullGap{K}" + n, g["null"][k]["gap"], "{:.0f}")
            m.ci("flHModOld" + n, g["H_old"], "{:.3f}")
            m.ci("flPctInterOld" + n, g["decomposition_old"]["intermittency"], "{:.0f}")
            m.ci("flPctOtherOld" + n, g["decomposition_old"]["other"], "{:.0f}")
            m.ci("flTotal" + n, g["decomposition"]["total"], "{:.3f}")
            for key, K in (("geometry", "Geometry"), ("intermittency", "Inter"), ("other", "Other")):
                m.ci("flPct" + K + n, g["decomposition"][key], "{:.0f}")
                m.ci("flPctRoute" + K + n, g["decomposition_route"][key], "{:.0f}")
            for wn, w in zip("ABCD", g["window"]["geo"]):  # windows of s: from 0.1 km, 0.1-1, 1-10, beyond 10 km
                for k, K in (("geo", "Geo"), ("model", "Mod"), ("dyn", "Dyn"), ("route", "Route")):
                    m.ci(f"flW{K}{wn}" + n, g["window"][k][w], "{:.3f}")
    (OUT / "generated").mkdir(parents=True, exist_ok=True)
    text = "% Generated by report.py: do not edit.\n\\providecommand{\\Hmod}{H_{\\mathrm{mod}}}\n\\providecommand{\\Hdyn}{H_{\\mathrm{dyn}}}\n"
    text += "\\providecommand{\\Hroute}{H_{\\mathrm{route}}}\n" + "\n".join(m.lines) + "\n"
    (OUT / "generated" / "flights_macros.tex").write_text(text)


def table(name, header, rows, align):
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    (OUT / "generated" / f"flights_table_{name}.tex").write_text(
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def rawcheck_table(check):
    rows = []
    for k, label in (("dyn", "dyn"), ("route", "route")):
        for w, x in check["h"][k].items():
            rows.append([label, "rule" if w == "rule" else w, fmt_ci(x["raw"]), fmt_ci(x["clean"]), fmt_ci(x["raw_minus_clean"])])
    table("rawcheck", r"curve & window of $s$ (km) & raw IGC & cleaned fixes & raw $-$ cleaned", rows, "llccc")
    m = Macros()
    for key in ("n_in_sample", "n_clean", "n_raw", "n_both"):
        m.add("flRaw" + "".join(p.capitalize() for p in key.split("_")), str(check[key]))
    m.ci("flRawDynRule", check["h"]["dyn"]["rule"]["raw_minus_clean"], "{:.3f}")
    m.ci("flRawRouteRule", check["h"]["route"]["rule"]["raw_minus_clean"], "{:.3f}")
    with open(OUT / "generated" / "flights_macros.tex", "a") as f:
        f.write("\n".join(m.lines) + "\n")


def tables(res):
    gl = lambda r, d: f"{RLONG[r]}, {DLONG[d]}"
    rows = []
    for r, d in GROUPS:
        ph, sp = res["phases"].get(gkey(r, d), {}), res["speed"].get(gkey(r, d))
        rows.append([gl(r, d), str(ph.get("n_flights_1hz", 0)), str(ph.get(pa.GLIDE, {}).get("n", 0)), str(ph.get(pa.PAUSE, {}).get("n", 0)),
                     fmt_ci([sp["v"], *sp["v_ci"]], "{:.1f}") if sp and ph.get(pa.GLIDE) else "--"])
    table("phases", "group & flights & glides & pauses & $v$ (m/s)", rows, "lrrrl")
    rows = []
    for r, d in GROUPS:
        for kind in (pa.GLIDE, pa.PAUSE):
            f = res["phases"].get(gkey(r, d), {}).get(kind)
            if f is None:
                continue
            rows.append([gl(r, d), kind, f"{f['mean']:.0f}", fmt_ci([f["mu"], *f["mu_ci"]], "{:.2f}"), fmt_ci([f["tau0"], *f["tau0_ci"]], "{:.0f}"),
                         fmt_ci([f["theta"], *f["theta_ci"]], "{:.0f}"), num("{:.0f}", f["aic_exp"] - f["aic_lomax"])])
    table("fits", r"group & phase & mean (s) & $\mu$ & $\tau_0$ (s) & $\theta$ (s) & $\Delta\mathrm{AIC}$", rows, "llrlllr")
    shown = [(r, d) for r, d in GROUPS if res["groups"].get(gkey(r, d)) is not None]
    gs = [res["groups"][gkey(r, d)] for r, d in shown]
    head = "curve & " + " & ".join(gl(r, d) for r, d in shown)
    names = (("geo", "crests"), ("model", "crest model"), ("dyn", "flights"), ("route", "flights, circles removed"), ("straight", "straight"), ("random", "random"))
    rows = [["number of flights"] + [str(g["n_flights"]) for g in gs]] + [[label] + [fmt_ci(g["H"][k]) for g in gs] for k, label in names]
    table("h", head, rows, "l" + "l" * len(gs))
    rows = []
    for k, label in (("straight", "straight"), ("random", "random")):
        rows += [[f"{label}: $H$"] + [fmt_ci(g["H"][k]) for g in gs], [f"{label}: $\\Delta H$"] + [fmt_ci(g["null"][k]["dH"]) for g in gs],
                 [f"{label}: gap (\\%)"] + [fmt_ci(g["null"][k]["gap"], "{:.0f}") for g in gs]]
    table("nulls", "null model & " + " & ".join(gl(r, d) for r, d in shown), rows, "l" + "l" * len(gs))
    rows = []
    for r, d in GROUPS:
        g = res["groups"].get(gkey(r, d))
        if g is None:
            continue
        dd = g["decomposition"]
        rows.append([gl(r, d), str(g["n_flights"]), f"{dd['total'][0]:.3f}"] + [fmt_ci(dd[k], "{:.0f}") for k in ("geometry", "intermittency", "other")])
    table("decomposition", r"group & flights & $1-\Hdyn$ & geometry (\%) & intermittency (\%) & remainder (\%)", rows, "lrrlll")
    rows = []
    for r, d in GROUPS:
        g = res["groups"].get(gkey(r, d))
        if g is None:
            continue
        for w in g["window"]["geo"]:
            rows.append([gl(r, d), w] + [fmt_ci(g["window"][k][w]) for k in dc.CURVES])
    table("windows", r"group & window of $s$ (km) & $\Hgeo$ & $\Hmod$ & $\Hdyn$ & $\Hroute$", rows, "llllll")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=str(OUT), help="the crests folder that receives figures/ and generated/")
    a = ap.parse_args()
    OUT = Path(a.out)
    cache = Path(a.cache) if a.cache else dc.newest_cache()
    res = json.loads((cache / "decomposition.json").read_text())
    (OUT / "generated").mkdir(parents=True, exist_ok=True)
    fig_phases(res)
    curves_figure(res, "msd", "flights_msd")
    alpha_figure(res)
    fig_decomposition(res)
    fig_models()
    macros(res, cache)
    tables(res)
    if (cache / "rawcheck.json").exists():
        rawcheck_table(json.loads((cache / "rawcheck.json").read_text()))
