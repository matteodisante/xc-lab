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
import pauses as pa

OUT = Path(__file__).parent / "crests"
GROUPS = [(r, d) for r in dc.REGIONS for d in dc.DISCS]
RNAME, DNAME = {"alps": "Alps", "pyrenees": "Pyr"}, {"para": "Para", "hang": "Hang"}
RLONG, DLONG = {"alps": "Alps", "pyrenees": "Pyrenees"}, {"para": "paragliders", "hang": "hang gliders"}
COLOR = {"geo": "#0B0B0B", "model": "#2A78D6", "dyn": "#EB6834", "route": "#52514E"}
LABEL = {"geo": "crests", "model": "model", "dyn": "flights (dyn)", "route": "flights (route)"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.titlesize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False})


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
def fig_phases(res):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8), sharey=True)
    for j, (r, d) in enumerate(GROUPS):
        for i, kind in enumerate((pa.GLIDE, pa.PAUSE)):
            ax = axs[i, j]
            f = res["phases"].get(gkey(r, d), {}).get(kind)
            ax.set(xscale="log", yscale="log", ylim=(3e-4, 1.5))
            if i == 0:
                ax.set_title(f"{RLONG[r]}, {DLONG[d]}")
            if f is None:
                continue
            t, s = (np.array(a) for a in f["survival"])
            keep = np.unique(np.round(np.geomspace(1, len(t), 60)).astype(int) - 1)
            ax.plot(t[keep], s[keep], ".", color="0.35", ms=3, label="data")
            g = np.geomspace(t.min(), t.max(), 200)
            x0 = f["x0"]
            ax.plot(g, pa.lomax_survival(g - 0.5, f["mu"], f["tau0"], x0), color=COLOR["dyn"], lw=1, label="Lomax")
            ax.plot(g, pa.exp_survival(g - 0.5, f["theta"], x0), color=COLOR["model"], lw=1, ls="--", label="exponential")
            ax.set_xlim(t.min() * 0.8, t.max() * 1.5)
            if i == 1:
                ax.set_xlabel("duration t (s)")
            if j == 0:
                ax.set_ylabel(f"{kind}: P(duration $\\geq$ t)")
    axs[0, 0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "flights_phases")


def curves_figure(res, what, ylabel, name):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=what == "alpha")
    for ax, (r, d) in zip(axs.flat, [(r, d) for r in dc.REGIONS for d in dc.DISCS]):
        g = res["groups"].get(gkey(r, d))
        ax.set_title(f"{RLONG[r]}, {DLONG[d]}")
        ax.set_xscale("log")
        if what == "msd":
            ax.set_yscale("log")
        if g is None:
            ax.text(0.5, 0.5, "too few flights", transform=ax.transAxes, ha="center")
            continue
        for k in dc.CURVES:
            c = g["curves"][k]
            tau = np.array(c["tau"])
            if what == "msd":
                ball = (g["v"] * tau) ** 2  # ballistic MSD at speed v
                y, (lo, hi) = np.exp(c["lnm"]) / ball, np.exp(np.array(c["lnm_band"])) / ball
                ax.set_yscale("log")
            else:
                y, (lo, hi) = np.array(c["alpha"]), np.array(c["alpha_band"])
            ls = "--" if k == "route" else "-"
            ax.plot(tau, y, color=COLOR[k], lw=1, ls=ls, label=LABEL[k])
            ax.fill_between(tau, lo, hi, color=COLOR[k], alpha=0.2, lw=0)
            if what == "alpha":
                s = g["start"][k][0]
                if np.isfinite(s):
                    ax.plot([s], [np.interp(s, tau, y)], "|", color=COLOR[k], ms=8)
                    ax.plot([s, g["tau_max"]], [g["H"][k][0]] * 2, color=COLOR[k], lw=0.9, ls=(0, (4, 2)))  # the fitted H
        if what == "msd":
            ax.axhline(1, color="0.6", lw=0.6, ls=":")
            ax.set_ylim(0.03, 1.5)
        else:
            ax.axhline(dc.ALPHA_END, color="0.6", lw=0.6, ls=":")
            ax.axvline(g["tau_max"], color="0.8", lw=0.6)
            ax.set_ylim(0.5, 1.05)
        ax.set_xlim(1, 1.2e4)
        ax.set_xlabel(r"lag $\tau$ (s)")
        ax.set_ylabel(ylabel)
    axs[0, 0].legend(loc="lower left" if what == "alpha" else "upper left", frameon=False)
    fig.tight_layout()
    save(fig, name)


def fig_decomposition(res):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    terms = [("geometry", "geometry", COLOR["geo"]), ("intermittency", "intermittency", COLOR["model"]), ("other", "remainder", COLOR["dyn"])]
    for i, (r, d) in enumerate(GROUPS):
        g = res["groups"].get(gkey(r, d))
        if g is None:
            continue
        for k, (key, label, c) in enumerate(terms):
            v, lo, hi = g["decomposition"][key]
            x = i + 0.25 * (k - 1)
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="o", color=c, ms=4, lw=1, capsize=2, label=label if i == 0 else None,
                        alpha=1 if g["n_flights"] >= dc.MIN_FLIGHTS else 0.35)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylim(-40, 90)
    for i, (r_, d_) in enumerate(GROUPS):
        if res["groups"].get(gkey(r_, d_)) is None:
            ax.text(i, 0, "too few\nflights", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4), [f"{RLONG[r]}\n{DLONG[d]}" for r, d in GROUPS])
    ax.set_ylabel(r"share of $1-H_{\mathrm{dyn}}$ (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    save(fig, "flights_decomposition")


# macros and tables
def macros(res, cache):
    m = Macros()
    m.add("flAlphaEnd", f"{dc.ALPHA_END:g}")
    m.add("flFloorM", f"{dc.FLOOR_M:g}")
    m.add("flNBoot", f"{dc.N_BOOT}")
    m.add("flNReal", f"{dc.N_REAL}")
    m.add("flMinFlights", f"{dc.MIN_FLIGHTS}")
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
    rows = []
    for r, d in GROUPS:
        g = res["groups"].get(gkey(r, d))
        if g is None:
            rows.append([gl(r, d)] + ["--"] * 5)
            continue
        rows.append([gl(r, d), str(g["n_flights"])] + [fmt_ci(g["H"][k]) for k in dc.CURVES])
    table("h", r"group & flights & $\Hgeo$ & $\Hmod$ & $\Hdyn$ & $\Hroute$", [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in rows], "lrllll")
    rows = [[gl(r, d), str(g["n_flights"]), fmt_ci(g["h10"]["H"]), f"{g['h10']['last_lag']:.0f}"]
            for r, d in GROUPS if (g := res["groups"].get(gkey(r, d))) is not None]
    table("h10", r"group & flights & $H$ & last lag (s)", rows, "lrll")
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
    curves_figure(res, "msd", r"MSD$/(v\tau)^2$", "flights_msd")
    curves_figure(res, "alpha", r"local slope $\alpha(\tau)$", "flights_alpha")
    fig_decomposition(res)
    macros(res, cache)
    tables(res)
    if (cache / "rawcheck.json").exists():
        rawcheck_table(json.loads((cache / "rawcheck.json").read_text()))
