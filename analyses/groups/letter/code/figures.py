"""Figures of the letter, from the JSON of the analyses (raw.py, matched.py, prepost.py, all with --open).

    uv run python letter/code/figures.py [--run NAME]    # from analyses/groups; writes letter/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator

HERE = Path(__file__).resolve().parents[1]
GROUPS = HERE.parent
sys.path.insert(0, str(GROUPS))
from openflights import MIN_OPENING  # noqa: E402
from tamsd import COLORS  # noqa: E402

GROUP, ALONE, CONTROL = COLORS["group"], COLORS["group_solo"], "#8a8a8a"
WINDOW = 2400  # s, the windows of prepost.py --open quoted in the letter
DATASET = "1000"  # the dataset of raw.py and matched.py whose pieces give H_loc(1000 s)
STAGES = ("pre", "start", "end", "post")
XMIN_TICKS = (300, 500, 1000, 2000, 4000)  # s, labelled thresholds of the Hill panel

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "legend.fontsize": 7})


def crossing(curve):
    """The first lag after the last one where the curve is not above 0, as vs-solo/code/comparisons.py."""
    above = np.asarray(curve["value"]) > 0
    return curve["lags"][int(np.flatnonzero(~above)[-1]) + 1]


def draw(ax, lags, entry, color, label, dashed=False):
    lo, hi = entry["band"]
    ax.fill_between(lags, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(lags, entry["value"], color=color, lw=1.4, ls="--" if dashed else "-", label=label)


def persistence(folder, out):
    raw = json.loads((folder / "raw.json").read_text())["gaps"][DATASET]["H_loc"]
    matched = json.loads((folder / "matched.json").read_text())["levels"]["all"]["delta curves"]
    prepost = json.loads((folder / f"prepost_{WINDOW}.json").read_text())
    lags_pp, diff = prepost["lags"], prepost["differences"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True, constrained_layout=True)
    draw(axes[0], raw["G - S"]["lags"], raw["G - S"], GROUP, "G $-$ S")
    draw(axes[0], raw["A - S"]["lags"], raw["A - S"], ALONE, "A $-$ S", dashed=True)
    for gap, color, dashed in (("G - S|G", GROUP, False), ("A - S|A", ALONE, True)):
        main = matched[gap]["main"]
        draw(axes[1], main["lags"], main, color, gap.replace("-", "$-$").replace("|", "$|$"), dashed)
    draw(axes[2], lags_pp, diff["g"]["dH_loc"], GROUP, "$g$, group")
    draw(axes[2], lags_pp, diff["d w"]["dH_loc"], CONTROL, "drift, $d_w$", dashed=True)

    cross = crossing(raw["G - S"])
    for ax, title in zip(axes, ("(a) raw", "(b) same route", "(c) same flight")):
        ax.set_xscale("log")
        ax.set_xlim(10, 2000)
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.axvline(cross, color="#888888", lw=0.7, ls=":")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"$\tau$ (s)")
        ax.legend(loc="lower right", handlelength=1.8)
    axes[0].set_ylabel(r"$\Delta H_\mathrm{loc}(\tau)$")

    stages = prepost["curves"]["stages"]
    inset = axes[2].inset_axes([0.17, 0.63, 0.34, 0.28])
    values = [stages[s]["H_loc(1000 s)"] for s in STAGES]
    bands = np.array([stages[s]["H_loc(1000 s) band"] for s in STAGES]).T
    x = np.arange(len(STAGES))
    inset.errorbar(x, values, yerr=[np.subtract(values, bands[0]), np.subtract(bands[1], values)], color=GROUP,
                   lw=1, marker="o", ms=3, capsize=0)
    inset.set_xticks(x, STAGES, fontsize=6)
    inset.tick_params(labelsize=6, length=2)
    inset.set_title(r"$H_\mathrm{loc}(1000\ \mathrm{s})$", fontsize=6, pad=2)

    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"persistence.{ext}", dpi=300)
    plt.close(fig)


def tail(folder, out):
    t = json.loads((folder / "tail.json").read_text())
    colors = {"S": COLORS["solo"], "A": ALONE, "G": GROUP}
    xmin = t["transition"]["xmin of S"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)

    ax = axes[0]
    for pop in ("S", "A", "G"):
        s = t["survival"][pop]
        ax.fill_between(s["grid"], *s["band"], color=colors[pop], alpha=0.18, lw=0)
        ax.plot(s["grid"], s["value"], color=colors[pop], lw=1.4, ls="--" if pop == "A" else "-", label=pop)
    ax.set(xscale="log", yscale="log", xlim=(100, 1e4), ylim=(1e-5, 1.5), xlabel=r"$\tau$ (s)",
           ylabel=r"$P(\tau_\mathrm{transition} > \tau)$")
    ax.legend(loc="lower left")

    ax = axes[1]
    for pop in ("S", "A", "G"):
        c = t["transition"]["populations"][pop]["curve"]
        ax.fill_between(c["xmin"], *c["band"], color=colors[pop], alpha=0.18, lw=0)
        ax.plot(c["xmin"], c["value"], color=colors[pop], lw=1.4, ls="--" if pop == "A" else "-", label=pop)
    ax.set(xscale="log", xlim=(300, 4000), ylim=(0, 8), xlabel=r"$x_\mathrm{min}$ (s)", ylabel=r"Hill $\mu$")
    ax.set_xticks(XMIN_TICKS, [f"{x:g}" for x in XMIN_TICKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    g = t["transition"]["populations"]["G"]["curve"]
    top = ax.secondary_xaxis("top")
    shown = [i for i, x in enumerate(g["xmin"]) if x <= 4000][::4]
    top.set_xticks([g["xmin"][i] for i in shown], [f"{g['runs'][i]:,}" for i in shown], fontsize=6)
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("transitions of G above", fontsize=7)
    ax.legend(loc="lower right")

    ax = axes[2]
    classes, labels = ("0 s", "1000 s", "2000 s"), ("all", r"$\geq$1000 s", r"$\geq$2000 s")
    for k, pop in enumerate(("S", "A", "G")):
        d = [t["descriptors"][pop][c]["height lost (m/km)"] for c in classes]
        x = np.arange(len(classes)) + (k - 1) * 0.22
        med = [v["median"] for v in d]
        lo, hi = np.array([v["median band"] for v in d]).T
        ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], color=colors[pop], marker="o",
                    ms=3.5, lw=0, elinewidth=1.2, label=pop)
    ax.set_xticks(range(len(classes)), labels)
    ax.set(ylabel="height lost (m per km)", xlabel="transition duration")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.legend(loc="upper right")

    for ax, letter, x in zip(axes, "abc", (0.88, 0.03, 0.03)):
        ax.text(x, 0.96, f"({letter})", transform=ax.transAxes, va="top", fontsize=8)
    for ax in axes[:2]:
        ax.axvline(xmin, color="#888888", lw=0.7, ls=":")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"tail.{ext}", dpi=300)
    plt.close(fig)


def relative(g, pre, post):
    """g as a percentage of the level outside the group, the mean of the windows before and after."""
    level = (pre["value"] + post["value"]) / 2
    return 100 * g["value"] / level, [100 * b / level for b in g["band"]]


def forest(ax, rows, color_of):
    """Rows of (label, value, band) from the top, a marker with its band each."""
    for k, (label, value, band, key) in enumerate(rows):
        ax.errorbar(value, -k, xerr=[[value - band[0]], [band[1] - value]], color=color_of(key), marker="o",
                    ms=3.5, lw=0, elinewidth=1.2)
    ax.set_yticks([-k for k in range(len(rows))], [r[0] for r in rows])
    ax.axvline(0, color="#bbbbbb", lw=0.6)
    ax.set_ylim(-len(rows) + 0.4, 0.6)


def how(folder, out):
    groups_out = folder.parent
    h = json.loads((groups_out / "headings.json").read_text())
    prepost = json.loads((folder / f"prepost_{WINDOW}.json").read_text())
    terrain = json.loads((folder / "terrain_compare.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})

    ax = axes[0]
    tau = np.asarray(h["tau"], dtype=float)
    pairs = h["classes"]["pairs"]
    shades = plt.cm.Oranges(np.linspace(0.45, 0.95, len(pairs)))
    for c, shade in zip(pairs, shades):
        ell = c["l_median"]["value"]
        curve = c["heading"]["curve"]
        x = tau - ell
        ax.fill_between(x, [p["band"][0] for p in curve], [p["band"][1] for p in curve], color=shade, alpha=0.2, lw=0)
        ax.plot(x, [p["value"] for p in curve], color=shade, lw=1.3, label=rf"$\ell \approx {ell:.0f}$ s")
    ax.axvline(0, color="#888888", lw=0.7, ls=":")
    ax.set(xlabel=r"$\tau - \ell$ (s)", ylabel=r"heading correlation $C(\tau)$")
    ax.legend(loc="upper left", fontsize=6, handlelength=1.4)

    stats = (("searches that end in a climb (%)", "searches ending\nin a climb"), ("time in climb (%)", "time in climb"),
             ("climb rate (m/s)", "climb rate"), ("glide speed (km/h)", "glide speed"),
             ("height lost per km of glide (m/km)", "height lost\nper km"))
    stages, g = prepost["phase statistics"]["stages"], prepost["differences"]["g"]
    rows = [(label, *relative(g[key], stages["pre"][key], stages["post"][key]), "G") for key, label in stats]
    forest(axes[1], rows, lambda _: GROUP)
    axes[1].set_xlabel("change in the group (%)")

    raw4, flight = terrain["raw"]["four classes"], terrain["flight"]
    rows = [("raw, plains and hills", *quote(raw4["hills"]["H_loc(1000 s) gaps"]["G - S"]), "low"),
            ("raw, low mountains", *quote(raw4["low mountains"]["H_loc(1000 s) gaps"]["G - S"]), "mid"),
            ("raw, high mountains", *quote(raw4["high mountains"]["H_loc(1000 s) gaps"]["G - S"]), "high"),
            ("flight, plains and hills", *quote(flight["lowland"]["effects"]["g"]["H_loc(1000 s)"]), "low"),
            ("flight, mountains", *quote(flight["mountain"]["effects"]["g"]["H_loc(1000 s)"]), "high")]
    relief = dict(zip(("low", "mid", "high"), plt.cm.Greys(np.linspace(0.45, 0.95, 3))))
    forest(axes[2], rows, relief.get)
    axes[2].set_xlabel(r"$\Delta H_\mathrm{loc}(1000\ \mathrm{s})$")

    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", fontsize=8, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"how.{ext}", dpi=300)
    plt.close(fig)


def quote(entry):
    return entry["value"], entry["band"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="d120_eh500_ez300_th45_g120_q0.1")
    run = parser.parse_args().run
    folder = GROUPS / "out" / run / f"open_min{MIN_OPENING:g}"
    persistence(folder, HERE / "figures")
    tail(folder, HERE / "figures")
    how(folder, HERE / "figures")


if __name__ == "__main__":
    main()
