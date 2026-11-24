"""The length of the glides (km): the chord of a glide, from its first to its last fix, per population S, A, G and ground
class, as phase_gaps.py does for the durations.

The runs, their population and their class are those of phase_gaps.py (a run belongs whole to the piece that holds its
middle; the draws are the shared two-way draws of contrasts.py). The glides are the runs of phase `transition` whose
ends are fixes of the flight (phase_terms.py: chord_m > 0), trimmed as the durations are (durations.trim). Per class and
"all" and per population: the number of glides, their mean and median length, the gaps G - A and A - S of both, the
survival P(X > x) and the Hill exponent mu above the common Kolmogorov-Smirnov threshold of S, A, G (tail_ks.py,
chosen again in every draw) with the gap mu_G - mu_A; bands are those of twoway.band. The scan of tail_ks.py runs on
the lengths in metres (one bin per metre below 200 m); the outputs are in km. mu does not depend on the unit.

    uv run python glide_length.py --declared-open --dt5 [--draws N]   # writes glide_length[_drawsN].json, .pdf and .png
    uv run python glide_length.py --declared-open --dt5 --draw         # the figure, from the JSON
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import longlags
import phase_gaps
import phase_terms
import raw
import tail
import twoway
from phase_gaps import CLASS_KEYS, DRAWS, POPS, rows_of, stratum, tail_statistics
from tamsd import COLORS, N_BOOT, save

sys.path.insert(0, str(Path(__file__).parents[1] / "phases"))
from durations import trim  # noqa: E402

NAME = "glide_length" + (f"_draws{DRAWS}" if DRAWS != N_BOOT else "")
KM = 1000.0
GAPS = {"G - A": ("G", "A"), "A - S": ("A", "S")}


def glides(rows):
    """The lengths (m, trimmed) and the flights of the glides among `rows`."""
    r = rows[(rows["phase"] == "transition") & (rows["glide_t"] > 0) & (rows["chord_m"] > 0)]
    x = r["chord_m"].to_numpy(float)
    keep = np.isin(x, trim(x)) if len(x) else np.zeros(0, bool)
    return x[keep], r["flight"].to_numpy()[keep]


def location(x, cell, counts):
    """The mean and the median of `x` (cells `cell`) as arrays over [point, draws]: each glide weighs the count of its
    cell (1 for the point)."""
    n_cells = counts.shape[1]
    total, size = np.bincount(cell, x, minlength=n_cells), np.bincount(cell, minlength=n_cells)
    mean = np.r_[total.sum() / size.sum(), (counts @ total) / (counts @ size)]
    median = np.r_[tail.median_draws(x, cell, np.ones((1, n_cells), np.uint8)), tail.median_draws(x, cell, counts)]
    return mean, median


def statistics(runs, code, counts):
    """Per class: the lengths of S, A, G (number, mean and median in km, gaps of GAPS) and the tail of tail_statistics."""
    out = {}
    for key in CLASS_KEYS:
        mine = stratum(runs, key)
        by_pop = {POPS[k]: glides(mine[mine["pop"] == i]) for i, k in enumerate(POPS)}
        loc = {pop: location(x, code.reindex(f).to_numpy(), counts) for pop, (x, f) in by_pop.items()}
        out[key] = {"glides": {pop: len(x) for pop, (x, _) in by_pop.items()}}
        for j, stat in enumerate(("mean", "median")):
            v = {pop: m[j] / KM for pop, m in loc.items()}
            out[key][stat] = {"populations": {pop: rows_of(a) for pop, a in v.items()},
                              "gaps": {gap: rows_of(v[a] - v[b]) for gap, (a, b) in GAPS.items()}}
        t = tail_statistics(by_pop, code, counts)
        out[key]["tail"] = None if t is None else in_km(t)
        print(key, out[key]["glides"], None if t is None else round(out[key]["tail"]["xmin"]["common"], 2), flush=True)
    return out


def in_km(t):
    """The tail dictionary of tail_statistics with the lengths (thresholds, grids) in km."""
    t["xmin"] = {"common": t["xmin"]["common"] / KM, "common_5_95": [v / KM for v in t["xmin"]["common_5_95"]],
                 "own": {k: v / KM for k, v in t["xmin"]["own"].items()},
                 "own_5_95": {k: [x / KM for x in v] for k, v in t["xmin"]["own_5_95"].items()}}
    for s in t["survival"].values():
        s["grid"] = [g / KM for g in s["grid"]]
    for h in t["hill"].values():
        h["xmin"] = [g / KM for g in h["xmin"]]
    return t


# ---- the table and the figure ----

def q(v, d=2):
    return f"{v['value']:.{d}f} [{v['band'][0]:.{d}f}, {v['band'][1]:.{d}f}]"


def resolved(v):
    return "yes" if v["band"][0] > 0 or v["band"][1] < 0 else "no"


def table(result):
    """Print, per class, the glides of A and G, the mean and median length (km) of A and G and their gap, mu of S, A, G and mu_G - mu_A."""
    for stat in ("mean", "median"):
        print(f"\n{stat} length (km), 90% bands")
        print(f"{'class':15s}{'N_A':>8s}{'N_G':>8s}  {'A':24s}{'G':24s}{'G - A':26s}resolved")
        for c in CLASS_KEYS:
            r = result["lengths"][c]
            p, g = r[stat]["populations"], r[stat]["gaps"]["G - A"]
            print(f"{c:15s}{r['glides']['A']:8d}{r['glides']['G']:8d}  {q(p['A']):24s}{q(p['G']):24s}{q(g, 3):26s}{resolved(g)}")
    print("\nHill exponent above the common threshold")
    print(f"{'class':15s}{'xmin km':>8s}  {'S':20s}{'A':20s}{'G':20s}{'mu_G - mu_A':24s}resolved")
    for c in CLASS_KEYS:
        t = result["lengths"][c]["tail"]
        if t is None:
            print(f"{c:15s}  no tail")
            continue
        g = t["gaps"]["G - A"]
        print(f"{c:15s}{t['xmin']['common']:8.2f}  {q(t['mu']['S']):20s}{q(t['mu']['A']):20s}{q(t['mu']['G']):20s}{q(g):24s}{resolved(g)}")


def survival_panel(ax, t, title, legend=False):
    """The survival of the lengths of S, A, G with its band and the common threshold (dotted)."""
    for pop, short in POPS.items():
        s = t["survival"][short]
        g, v, b = np.array(s["grid"]), np.array(s["value"]), np.array(s["band"])
        ok = v > 0
        ax.fill_between(g[ok], np.maximum(b[0][ok], 1e-9), b[1][ok], color=COLORS[pop], alpha=0.3, lw=0)
        ax.plot(g[ok], v[ok], color=COLORS[pop], lw=1.5, label=short)
    ax.axvline(t["xmin"]["common"], color="#888888", lw=0.8, ls=":")
    ax.set(xscale="log", yscale="log", xlim=(0.1, 100), ylim=(1e-5, 1.2), xlabel="length $x$ (km)", title=title)
    ax.grid(color="#eeeeee", lw=0.5)
    if legend:
        ax.set_ylabel("$P(X>x)$")
        ax.legend(frameon=False, loc="lower left", fontsize=7.5, handlelength=1.2)


def forest(ax, result, values, colors, offsets, xlabel, title, labels=True):
    """Per class (rows), the quantities `values` (functions of a class's tail) with their bands."""
    for i, c in enumerate(CLASS_KEYS):
        t = result["lengths"][c]["tail"]
        for f, color, off in zip(values, colors, offsets):
            if t:
                v = f(t)
                ax.errorbar(v["value"], i + off, xerr=[[v["value"] - v["band"][0]], [v["band"][1] - v["value"]]], fmt="o", ms=4.5,
                            color=color, elinewidth=1.6, capsize=3)
    if title == "G − A":
        ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(CLASS_KEYS)), CLASS_KEYS if labels else [], fontsize=8)
    ax.set(ylim=(len(CLASS_KEYS) - 0.4, -0.6), xlabel=xlabel, title=title)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))


def draw(result):
    """The figure of the story for the glides: survival of S, A, G on every flight and in the high mountains, mu per class, mu_G - mu_A."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.3, 1]})
    survival_panel(axes[0], result["lengths"]["all"]["tail"], "every flight", legend=True)
    survival_panel(axes[1], result["lengths"]["high mountains"]["tail"], "high mountains")
    axes[1].set_yticklabels([])
    forest(axes[2], result, [lambda t, s=s: t["mu"][s] for s in "SAG"], [COLORS[p] for p in POPS], (-0.26, 0, 0.26),
           r"$\mu$ (Hill exponent)", "exponents")
    forest(axes[3], result, [lambda t: t["gaps"]["G - A"]], [raw.GAP_COLORS["G - A"]], (0,), r"$\mu_G-\mu_A$", "G − A", labels=False)
    save(fig, longlags.RESULTS / f"{NAME}.pdf")


if __name__ == "__main__":
    if "--draw" in sys.argv:
        draw(json.loads((longlags.RESULTS / f"{NAME}.json").read_text()))
        sys.exit()
    p, code, counts, classes = phase_gaps.context()
    runs = phase_gaps.assign(phase_terms.table(phase_terms.SET, phase_terms.kept_flights()), p, classes)
    result = {"labels": phase_terms.SET, "rule": phase_terms.RULE, "draws": DRAWS, "unit": "km", "classes": CLASS_KEYS,
              "lengths": statistics(runs, code, counts)}
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    (longlags.RESULTS / f"{NAME}.json").write_text(json.dumps(result, indent=1))
    table(result)
    draw(result)
