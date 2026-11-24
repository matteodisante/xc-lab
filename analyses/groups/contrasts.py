"""S, A and G under a stratum: the chained alpha*, the gaps of raw.GAPS, their normalised form and the figures of the
story (Ch. 2-5).

One engine for every design. In a stratum (a mask over the pieces) cohorts.chained gives alpha* of each population
with the cells and the draws of twoway.counts shared by all of them, so a gap is a difference of rows and has the
band of its own draws. The normalised gap (a - b) / (1 - b) is the share of the distance of b from ballistic
(alpha* = 1) that a closes, computed draw by draw. A gap is drawn on the lags shared by its two populations, solid
up to cohorts.SELECTION_FREE, dashed beyond. Per stratum the JSON holds, at the first and at the last lag, the
flights and pieces of each population, alpha*, "gaps" and "relative gaps" (with "solid": the lags drawn solid). G - S = (G - A) + (A - S) exactly.

Designs (one contrasts_<design>.json each):
- all: no stratum;
- ground: each ground class of ground.py, and "all";
- ground_opening: each class at each opening threshold of THRESHOLDS (keys "<class>|none", "<class>|0.1", ...);
- pilot: per class and "all", S' = the solo flights of the pilots who also have a flight with group time, and the
  gaps A - S' and G - S' next to A - S and G - S; "S' share" is the share of the flights of S that S' holds.

With --perflight (designs all and ground only) the chains give one weight per flight instead of one per piece
(cohorts.chained, per_flight): contrasts_<design>_perflight.json holds the gaps per flight ("gaps"), per piece
("gaps per piece") and their difference ("gaps minus per piece", on the draws the two share), and weights.pdf draws them.

    uv run python contrasts.py --declared-open --dt5 [design ...]    # after longlags.py and ground.py; writes
        # contrasts_<design>.json and the figures into longlags.RESULTS (open_declared_dt5), all designs by default
    uv run python contrasts.py --declared-open --dt5 --perflight [all] [ground]    # one weight per flight
"""

import hashlib
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import closure
import cohorts
import ground
import isolated
import longlags
import raw
import twoway
from tamsd import COLORS, LABELS, OUT, SEED, read_flights, save

CLASSES = ("plains", "hills", "low mountains", "high mountains")
CLASS_COLORS = dict(zip(CLASSES, ("#8fcf6b", "#3f9a4a", "#a8793a", "#5a3a1c")))  # light green to dark brown
ALL_COLOR = "#888888"
THRESHOLDS = (None, 0.1, 0.5)  # openings: None, no filter
THRESHOLD_COLORS = ("#555555", "#1f77b4", "#d95f02")
ALPHA_YLIM = (0.75, 1.00)  # alpha axis of ground_alpha.pdf: the mountain curves go below cohorts.YLIM
YLIM_PAD = 0.01  # a gap axis widens to the solid values rounded out by this much
REL_YLIM = (-0.15, 0.45)  # axis of the normalised gaps: the solid values of the classes, rounded out
SELF = "solo'"  # S': the solo flights of the pilots who also have group time
PILOT_GAPS = {**raw.GAPS, "A - S'": ("group_solo", SELF), "G - S'": ("group", SELF)}
PILOT_COLORS = {"A - S": raw.GAP_COLORS["A - S"], "A - S'": "#0b5a3e", "G - S'": COLORS["group"]}

_CHAINS = {}  # the chains are deterministic given the draws: a population in a stratum is chained once


def chain(p, mask, code, counts, rng, per_flight=False):
    key = hashlib.md5(mask.tobytes()).digest(), per_flight
    if key not in _CHAINS:
        _CHAINS[key] = cohorts.chained(p, mask, code, counts, rng, per_flight)
    return _CHAINS[key]


def stratum(ctx, mask, extra=None, gaps=raw.GAPS):
    """alpha* of S, A, G (and of the populations `extra`: name -> mask) on the pieces of `mask`, and the gaps of
    `gaps` (name -> populations), plain and normalised, each on the lags of its two populations."""
    p, masks = ctx["p"], ctx["masks"]
    pops = {pop: masks[pop] & mask for pop in raw.POPS} | {k: m & mask for k, m in (extra or {}).items()}
    per_flight = ctx.get("per_flight", False)
    ch = {k: chain(p, m, ctx["code"], ctx["counts"], ctx["rng"], per_flight) for k, m in pops.items()}
    out = {"flights": {k: c["flights"][0] for k, c in ch.items()}, "pieces": {k: c["pieces"][0] for k, c in ch.items()},
           "flights_last": {k: c["flights"][-1] for k, c in ch.items()}, "pieces_last": {k: c["pieces"][-1] for k, c in ch.items()},
           "alpha": {k: {"lags": c["lags"].tolist() if hasattr(c["lags"], "tolist") else c["lags"], "pieces": c["pieces"],
                         **twoway.quote(c["rows"])} for k, c in ch.items()},
           "gaps": {}, "relative gaps": {}}
    for name, (a, b) in gaps.items():
        n = min(len(ch[a]["lags"]), len(ch[b]["lags"]))
        lags = list(map(int, ch[a]["lags"][:n]))
        x, y = ch[a]["rows"][:, :n], ch[b]["rows"][:, :n]
        solid = cohorts.solid(lags).tolist()
        out["gaps"][name] = {"lags": lags, "solid": solid, **twoway.quote(x - y)}
        out["relative gaps"][name] = {"lags": lags, "solid": solid, **twoway.quote((x - y) / (1 - y))}
        if per_flight:  # the same gap with one weight per piece, and the change, on the draws they share
            piece = {k: chain(p, m, ctx["code"], ctx["counts"], ctx["rng"]) for k, m in pops.items()}
            u, v = piece[a]["rows"][:, :n], piece[b]["rows"][:, :n]
            out.setdefault("gaps per piece", {})[name] = {"lags": lags, "solid": solid, **twoway.quote(u - v)}
            out.setdefault("gaps minus per piece", {})[name] = {"lags": lags, "solid": solid, **twoway.quote((x - y) - (u - v))}
    out["last_lag"] = {name: g["lags"][-1] for name, g in out["gaps"].items()}
    return out


# ---- the designs ----

def design_all(ctx):
    return {"all": stratum(ctx, np.ones(len(ctx["where"]), bool))}


def design_ground(ctx):
    out = {c: stratum(ctx, ctx["where"] == c) for c in CLASSES} | design_all(ctx)
    t = pd.read_parquet(OUT / "ground.parquet").set_index("flight").reindex(np.unique(ctx["p"]["flight"]))
    known = t[(t["class"] != ground.terrain.UNKNOWN) & (t["class_all"] != ground.terrain.UNKNOWN)]
    out["share"] = {"flights": len(t), "unknown": float((t["class"] == ground.terrain.UNKNOWN).mean()),
                    "changed": float((known["class"] != known["class_all"]).mean())}
    out["constants"] = {"trim": ground.TRIM, "edges_m": ground.terrain.EDGES_M, "cell_m": ground.terrain.CELL_M,
                        "min_on_dem": ground.terrain.MIN_ON_DEM}
    return out


def threshold(x):
    return "none" if x is None else f"{x:g}"


def design_ground_opening(ctx):
    above = {x: np.ones(len(ctx["where"]), bool) if x is None else ctx["opening"] > x for x in THRESHOLDS}
    out = {f"{c}|{threshold(x)}": stratum(ctx, (ctx["where"] == c) & above[x]) for c in CLASSES for x in THRESHOLDS}
    out["classes"] = classes_summary(ctx, above[THRESHOLDS[-1]])
    return out


def classes_summary(ctx, strict):
    """Per class: the median opening of the flights of S, A, G, and the flights (with a piece of S, A or G) of the
    class among all and among those on `strict` pieces; "known": the same for every class together."""
    p, masks = ctx["p"], ctx["masks"]
    ratio = pd.Series(ctx["opening"], index=p["flight"]).groupby(level=0).first()  # one opening per flight

    def flights(mask):
        return np.unique(p["flight"][mask & np.isin(p["population"], list(raw.POPS))])

    out = {c: {"median": {pop: float(ratio.reindex(np.unique(p["flight"][masks[pop] & (ctx["where"] == c)])).median())
                          for pop in raw.POPS},
               "flights": len(flights(ctx["where"] == c)), "flights_strict": len(flights((ctx["where"] == c) & strict))}
           for c in CLASSES}
    out["known"] = {"flights": sum(v["flights"] for v in out.values()), "flights_strict": sum(v["flights_strict"] for v in out.values())}
    return out


def design_pilot(ctx):
    """S' needs the pilot of each flight (isolated.codes) and the flights with group time."""
    flights, summary = ctx["flights"], ctx["summary"]
    pilot = pd.Series(isolated.codes(flights, summary, flights["flight"].to_numpy())[0], index=flights["flight"])
    grouped = summary.set_index("flight")["n_entries"].reindex(flights["flight"]).fillna(0).to_numpy() > 0
    grouped &= ctx["kept"]
    mine = np.isin(pilot.reindex(ctx["p"]["flight"]).to_numpy(), np.unique(pilot.to_numpy()[grouped]))
    extra = {SELF: ctx["masks"]["solo"] & mine}
    strata = {**{c: ctx["where"] == c for c in CLASSES}, "all": np.ones(len(mine), bool)}
    out = {k: stratum(ctx, m, extra, PILOT_GAPS) for k, m in strata.items()}
    for s in out.values():
        s["S' share"] = s["flights"][SELF] / s["flights"]["solo"]
    out["group pilots"] = int(len(np.unique(pilot.to_numpy()[grouped])))
    return out


PERFLIGHT_DESIGNS = ("all", "ground")  # the designs that --perflight runs
DESIGNS = {"all": design_all, "ground": design_ground, "ground_opening": design_ground_opening, "pilot": design_pilot}


# ---- the figures ----

def curve(ax, g, color, label=None, ls="-", alpha=1.0, band=True, lw=1.5):
    """The curve `g` (lags, value, band): its band, then solid up to cohorts.SELECTION_FREE and dashed beyond."""
    lags, v, b = np.array(g["lags"]), np.array(g["value"]), np.array(g["band"])
    if band:
        ax.fill_between(lags, b[0], b[1], color=color, alpha=0.15 * alpha, lw=0)
    keep = cohorts.solid(lags)
    tail = np.arange(len(keep)) >= keep.sum() - 1  # the solid part is a prefix: the window grows with the lag
    ax.plot(lags[keep], v[keep], color=color, lw=lw, ls=ls, alpha=alpha, label=label)
    ax.plot(lags[tail], v[tail], color=color, lw=lw, ls="--", alpha=alpha)


def solid_values(gs):
    """The values and band edges of the curves `gs` where they are solid."""
    return np.concatenate([np.array(x)[cohorts.solid(g["lags"])] for g in gs for x in (g["value"], *g["band"])])


def ylim(gs, base=cohorts.GAP_YLIM):
    """`base` widened to hold the values of `gs` where solid, rounded out by YLIM_PAD."""
    v = solid_values(gs)
    return (min(base[0], np.floor(v.min() / YLIM_PAD) * YLIM_PAD), max(base[1], np.ceil(v.max() / YLIM_PAD) * YLIM_PAD))


def title(name):
    return name.replace(" - ", " $-$ ")


def finish(ax, name=None):
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xscale="log", xlabel=r"$\tau$ (s)")
    if name:
        ax.set_title(name, fontsize=9)
    ax.grid(color="#eeeeee", lw=0.5)


def identity_figure(s, path):
    """G - S, G - A and A - S together on the pieces of the stratum: the first is the sum of the other two."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, g in s["gaps"].items():
        curve(ax, g, raw.GAP_COLORS[name], title(name))
    finish(ax)
    ax.set(ylabel=r"$\Delta\alpha^*(\tau)$", ylim=ylim(s["gaps"].values()))
    ax.legend(frameon=False, fontsize=7)
    save(fig, path)


def identity_ground_figure(r, path):
    """The same, one panel per ground class."""
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(14, 3.8), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for name, g in r[c]["gaps"].items():
            curve(ax, g, raw.GAP_COLORS[name], title(name))
        finish(ax, c)
    axes[0].set(ylabel=r"$\Delta\alpha^*(\tau)$", ylim=ylim([g for c in CLASSES for g in r[c]["gaps"].values()]))
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, path)


def ground_gap_figure(r, path, key="gaps", ylabel=r"$\Delta\alpha^*(\tau)$", lim=None):
    """One panel per gap, one curve per class and the whole cache in grey."""
    fig, axes = plt.subplots(1, len(raw.GAPS), figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, raw.GAPS):
        for k, color in (*CLASS_COLORS.items(), ("all", ALL_COLOR)):
            curve(ax, r[k][key][name], color, k)
        finish(ax, title(name))
    axes[0].set(ylabel=ylabel, ylim=lim or ylim([r[k][key][n] for k in CLASSES for n in raw.GAPS]))
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, path)


def ground_alpha_figure(r, path):
    """One panel per class: alpha* of S, A, G."""
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(14, 3.8), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for pop, short in raw.POPS.items():
            curve(ax, r[c]["alpha"][pop], COLORS[pop], short)
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=c)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\alpha^*(\tau)$", ylim=ALPHA_YLIM)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, path)


def opening_figure(r, key, path, label):
    """Rows: the gaps; columns: the classes; one curve per opening threshold. Each row shares its axis."""
    fig, axes = plt.subplots(len(raw.GAPS), len(CLASSES), figsize=(14, 9), sharex=True, sharey="row")
    for i, name in enumerate(raw.GAPS):
        gs = [r[f"{c}|{threshold(x)}"][key][name] for c in CLASSES for x in THRESHOLDS]
        for j, c in enumerate(CLASSES):
            for x, color in zip(THRESHOLDS, THRESHOLD_COLORS):
                curve(axes[i, j], r[f"{c}|{threshold(x)}"][key][name], color, "no filter" if x is None else f"$O>{x:g}$")
            finish(axes[i, j], c if i == 0 else None)
        axes[i, 0].set(ylabel=f"{title(name)}\n{label}", ylim=ylim(gs, (0, 0)))
    axes[0, 0].legend(frameon=False, fontsize=7)
    save(fig, path)


def pilot_figure(r, path):
    """One panel per stratum: A - S and A - S' (G - S' light dashed), with bands."""
    keys = ("all", *CLASSES)
    fig, axes = plt.subplots(1, len(keys), figsize=(16, 3.8), sharey=True)
    for ax, k in zip(axes, keys):
        for name, color in PILOT_COLORS.items():
            light = name == "G - S'"
            curve(ax, r[k]["gaps"][name], color, title(name), ls="--" if light else "-", alpha=0.6 if light else 1.0, band=not light)
        finish(ax, k)
    axes[0].set(ylabel=r"$\Delta\alpha^*(\tau)$", ylim=ylim([r[k]["gaps"][n] for k in keys for n in PILOT_COLORS]))
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, path)


def weights_figure(r, path):
    """Rows: G - S, G - A, A - S per piece (grey) and per flight (colour), then their differences (per flight minus per
    piece); columns: the classes and "all". Each row shares its axis."""
    keys = (*CLASSES, "all")
    fig, axes = plt.subplots(len(raw.GAPS) + 1, len(keys), figsize=(14, 11), sharex=True, sharey="row")
    for i, name in enumerate(raw.GAPS):
        gs = [r[k][key][name] for k in keys for key in ("gaps", "gaps per piece")]
        for j, k in enumerate(keys):
            curve(axes[i, j], r[k]["gaps per piece"][name], "#777777", "per piece")
            curve(axes[i, j], r[k]["gaps"][name], raw.GAP_COLORS[name], "per flight")
            finish(axes[i, j], k if i == 0 else None)
        axes[i, 0].set(ylabel=rf"{title(name)}: $\Delta\alpha^*(\tau)$", ylim=ylim(gs))
    for j, k in enumerate(keys):
        for name in raw.GAPS:
            curve(axes[-1, j], r[k]["gaps minus per piece"][name], raw.GAP_COLORS[name], title(name))
        finish(axes[-1, j])
    axes[-1, 0].set(ylabel=r"per flight $-$ per piece", ylim=ylim([r[k]["gaps minus per piece"][n] for k in keys for n in raw.GAPS], (0, 0)))
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[-1, 0].legend(frameon=False, fontsize=7)
    save(fig, path)


def figures(design, r, folder):
    if design == "all":
        identity_figure(r["all"], folder / "identity.pdf")
    elif design == "ground":
        identity_ground_figure(r, folder / "identity_ground.pdf")
        ground_gap_figure(r, folder / "ground_gaps.pdf")
        ground_gap_figure(r, folder / "ground_relgaps.pdf", "relative gaps", r"$(\alpha^*_a-\alpha^*_b)/(1-\alpha^*_b)$", REL_YLIM)
        ground_alpha_figure(r, folder / "ground_alpha.pdf")
    elif design == "ground_opening":
        opening_figure(r, "gaps", folder / "ground_opening.pdf", r"$\Delta\alpha^*(\tau)$")
        opening_figure(r, "relative gaps", folder / "ground_opening_rel.pdf", r"$(\alpha^*_a-\alpha^*_b)/(1-\alpha^*_b)$")
    elif design == "pilot":
        pilot_figure(r, folder / "pilot.pdf")


def context():
    """The pieces, the populations and the shared draws, as cohorts.py __main__."""
    flights = raw.with_takeoffs(read_flights())
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    code, pilot, block = twoway.cells(flights, summary)
    rng = np.random.default_rng(SEED)
    counts = twoway.counts(pilot, block, rng)
    p = longlags.load_pieces()
    classes = pd.read_parquet(OUT / "ground.parquet").set_index("flight")["class"]  # a class belongs to the flight
    opening = closure.flight_table().set_index("flight")["ratio"]
    kept = (flights["dt_native"] <= longlags.MAX_DT + 1e-6).to_numpy() if longlags.DT5 else np.ones(len(flights), bool)
    return {"p": p, "masks": longlags.populations(p), "code": code, "counts": counts, "rng": rng, "flights": flights,
            "summary": summary, "kept": kept, "where": classes.reindex(p["flight"]).to_numpy(),
            "opening": opening.reindex(p["flight"]).to_numpy()}


if __name__ == "__main__":
    perflight = "--perflight" in sys.argv
    wanted = [a for a in sys.argv[1:] if a in DESIGNS and (not perflight or a in PERFLIGHT_DESIGNS)] or \
        (list(PERFLIGHT_DESIGNS) if perflight else list(DESIGNS))
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    ctx = context() | {"per_flight": perflight}
    for design in wanted:
        result = DESIGNS[design](ctx)
        if perflight:
            (longlags.RESULTS / f"contrasts_{design}_perflight.json").write_text(json.dumps(result))
            if design == "ground":
                weights_figure(result, longlags.RESULTS / "weights.pdf")
        else:
            (longlags.RESULTS / f"contrasts_{design}.json").write_text(json.dumps(result))
            figures(design, result, longlags.RESULTS)
        print(design, "done", flush=True)
