"""Is the path of group time straighter, or the same path flown more regularly? alpha* along the route.

The pieces of S, A, G of route_arc.py, measured in arc length: the route (each climb of our refit replaced by its chord)
and, as the check of the method, the path with the circles kept. Each kind keeps the pieces of at least
tamsd.TAU_MAX / SUPPORT steps of 10 (a lag of LONG_LAGS is lag / 10 steps of route_arc.DS_M), then
cohorts.chained gives MSD* and alpha* per population and stratum (the four ground classes and "all") with the draws
of twoway.counts shared with the chains in time (contrasts.py), so every gap, and every difference between a gap on
the route and the same gap in time, is a difference of rows with the band of its own draws.

Arc-length lags are mapped to time by the median route speed of S: t = s / v (approximate: the speed varies from
piece to piece). The gaps in time are those of contrasts.py (checked against contrasts_ground.json). The check:
alpha* of the path (circles kept) against alpha* in time at the same lag of LONG_LAGS, over the lags that are
selection-free (cohorts.solid).

    uv run python route_alpha.py --declared-open --dt5    # after route_arc.py; writes route_alpha.json,
        # route_gaps.pdf and route_check.pdf into longlags.RESULTS
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cohorts
import contrasts
import longlags
import raw
import route_arc
import tamsd
import twoway
from tamsd import COLORS, LABELS, OUT, SEED, read_flights, save

STRATA = (*contrasts.CLASSES, "all")
GAPS = ("G - A", "A - S", "G - S")
DRAWN = ("G - A", "A - S")  # the gaps of route_gaps.pdf; G - S is their sum
SCALES = {"ThreeTwenty": 320, "Thousand": 1000}  # s: the time scales at which the gaps are quoted
MIN_STEPS = tamsd.TAU_MAX / tamsd.SUPPORT / 10  # steps of a piece that enters: as the 1250 s of the time pieces


def kind_pieces(p, d, kind):
    """The pieces of POPS in the form cohorts.chained reads: duration = steps of the grid x 10, so that the lag and
    the support rule are those of the time pieces."""
    rows = d["row"]
    q = {"flight": p["flight"][rows], "population": p["population"][rows], "duration": d[f"steps_{kind}"] * 10.0,
         "sums": d[f"sums_{kind}"], "origins": d[f"origins_{kind}"], "time": p["duration"][rows],
         "length": d[f"length_{kind}"]}
    keep = q["duration"] >= MIN_STEPS * 10
    return {k: v[keep] for k, v in q.items()}


def chains(q, where, code, counts, rng):
    """{stratum: {population: cohorts.chained}} of the pieces `q`."""
    out = {}
    for s in STRATA:
        m = np.ones(len(where), bool) if s == "all" else where == s
        out[s] = {pop: cohorts.chained(q, (q["population"] == pop) & m, code, counts, rng) for pop in raw.POPS}
    return out


def gap(ch, name):
    """The lags and the rows of a gap of raw.GAPS from the chains of a stratum."""
    a, b = raw.GAPS[name]
    n = min(len(ch[a]["lags"]), len(ch[b]["lags"]))
    return np.asarray(ch[a]["lags"][:n]), ch[a]["rows"][:, :n] - ch[b]["rows"][:, :n]


def at(lags, rows, x):
    """The rows of a curve on `lags` linearly interpolated in log lag at `x` (inside the range of lags)."""
    lx = np.log(lags)
    i = np.clip(np.searchsorted(lx, np.log(x)) - 1, 0, len(lx) - 2)
    w = (np.log(x) - lx[i]) / (lx[i + 1] - lx[i])
    return rows[:, i] * (1 - w) + rows[:, i + 1] * w


def curve(lags, rows, x):
    """The curve at the lags x of the plot: lags (those of the chain), solid (selection-free), x, value, band."""
    return {"lags": lags.tolist(), "x": np.asarray(x).tolist(), "solid": cohorts.solid(lags).tolist(), **twoway.quote(rows)}


def summarise(route, time, factor):
    """Per stratum: alpha* and the gaps on the route, the gaps in time on the same axis (x = time-equivalent lag, s)
    and their difference time - route where the time curve has the lag; the gaps at SCALES."""
    out = {}
    for s in STRATA:
        r = {"flights": {k: c["flights"][0] for k, c in route[s].items()}, "pieces": {k: c["pieces"][0] for k, c in route[s].items()},
             "alpha": {pop: curve(np.asarray(c["lags"]), c["rows"], np.asarray(c["lags"]) * factor) for pop, c in route[s].items()},
             "gaps": {}, "time": {}, "minus": {}, "at": {}}
        for name in GAPS:
            lags, rows = gap(route[s], name)
            x = lags * factor
            tl, tr = gap(time[s], name)
            inside = (x >= tl[0]) & (x <= tl[-1])
            r["gaps"][name] = curve(lags, rows, x)
            r["time"][name] = curve(tl, tr, tl)
            r["minus"][name] = {"lags": lags[inside].tolist(), "x": x[inside].tolist(), "solid": cohorts.solid(lags[inside]).tolist(),
                                **twoway.quote(at(tl, tr, x[inside]) - rows[:, inside])}
            for tag, tau in SCALES.items():
                k = int(np.argmin(np.where(inside, np.abs(np.log(x / tau)), np.inf)))
                i, g, m = int(inside[:k].sum()), r["gaps"][name], r["minus"][name]
                r["at"].setdefault(tag, {"lag": int(lags[k]), "time_lag": float(x[k])})[name] = {
                    "route": {"value": g["value"][k], "band": [b[k] for b in g["band"]]},
                    "minus": {"value": m["value"][i], "band": [b[i] for b in m["band"]]}}
        out[s] = r
    return out


def check(arc, time):
    """alpha* of the path (circles kept) minus alpha* in time, per population, in the stratum "all", at the lags the
    chains share; the largest absolute difference over the selection-free lags."""
    out = {}
    for pop in raw.POPS:
        a, t = arc["all"][pop], time["all"][pop]
        n = min(len(a["lags"]), len(t["lags"]))
        lags = np.asarray(a["lags"][:n])
        out[pop] = {"lags": lags.tolist(), "solid": cohorts.solid(lags).tolist(), "arc": twoway.quote(a["rows"][:, :n]),
                    "time": twoway.quote(t["rows"][:, :n]), "minus": twoway.quote(a["rows"][:, :n] - t["rows"][:, :n])}
    out["max_abs"] = max(abs(np.array(v["minus"]["value"])[v["solid"]]).max() for v in out.values())
    return out


# ---- the figures ----

def gaps_figure(r, path):
    """One panel per stratum: G - A and A - S on the route (band) and in time (dotted), against the time lag; the
    route lags are mapped by the median route speed of S."""
    fig, axes = plt.subplots(1, len(STRATA), figsize=(16, 3.8), sharey=True)
    for ax, s in zip(axes, STRATA):
        for name in DRAWN:
            c, t, color = r[s]["gaps"][name], r[s]["time"][name], raw.GAP_COLORS[name]
            x, keep = np.array(c["x"]), np.array(c["solid"])
            ax.fill_between(x, *c["band"], color=color, alpha=0.15, lw=0)
            ax.plot(x[keep], np.array(c["value"])[keep], color=color, lw=1.5, label=contrasts.title(name) + ", route")
            tx, tk = np.array(t["x"]), np.array(t["solid"])
            ax.plot(tx[tk], np.array(t["value"])[tk], color=color, lw=1.2, ls=":", label=contrasts.title(name) + ", time")
        contrasts.finish(ax, s)
    gs = [{"lags": r[s][k][n]["lags"], "value": r[s][k][n]["value"], "band": r[s][k][n]["band"]} for s in STRATA for k in ("gaps", "time") for n in DRAWN]
    axes[0].set(ylabel=r"$\Delta\alpha$", ylim=contrasts.ylim(gs))
    axes[0].legend(frameon=False, fontsize=6.5)
    save(fig, path)


def check_figure(c, path):
    """Left: alpha* of S, A, G of the path with circles kept (band) and in time (dotted); right: their difference."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    for pop, short in raw.POPS.items():
        v, lags, keep = c[pop], np.array(c[pop]["lags"]), np.array(c[pop]["solid"])
        for ax, g, ls in ((left, v["arc"], "-"), (right, v["minus"], "-")):
            ax.fill_between(lags, *g["band"], color=COLORS[pop], alpha=0.15, lw=0)
            ax.plot(lags[keep], np.array(g["value"])[keep], color=COLORS[pop], lw=1.5, ls=ls, label=short if ax is left else None)
        left.plot(lags[keep], np.array(v["time"]["value"])[keep], color=COLORS[pop], lw=1.0, ls=":")
    left.plot([], [], color="k", lw=1.0, ls=":", label=r"$\alpha^*$ in time")
    left.set(ylabel=r"$\alpha^*$ of the path (circles kept)", ylim=cohorts.YLIM)
    right.axhline(0, color="k", lw=0.6)
    right.set(ylabel=r"$\alpha^*_\mathrm{arc}-\alpha^*_\mathrm{time}$")
    for ax in (left, right):
        ax.set(xscale="log", xlabel=r"lag (s; arc length / $\Delta s\times10$ s)")
        ax.grid(color="#eeeeee", lw=0.5)
    left.legend(frameon=False, fontsize=7)
    save(fig, path)


def context():
    flights = raw.with_takeoffs(read_flights())
    code, pilot, block = twoway.cells(flights, pd.read_parquet(LABELS / "flight_summary.parquet"))
    counts = twoway.counts(pilot, block, np.random.default_rng(SEED))  # the draws of contrasts.context
    return code, counts


def compare_saved(time):
    """The chains in time against contrasts_ground.json: the same alpha* to rounding."""
    saved = json.loads((longlags.RESULTS / "contrasts_ground.json").read_text())
    return max(abs(np.array(saved[s]["alpha"][pop]["value"]) - time[s][pop]["rows"][0]).max() for s in STRATA for pop in raw.POPS)


if __name__ == "__main__":
    code, counts = context()
    rng = np.random.default_rng(SEED)
    p, d = longlags.load_pieces(), route_arc.load()
    assert len(d["row"]) == np.isin(p["population"], route_arc.POPS).sum(), "route_arc.py has not finished"
    classes = pd.read_parquet(OUT / "ground.parquet").set_index("flight")["class"]
    q = {k: kind_pieces(p, d, k) for k in route_arc.KINDS}
    time = chains(p, classes.reindex(p["flight"]).to_numpy(), code, counts, rng)
    print("time chains, largest difference from contrasts_ground.json:", compare_saved(time), flush=True)
    ch = {k: chains(q[k], classes.reindex(q[k]["flight"]).to_numpy(), code, counts, rng) for k in route_arc.KINDS}
    s = q["route"]["population"] == "solo"
    speed = {"route": float(np.median((q["route"]["length"] / q["route"]["time"])[s])),
             "path": float(np.median((q["arc"]["length"] / q["arc"]["time"])[q["arc"]["population"] == "solo"]))}
    factor = route_arc.DS_M / 10 / speed["route"]  # s of time per lag of LONG_LAGS (a lag is lag / 10 steps of DS_M)
    result = {"ds_m": route_arc.DS_M, "speed_route_S": speed["route"], "speed_path_S": speed["path"], "seconds_per_lag": factor,
              "min_steps": MIN_STEPS, "pieces": {k: len(v["flight"]) for k, v in q.items()}, "of": int(len(d["row"])),
              "scales": SCALES, "route": summarise(ch["route"], time, factor), "check": check(ch["arc"], time),
              "arc": {st: {pop: {"lags": c["lags"], "pieces": c["pieces"], **twoway.quote(c["rows"])} for pop, c in ch["arc"][st].items()} for st in STRATA}}
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    (longlags.RESULTS / "route_alpha.json").write_text(json.dumps(result))
    gaps_figure(result["route"], longlags.RESULTS / "route_gaps.pdf")
    check_figure(result["check"], longlags.RESULTS / "route_check.pdf")
    print("DS", route_arc.DS_M, "m; route speed of S", round(speed["route"], 2), "m/s; check max", round(result["check"]["max_abs"], 4))
