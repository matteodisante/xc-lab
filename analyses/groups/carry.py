"""Carry-over of the group on the solo time of group flights: does A - S fall as the distance from the group grows?

The windows of prepost.py (WINDOW_S s, cached pieces) in the solo time of the flights with group time (ctrlA1, ctrlA2,
population A) and in solo flights (ctrlS1, ctrlS2, population S), alike in length and in how they were drawn. A window of A
is binned by its distance d from the nearest stretch of group time of the same flight, the gap in time between the two,
with the edges BINS_MIN (minutes). In each ground class and in "all" the exponent of a window set is the one fitted over
the lags LONG_FIT of its ensemble MSD, one weight per window, with the draws of twoway.py shared by every set, so that
A(d) - S has the band of its own draws. If the group carries over, A(d) - S falls toward 0 as d grows.

    uv run python carry.py --declared-open --dt5 [--window SECONDS]    # after prepost.py --ground (its cached pieces);
        # writes carry.json and carry.pdf into longlags.RESULTS (open_declared_dt5)
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import decompose
import longlags
import prepost
import tamsd
import twoway
from blocks import with_takeoffs
from contrasts import CLASSES
from tamsd import LABELS, N_BOOT, SEED, read_flights, save

BINS_MIN = (0, 20, 40, 80, np.inf)  # min: edges of the bins of the distance from the group time
POPULATIONS = {"A": ("ctrlA1", "ctrlA2"), "S": ("ctrlS1", "ctrlS2")}  # windows of each population
BIN_COLOR, S_COLOR = "#1baf7a", "#2a78d6"


def label(lo, hi):
    """The text of a bin of BINS_MIN, in minutes."""
    return f"{lo:g}--{hi:g}~min" if np.isfinite(hi) else f"$\\ge{lo:g}$~min"


def distance(flight, a, b, spans):
    """Minutes from each window [a, b) (flight clock) of `flight` to the nearest group stretch of the same flight."""
    group = spans[spans["population"] == "group"][["flight", "a", "b"]].rename(columns={"a": "ga", "b": "gb"})
    w = pd.DataFrame({"i": np.arange(len(a)), "flight": flight, "a": a, "b": b}).merge(group, on="flight")
    w["d"] = np.maximum(0, np.maximum(w["ga"] - w["b"], w["a"] - w["gb"]))
    return w.groupby("i")["d"].min().reindex(np.arange(len(a))).to_numpy() / 60


def exponent(units, counts, mask, lags):
    """The exponent over the lags LONG_FIT of the windows of `mask`: the point estimate, then one value per draw
    (NaN where a draw keeps none of them), and the windows and flights."""
    mean, draws = tamsd.ensemble(units.values[mask], units.block[mask], None, counts)
    msd = np.vstack([mean, draws])
    ok = np.isfinite(msd).all(axis=1)
    rows = np.full(len(msd), np.nan)
    rows[ok] = prepost.long_fit(msd[ok], lags)
    return rows, {"windows": int(mask.sum()), "flights": int(np.unique(units.flight[mask]).size)}


def run(flights, membership, summary, width):
    flights, spans, wins, p, lags = prepost.window_pieces(flights, membership, summary, width)
    block, of_pilot, of_block = twoway.cells(flights, summary)
    counts = twoway.counts(of_pilot, of_block, np.random.default_rng(SEED))
    units = decompose.msd_units(p, "episode", decompose.flight_table(flights, summary)[0], block)
    where = prepost.ground_class(units.flight)
    dist = distance(*prepost.unit_windows(p, units, wins), spans)
    a, s = (np.isin(units.population, POPULATIONS[k]) for k in ("A", "S"))
    edges = list(zip(BINS_MIN[:-1], BINS_MIN[1:]))
    result = {"window_s": width, "lags": lags.tolist(), "long_fit": prepost.LONG_FIT, "n_boot": N_BOOT,
              "bins_min": [[lo, hi if np.isfinite(hi) else None] for lo, hi in edges],
              "labels": [label(lo, hi) for lo, hi in edges],
              "plain": [f"{lo:g}-{hi:g} min" if np.isfinite(hi) else f">={lo:g} min" for lo, hi in edges], "classes": {}}
    for k in (*CLASSES, "all"):
        here = np.ones(len(where), bool) if k == "all" else where == k
        s_rows, s_n = exponent(units, counts, here & s, lags)
        out = {"S": {**s_n, **twoway.quote(s_rows)}, "bins": []}
        for name, m in (*(((label(lo, hi), (dist >= lo) & (dist < hi)) for lo, hi in edges)), ("any", np.ones(len(dist), bool))):
            rows, n = exponent(units, counts, here & a & m, lags)
            entry = {"label": name, **n, "alpha": twoway.quote(rows), "diff": twoway.quote(rows - s_rows)}
            if name == "any":
                out["A any"] = entry
            else:
                out["bins"].append(entry)
        result["classes"][k] = out
        print(k, "S", s_n["windows"], [(b["label"], b["windows"], round(b["diff"]["value"], 4)) for b in out["bins"]], flush=True)
    return result


def figure(result, path):
    """One column per class: the exponent of A(d) per bin with that of S as a band (top), A(d) - S (bottom)."""
    keys = [*CLASSES, "all"]
    fig, axes = plt.subplots(2, len(keys), figsize=(12, 5), sharey="row", constrained_layout=True)
    n = len(result["labels"])
    for j, k in enumerate(keys):
        r = result["classes"][k]
        top, bottom = axes[:, j]
        top.axhspan(*r["S"]["band"], color=S_COLOR, alpha=0.2, lw=0)
        top.axhline(r["S"]["value"], color=S_COLOR, lw=1, label="S")
        for ax, key in ((top, "alpha"), (bottom, "diff")):
            for i, b in enumerate(r["bins"]):
                v = b[key]
                ax.errorbar(i, v["value"], [[v["value"] - v["band"][0]], [v["band"][1] - v["value"]]], fmt="o", ms=4,
                            capsize=2, color=BIN_COLOR)
            ax.set(xticks=range(n), xlim=(-0.5, n - 0.5))
            ax.grid(color="#eeeeee", lw=0.5)
        bottom.axhline(0, color="k", lw=0.6)
        top.set_title(f"{k}, S {r['S']['windows']:,} windows".replace(",", " "), fontsize=8)
        bottom.set_xticklabels([f"{text}\n{b['windows']:,}".replace(",", " ") for text, b in zip(result["plain"], r["bins"])], fontsize=6)
        top.set_xticklabels([])
    axes[0, 0].set_ylabel(r"$\alpha$ of the window")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].set_ylabel(r"$\alpha_{A(d)}-\alpha_S$")
    lo, hi = result["long_fit"]
    fig.suptitle(f"Solo time of group flights by distance d from the group time, windows of {result['window_s']} s, "
                 f"lags {lo} to {hi} s", fontsize=9)
    save(fig, path)


if __name__ == "__main__":
    flights = with_takeoffs(read_flights())
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    width = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else prepost.WINDOW_S
    tamsd.folders()
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    result = run(flights, membership, summary, width)
    (longlags.RESULTS / "carry.json").write_text(json.dumps(result, indent=1))
    figure(result, longlags.RESULTS / "carry.pdf")
