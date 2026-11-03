"""How open the declared open flights are, and on which flights the long lags rest.

The opening of a flight is its takeoff to landing distance over the path flown (openflights.opening). For
the declared open paragliders (the flights of tamsd.py run with --declared-open) it is given on every flight,
and on the flights of each dataset of longlags.py: those with a piece of S, G or A of at least
T / SUPPORT s for the longest lag T of each panel long enough for the longest lag of each panel. The share of the
flights kept by a threshold on the opening is given for takeoffs below and above MOUNTAIN_M m, a proxy of
lowland and mountain flying. The labels of the FFVL catalogue are counted with their class.

    uv run python opening.py --declared-open [--run NAME]    # after longlags.py --declared-open; writes out/<run>/open_declared/opening.*
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import closure
import longlags
from openflights import DECLARED_OPEN, MIN_OPENING
from tamsd import CACHE, FIXES, OUT

REJECTED = 0.5  # a stricter threshold on the opening, shown against MIN_OPENING
MOUNTAIN_M = 1000  # m: takeoff altitude above which a flight counts as a mountain flight
LABEL = {"solo": "S", "group": "G", "group_solo": "A"}


def labels(f):
    """Flights of each label of the catalogue, with its class."""
    kind = {**{t: "open" for t in DECLARED_OPEN}, **{t: "closed" for t in closure.CLOSED}}
    count = f["flight_type"].value_counts()
    return [{"label": t, "class": kind.get(t, "neither"), "flights": int(n)} for t, n in count.items()]


def described(ratio):
    """Flights, quartiles of the opening and the shares above MIN_OPENING and REJECTED."""
    return {"flights": int(len(ratio)), "above the threshold": int((ratio > MIN_OPENING).sum()), "quartiles": [round(float(q), 2) for q in ratio.quantile([0.25, 0.5, 0.75])],
            "above the threshold (%)": round(100 * float((ratio > MIN_OPENING).mean()), 1),
            "above the stricter threshold (%)": round(100 * float((ratio > REJECTED).mean()), 1)}


if __name__ == "__main__":
    every = closure.flight_table()
    open_ = every[every["declared"] == "open"].set_index("flight")
    meta = pd.read_parquet(FIXES["para"].parent / "flights_meta.parquet", columns=["flight_id", "alt0"]).astype({"flight_id": str})
    open_["alt0"] = open_["flight_id"].astype(str).map(meta.set_index("flight_id")["alt0"])
    p = dict(np.load(CACHE / "long_pieces.npz"))
    masks = longlags.populations(p)
    panels = json.loads((OUT / "longlags.json").read_text())["panels"]
    shortest = {f"lags up to {v['tau_max']} s": v["tau_max"] for v in panels.values()}
    sets = {"every declared open flight": {"all": open_.index.to_numpy()}}
    for name, tau in shortest.items():
        by_pop = {LABEL[k]: np.unique(p["flight"][longlags.long_enough(p, masks[k], tau)]) for k in LABEL}
        sets[name] = {"all": np.unique(np.concatenate(list(by_pop.values()))), **by_pop}
    result = {"min_opening": MIN_OPENING, "stricter": REJECTED, "mountain_m": MOUNTAIN_M, "labels": labels(every),
              "datasets": {name: {k: described(open_["ratio"].reindex(v).dropna()) for k, v in s.items()} for name, s in sets.items()},
              "by takeoff": {f"below {MOUNTAIN_M} m": described(open_.loc[open_["alt0"] < MOUNTAIN_M, "ratio"]),
                             f"{MOUNTAIN_M} m or more": described(open_.loc[open_["alt0"] >= MOUNTAIN_M, "ratio"])}}
    (OUT / "opening.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: {n: (v["all"]["flights"], v["all"]["quartiles"]) for n, v in result[k].items()} if k == "datasets" else result[k]
                      for k in ("datasets", "by takeoff")}, indent=1))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
    bins = np.linspace(0, 1, 41)
    for (name, s), color in zip(sets.items(), ("#555555", "#eb6834", "#8c2f0f")):
        r = open_["ratio"].reindex(s["all"]).dropna()
        left.hist(r.clip(0, 1), bins=bins, density=True, histtype="step", lw=1.2, color=color,
                  label=f"{name} ({len(r):,}), median {r.median():.2f}")
    for (name, keep), color in zip(((f"takeoff below {MOUNTAIN_M} m", open_["alt0"] < MOUNTAIN_M),
                                    (f"takeoff at {MOUNTAIN_M} m or more", open_["alt0"] >= MOUNTAIN_M)), ("#1baf7a", "#555555")):
        r = open_.loc[keep, "ratio"]
        right.hist(r.clip(0, 1), bins=bins, density=True, histtype="step", lw=1.2, color=color,
                   label=f"{name} ({len(r):,}), median {r.median():.2f}")
    for ax in (left, right):
        ax.axvline(MIN_OPENING, color="#1a7f5a", lw=1.1)
        ax.set(xlabel="opening: takeoff to landing / path flown", ylabel="density")
        ax.legend(frameon=False, fontsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(OUT / "opening.pdf", bbox_inches="tight")
    fig.savefig(OUT / "opening.png", dpi=200, bbox_inches="tight")
