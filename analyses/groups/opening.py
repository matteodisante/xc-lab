"""How open the declared open flights are, and on which flights the long lags rest.

The opening of a flight is its takeoff to landing distance over the path flown (openflights.opening). For
the declared open paragliders (the flights of tamsd.py run with --declared-open) it is given on every flight,
and on the flights of each dataset of longlags.py: those with a piece of S, G or A of at least
T / SUPPORT s for the longest lag T of each panel long enough for the longest lag of each panel. The share of the
flights kept by a threshold on the opening is given for takeoffs below and above MOUNTAIN_M m, a proxy of
lowland and mountain flying, for flights shorter and longer than DURATION_S, and for solo and group flights
(entered a group at least once; their group and solo time share the flight, hence its opening). The labels of the FFVL catalogue are counted with their class.

    uv run python opening.py --declared-open [--run NAME]    # after longlags.py --declared-open; writes out/<run>/open_declared/opening.*
    # with --dt5 only the flights up to longlags.MAX_DT s (the labels of the catalogue still count every flight),
    # written to longlags.RESULTS (open_declared_dt5), after longlags.py --dt5
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import closure
import longlags
from openflights import DECLARED_OPEN, MIN_OPENING
from tamsd import COLORS, FIXES

REJECTED = 0.5  # a stricter threshold on the opening, shown against MIN_OPENING
MOUNTAIN_M = 1000  # m: takeoff altitude above which a flight counts as a mountain flight
DURATION_S = 9000  # s: near the median duration of the declared open flights; splits short and long flights
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
            "above the stricter threshold": int((ratio > REJECTED).sum()),
            "above the stricter threshold (%)": round(100 * float((ratio > REJECTED).mean()), 1)}


if __name__ == "__main__":
    every = closure.flight_table()
    open_ = every[every["declared"] == "open"].set_index("flight")
    if longlags.DT5:
        open_ = open_[open_["dt_native"] <= longlags.MAX_DT + 1e-6].copy()
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(FIXES["para"].parent / "flights_meta.parquet", columns=["flight_id", "alt0", "duration_flight_s"]).astype({"flight_id": str})
    for c in ("alt0", "duration_flight_s"):
        open_[c] = open_["flight_id"].astype(str).map(meta.set_index("flight_id")[c])
    p = longlags.load_pieces()
    masks = longlags.populations(p)
    panels = json.loads((longlags.RESULTS / "longlags.json").read_text())["panels"]
    shortest = {f"lags up to {v['tau_max']} s": v["tau_max"] for v in panels.values()}
    sets = {"every declared open flight": {"all": open_.index.to_numpy()}}
    for name, tau in shortest.items():
        by_pop = {LABEL[k]: np.unique(p["flight"][longlags.long_enough(p, masks[k], tau)]) for k in LABEL}
        sets[name] = {"all": np.unique(np.concatenate(list(by_pop.values()))), **by_pop}
    result = {"min_opening": MIN_OPENING, "stricter": REJECTED, "mountain_m": MOUNTAIN_M, "duration_s": DURATION_S,
              "labels": labels(every),
              "datasets": {name: {k: described(open_["ratio"].reindex(v).dropna()) for k, v in s.items()} for name, s in sets.items()},
              "by takeoff": {f"below {MOUNTAIN_M} m": described(open_.loc[open_["alt0"] < MOUNTAIN_M, "ratio"]),
                             f"{MOUNTAIN_M} m or more": described(open_.loc[open_["alt0"] >= MOUNTAIN_M, "ratio"])},
              "by duration": {f"below {DURATION_S} s": described(open_.loc[open_["duration_flight_s"] < DURATION_S, "ratio"]),
                              f"{DURATION_S} s or more": described(open_.loc[open_["duration_flight_s"] >= DURATION_S, "ratio"])}}
    kinds = {"solo flights": ~open_["group"], "group flights": open_["group"]}
    strata = {f"takeoff below {MOUNTAIN_M} m": open_["alt0"] < MOUNTAIN_M, f"takeoff at {MOUNTAIN_M} m or more": open_["alt0"] >= MOUNTAIN_M,
              f"shorter than {DURATION_S} s": open_["duration_flight_s"] < DURATION_S,
              f"{DURATION_S} s or longer": open_["duration_flight_s"] >= DURATION_S}
    result["by kind"] = {k: {**described(open_.loc[m, "ratio"]), "median duration (s)": float(open_.loc[m, "duration_flight_s"].median()),
                             "mountain (%)": round(100 * float((open_.loc[m, "alt0"] >= MOUNTAIN_M).mean()), 1),
                             "median by stratum": {name: round(float(open_.loc[m & st, "ratio"].median()), 2) for name, st in strata.items()}}
                         for k, m in kinds.items()}
    (longlags.RESULTS / "opening.json").write_text(json.dumps(result, indent=1))
    fig, ax = plt.subplots(figsize=(4.2, 2.9), constrained_layout=True)
    for (name, m), color in zip(kinds.items(), (COLORS["solo"], COLORS["group"])):
        r = open_.loc[m, "ratio"]
        ax.hist(r.clip(0, 1), bins=np.linspace(0, 1, 41), density=True, histtype="step", lw=1.3, color=color,
                label=f"{name} ({len(r):,}), median {r.median():.2f}")
    for x, style in ((MIN_OPENING, "-"), (REJECTED, ":")):
        ax.axvline(x, color="#555555", lw=0.9, ls=style)
    ax.set(xlabel="opening: takeoff to landing / path flown", ylabel="density")
    ax.legend(frameon=False, fontsize=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(longlags.RESULTS / "opening_kind.pdf", bbox_inches="tight")
    fig.savefig(longlags.RESULTS / "opening_kind.png", dpi=200, bbox_inches="tight")
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
    fig.savefig(longlags.RESULTS / "opening.pdf", bbox_inches="tight")
    fig.savefig(longlags.RESULTS / "opening.png", dpi=200, bbox_inches="tight")
