"""Pilots across days: loyalty, clubs and who flies with whom, against the partners of chance.

Population, pair-days and null as partners.py: named paraglider pilots, one pair-day per day and
pair with a pair interval, partners redrawn by the configuration model within each pool of
site-days (N_NULL draws, same seed and same draws as partners.py). Each pilot keeps its number s of
pair-days in every draw; the run counts the draws in which it does not.

D1 loyalty per pilot. For pilots with s >= 2: k = number of distinct partners, real and in each
draw; q = (#{k_draw < k} + U (#{k_draw = k} + 1)) / (N_NULL + 1), the randomized probability
integral transform, U ~ Uniform(0, 1) from its own generator seeded with SEED. The real k and the
draws are exchangeable without loyalty, so q is then exactly uniform (density 1); loyalty moves it
towards 0. Classes of s: tertiles of s, edges in "measured". Per class: the histogram of q (N_BINS
bins), the share of pilots with q < Q_LOW, the median of k / mean(k_draw) and its null value.

D2 loyalty inside and outside clubs. A pair is repeated if it meets on 2 days or more. The
pair-days on repeated pairs, split by the clubs of the two flights that day: same club, different
clubs, club unknown for at least one. Real count, null mean and band, excess (real - null mean) and
its share of the total excess.

D3 who flies with whom. Experience E = earlier flights in the catalog at the date of the flight
(hierarchy.experience), x = log10(1 + E), the pilot's first flight of the day. Each pair-day counts
in both directions. Per decile of own x (edges in "measured"): mean partner x; the Pearson
correlation of x across the two ends. Same for the wing class (WING, 1-5, unknown left out): mean
partner class per own class, share of pair-days with the same class, correlation.

Bands: 5-95% of the draws. The real values are points: partners.py gives them no bootstrap.

    uv run python network.py [--run NAME]    # from this folder, after the runs of partners.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blocks import with_takeoffs
from hierarchy import experience
from partners import TRIES, null_draw, pair_days, pools_of
from tamsd import LABELS, OUT, SEED, save

N_NULL = 100  # as partners.py
S_MIN = 2  # pilots with fewer pair-days have k = s
N_CLASSES_S, N_CLASSES_E = 3, 10  # tertiles of s, deciles of x
N_BINS = 20  # bins of q
Q_LOW = 0.05  # q below it: no loyalty is rejected at 5%
BAND = (5, 95)  # % of the draws
RELATIONS = ["same club", "different clubs", "club unknown"]
ORANGE, GREY = "#eb6834", "#888888"


def pit(k, draws, rng):
    """Randomized PIT of k (n,) among draws (n, N): (#{d < k} + U (#{d = k} + 1)) / (N + 1)."""
    k = np.asarray(k)[:, None]
    u = rng.random(len(k))
    return ((draws < k).sum(1) + u * ((draws == k).sum(1) + 1)) / (draws.shape[1] + 1)


def distinct(df, pilots):
    """Number of distinct partners (over the pair-days of df) and of pair-days of each pilot."""
    pairs = df[["p", "q"]].drop_duplicates()
    ends = lambda d: pd.concat([d["p"], d["q"]]).value_counts().reindex(pilots, fill_value=0).to_numpy()
    return ends(pairs), ends(df)


def attributes(days, flights):
    """Club code (-1 unknown), x = log10(1 + E) and wing class of each pilot on each day."""
    ends = pd.concat([days[["date", "p", "club_p"]].set_axis(["date", "pilot", "club"], axis=1),
                      days[["date", "q", "club_q"]].set_axis(["date", "pilot", "club"], axis=1)])
    club = ends.drop_duplicates(["date", "pilot"]).set_index(["date", "pilot"])["club"]
    x = experience(flights).assign(date=flights.set_index("flight")["date"])
    x = x.groupby(["date", "pilot"]).agg(earlier=("earlier", "min"), wing=("wing", "first"))
    a = pd.DataFrame({"club": club}).join(x, how="left")
    a["club"] = pd.factorize(a["club"])[0]
    a["x"] = np.log10(1 + a.pop("earlier"))
    return a


def ends(df, attrs):
    """Club, x and wing class at the two ends (p, q) of the pair-days of df."""
    out = {}
    for end in ("p", "q"):
        at = attrs.reindex(pd.MultiIndex.from_arrays([df["date"], df[end]]))
        out[end] = {c: at[c].to_numpy() for c in ("club", "x", "wing")}
    return out["p"], out["q"]


def relation(cp, cq):
    """0 same club, 1 different clubs, 2 club unknown for at least one."""
    return np.where((cp < 0) | (cq < 0), 2, np.where(cp == cq, 0, 1))


def mixing(a, b, cls, n_cls):
    """Both directions of the pair-day ends a, b: mean partner value per class of own value (cls
    gives the class of a and b, or None), same-class share, Pearson correlation."""
    own, other = np.r_[a, b], np.r_[b, a]
    c = np.r_[cls[0], cls[1]]
    n = np.bincount(c, minlength=n_cls)
    return {"mean": (np.bincount(c, other, n_cls) / np.where(n, n, np.nan)).tolist(),
            "same": float((c == np.r_[cls[1], cls[0]]).mean()), "corr": float(np.corrcoef(own, other)[0, 1])}


def stats(df, attrs, pilots, edges):
    """Everything one set of pair-days gives: k and s per pilot, pair-days on repeated pairs per
    club relation, and the mixing of x and of the wing class."""
    k, s = distinct(df, pilots)
    p, q = ends(df, attrs)
    rep = df.groupby(["p", "q"])["p"].transform("size").to_numpy() >= 2
    rel = relation(p["club"], q["club"])[rep]
    known = ~np.isnan(p["wing"]) & ~np.isnan(q["wing"])
    w = (p["wing"][known].astype(int) - 1, q["wing"][known].astype(int) - 1)
    return {"k": k, "s": s, "n": len(df), "rep": np.bincount(rel, minlength=3),
            "repeat_pairs": int((df.groupby(["p", "q"]).size() >= 2).sum()),
            "x": mixing(p["x"], q["x"], (np.searchsorted(edges, p["x"], "right"), np.searchsorted(edges, q["x"], "right")),
                        len(edges) + 1),
            "wing": mixing(p["wing"][known], q["wing"][known], w, 5)}


def excess(real, draws):
    """Excess of the real counts over the null mean per relation, its band and its share of the total."""
    mean = draws.mean(0)
    exc = real - mean
    lo, hi = (real - np.percentile(draws, BAND[1], 0), real - np.percentile(draws, BAND[0], 0))
    return {"real": real.tolist(), "null_mean": mean.tolist(),
            "null_band": np.percentile(draws, BAND, 0).tolist(), "excess": exc.tolist(),
            "excess_band": [lo.tolist(), hi.tolist()], "share": (exc / exc.sum()).tolist()}


def band(values):
    """Mean and band over the draws of an (N, ...) array."""
    v = np.array(values, float)
    return {"mean": np.nanmean(v, 0).tolist(), "band": np.nanpercentile(v, BAND, 0).tolist()}


def loyalty(real, draws, rng):
    """D1: per class of s, the histogram of q, share q < Q_LOW, median k / k_null."""
    s = real["s"]
    keep = s >= S_MIN
    k, kd = real["k"][keep], np.array([d["k"][keep] for d in draws]).T
    s = s[keep]
    edges = np.unique(np.quantile(s, np.arange(1, N_CLASSES_S) / N_CLASSES_S, method="lower"))
    cls = np.searchsorted(edges, s, "left")
    q = pit(k, kd, rng)
    ratio, ratio_d = k / kd.mean(1), kd / kd.mean(1, keepdims=True)
    bins = np.linspace(0, 1, N_BINS + 1)
    out = []
    for c in range(len(edges) + 1):
        m = cls == c
        lab = f"s <= {edges[0]}" if c == 0 else f"s > {edges[-1]}" if c == len(edges) else f"{edges[c - 1] + 1} <= s <= {edges[c]}"
        out.append({"label": lab, "pilots": int(m.sum()), "hist": (np.histogram(q[m], bins)[0] / m.sum()).tolist(),
                    "low": float((q[m] < Q_LOW).mean()),
                    "median_ratio": float(np.median(ratio[m])),
                    "median_ratio_null": float(np.mean(np.median(ratio_d[m], 0)))})
    return out, edges.tolist(), len(s)


def network(days, flights, rng, pit_rng):
    pilots = pd.Index(np.unique(days[["p", "q"]].to_numpy()))
    attrs = attributes(days, flights)
    p, q = ends(days, attrs)
    x = np.r_[p["x"], q["x"]]
    e_edges = np.unique(np.quantile(x, np.arange(1, N_CLASSES_E) / N_CLASSES_E))
    real = stats(days, attrs, pilots, e_edges)
    pools = pools_of(days)
    draws = [stats(null_draw(pools, rng), attrs, pilots, e_edges) for _ in range(N_NULL)]
    s_off = [int((d["s"] != real["s"]).sum()) for d in draws]
    d1, s_edges, n_s = loyalty(real, draws, pit_rng)
    cls_x = np.searchsorted(e_edges, x, "right")
    d3 = {"x_mean": (np.bincount(cls_x, x) / np.bincount(cls_x)).tolist(), "x_pairdays": np.bincount(cls_x).tolist()}
    for key in ("x", "wing"):
        d3[key] = {"real": real[key], "null": {f: band([d[key][f] for d in draws]) for f in ("mean", "same", "corr")}}
    return {"params": {"N_NULL": N_NULL, "SEED": SEED, "TRIES": TRIES, "S_MIN": S_MIN, "N_CLASSES_S": N_CLASSES_S,
                       "N_CLASSES_E": N_CLASSES_E, "N_BINS": N_BINS, "Q_LOW": Q_LOW, "BAND": list(BAND)},
            "measured": {"s_edges": s_edges, "x_edges": e_edges.tolist(), "pilots": len(pilots), "pilots_s_min": n_s,
                         "pair_days": len(days), "null_pair_days": [min(d["n"] for d in draws), max(d["n"] for d in draws)],
                         "draws_with_s_changed": int(sum(o > 0 for o in s_off)), "max_pilots_s_changed": max(s_off),
                         "x_missing": int(np.isnan(x).sum()), "repeat_pairs": real["repeat_pairs"],
                         "repeat_pairs_null_mean": float(np.mean([d["repeat_pairs"] for d in draws]))},
            "d1": d1, "d2": {"relations": RELATIONS, **excess(real["rep"], np.array([d["rep"] for d in draws]))},
            "d3": d3}


def plot(r, path):
    """(a) q per class of s against the flat density of no loyalty, (b) excess per club relation, (c) partner x against own x."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.8))
    edges = np.linspace(0, 1, N_BINS + 1)
    a.axhline(1, color=GREY, lw=1, zorder=0)
    for i, cl in enumerate(r["d1"]):
        a.stairs(np.array(cl["hist"]) * N_BINS, edges, color=ORANGE, alpha=[0.45, 0.75, 1.0][i % 3], lw=1.5, label=cl["label"])
    a.set(xlabel="quantile q of the distinct partners", ylabel="density of pilots")
    a.legend(frameon=False, fontsize=8, title="pair-days of a pilot (grey: no loyalty)", title_fontsize=8)
    d2 = r["d2"]
    x, e = np.arange(2), np.array(d2["excess"])[:2]
    lo, hi = np.array(d2["excess_band"])[:, :2]
    b.bar(x, e, color=ORANGE, width=0.6)
    b.errorbar(x, e, yerr=[e - lo, hi - e], fmt="none", ecolor=GREY, capsize=4, lw=1.2)
    b.axhline(0, color=GREY, lw=1)
    b.set_xticks(x, [s.replace(" ", "\n", 1) for s in d2["relations"][:2]])
    b.set(ylabel="excess pair-days on repeated pairs")
    d3 = r["d3"]
    own = np.array(d3["x_mean"])
    nm, (nl, nh) = np.array(d3["x"]["null"]["mean"]["mean"]), d3["x"]["null"]["mean"]["band"]
    c.fill_between(own, nl, nh, color=GREY, alpha=0.3, lw=0)
    c.plot(own, nm, color=GREY, lw=1.2, label="partners at random")
    c.plot(own, d3["x"]["real"]["mean"], color=ORANGE, marker="o", ms=4, lw=1.4, label="observed")
    c.set(xlabel="own log10(1 + earlier flights)", ylabel="partner's")
    c.legend(frameon=False, fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    days = pair_days(pd.read_parquet(LABELS / "pair_intervals.parquet"), flights)
    result = network(days, flights, np.random.default_rng(SEED), np.random.default_rng(SEED))
    (OUT / "network.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "network.pdf")
    print(json.dumps({k: result[k] for k in ("measured", "d1", "d2")}, indent=0), flush=True)
