"""Chains or compact groups: the shape of the configurations of each group.

Inside a group the members are fixed but its edges, the pair intervals among them, can start
and end: each edge set is one configuration. In a configuration the graph has m edges among n
members: density m / (n (n - 1) / 2) and diameter (the longest shortest path, in edges). Shapes:
complete (every
member in contact with every other), tree (m = n - 1, the fewest edges that keep n members
connected: a chain or a star), between (anything else). Shares are of the time spent at each
size (8+ pooled).

The group effect by shape: the TA-MSD pieces of size.py (paragliders) lie inside one
group; each takes the shape of the configuration at its middle. H and H_loc(1000 s) of triangles
against chains of 3 pilots, and of complete against tree graphs of 4 pilots or more, with the
bands of the blocks of the note (tamsd.summarize).

    uv run python chains.py [--run NAME]    # from this folder, after size.py; about 1 min
"""

import json
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import size
from tamsd import CACHE, LABELS, LAGS, OUT, SEED, read_pieces, save, summarize

N_MAX = 8
SHAPES = ("complete", "between", "tree")
COLORS = dict(zip(SHAPES, ["#2a78d6", "#8e5bd0", "#eb6834"]))
CLASSES = {"n3 triangle": "3 pilots, triangle", "n3 chain": "3 pilots, chain",
           "n4+ complete": "4+ pilots, complete", "n4+ tree": "4+ pilots, tree"}


def diameter(n, edges):
    """Longest shortest path between members 0..n-1 joined by `edges` (inf if not connected)."""
    nb = [[] for _ in range(n)]
    for a, b in edges:
        nb[a].append(b)
        nb[b].append(a)
    worst = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if min(dist) < 0:
            return np.inf
        worst = max(worst, max(dist))
    return worst


def shape(n, m):
    return "complete" if m == n * (n - 1) // 2 else "tree" if m == n - 1 else "between"


def pieces_of(group, edges):
    """The configurations of one group, its stretches with a constant edge set: (t0, t1, m,
    diameter)."""
    members = {f: k for k, f in enumerate(group.members)}
    t0, t1 = group.t_start, group.t_end
    e = edges[(edges["t_on"] < t1) & (edges["t_off"] > t0)]
    cuts = np.unique(np.clip(np.r_[t0, t1, e["t_on"], e["t_off"]], t0, t1))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        live = e[(e["t_on"] <= a) & (e["t_off"] >= b)]
        pairs = [(members[x], members[y]) for x, y in zip(live["a"], live["b"])]
        out.append((a, b, len(pairs), diameter(len(members), pairs)))
    return out


def graphs(groups, intervals):
    """Per configuration of every group of 3 pilots or more: group_id, n,
    t0, t1, m, diameter, shape."""
    by_date = dict(tuple(intervals.groupby("date")))
    rows = []
    for group in groups[groups["n"] >= 3].itertuples():
        day = by_date[group.date]
        mine = day[day["a"].isin(group.members) & day["b"].isin(group.members)]
        for t0, t1, m, d in pieces_of(group, mine):
            rows.append((group.group_id, group.n, t0, t1, m, d))
    g = pd.DataFrame(rows, columns=["group_id", "n", "t0", "t1", "m", "diameter"])
    g["shape"] = [shape(n, m) for n, m in zip(g["n"], g["m"])]
    return g


def shares(g):
    """Per size 3..N_MAX (pooled above): hours, share of time of each shape, mean density, share
    of time with diameter 1, 2, 3 and 4 or more, and share of stretches not connected."""
    g = g.assign(w=(g["t1"] - g["t0"]) / 3600, k=np.minimum(g["n"], N_MAX),
                 density=g["m"] / (g["n"] * (g["n"] - 1) / 2))
    connected = np.isfinite(g["diameter"])
    out = {"sizes": list(range(3, N_MAX + 1)), "hours": [], "shape": {s: [] for s in SHAPES}, "density": [],
           "diameter": {str(d): [] for d in (1, 2, 3, 4)}, "not_connected": float(1 - np.average(connected, weights=g["w"]))}
    for k in out["sizes"]:
        x = g[(g["k"] == k) & connected]
        w = x["w"].sum()
        out["hours"].append(float(w))
        for s in SHAPES:
            out["shape"][s].append(float(x.loc[x["shape"] == s, "w"].sum() / w))
        out["density"].append(float(np.average(x["density"], weights=x["w"])))
        for d in (1, 2, 3):
            out["diameter"][str(d)].append(float(x.loc[x["diameter"] == d, "w"].sum() / w))
        out["diameter"]["4"].append(float(x.loc[x["diameter"] >= 4, "w"].sum() / w))
    return out


def classify(p, flights, membership, g):
    """The class of CLASSES of each piece of size.py (None outside them), from its configuration at
    the middle of the piece."""
    utc0 = flights.set_index("flight")["utc0"]
    mid = utc0.reindex(p["flight"]).to_numpy() + p["start"] + p["duration"] / 2
    mb = membership[membership["state"] == "G"]
    mb = pd.DataFrame({"flight": mb["flight"].astype(np.int64), "t_start": mb["t_start"].astype(float),
                       "t_end": mb["t_end"].astype(float), "group_id": mb["group_id"].astype(np.int64),
                       "n": mb["n"].astype(np.int64)}).sort_values("t_start")
    pieces = pd.DataFrame({"flight": p["flight"].astype(np.int64), "t": mid.astype(float),
                           "row": np.arange(len(mid))}).sort_values("t")
    found = pd.merge_asof(pieces, mb, left_on="t", right_on="t_start", by="flight")
    found = found[found["t"] < found["t_end"]]
    stretches = g[["group_id", "t0", "t1", "shape"]].astype({"group_id": np.int64, "t0": float, "t1": float})
    stretch = pd.merge_asof(found.sort_values("t"), stretches.sort_values("t0"), left_on="t", right_on="t0", by="group_id")
    stretch = stretch[stretch["t"] < stretch["t1"]]
    label = np.full(len(mid), None, object)
    n, s, rows = stretch["n"].to_numpy(), stretch["shape"].to_numpy(), stretch["row"].to_numpy()
    label[rows[(n == 3) & (s == "complete")]] = "n3 triangle"
    label[rows[(n == 3) & (s == "tree")]] = "n3 chain"
    label[rows[(n >= 4) & (s == "complete")]] = "n4+ complete"
    label[rows[(n >= 4) & (s == "tree")]] = "n4+ tree"
    return label


def effect(flights, membership, summary, g, rng):
    """TA-MSD, H and H_loc of the pieces of each class of CLASSES (paragliders)."""
    p = read_pieces(size.SIZE_CACHE["pieces"])["para"]
    p = {**p, "population": classify(p, flights, membership, g)}
    out = summarize(p, summary, rng, populations=CLASSES)
    k = int(np.searchsorted(LAGS, 1000))
    for r in out.values():
        r["H_loc_1000"], r["H_loc_1000_band"] = r["H_loc"][k], [r["H_loc_band"][0][k], r["H_loc_band"][1][k]]
    return out


def plot(r, path):
    """Left: share of time of each shape by size; middle: share of time by diameter; right: H_loc by class."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    sh = r["shares"]
    x = np.array(sh["sizes"])
    bottom = np.zeros(len(x))
    for s in SHAPES:
        a.bar(x, sh["shape"][s], bottom=bottom, color=COLORS[s], width=0.7, label=s, edgecolor="white", lw=1)
        bottom += sh["shape"][s]
    a.set(xlabel=f"pilots (last: {N_MAX}+)", ylabel="share of the time", ylim=(0, 1))
    a.legend(frameon=False, fontsize=8, loc="lower left")
    bottom = np.zeros(len(x))
    for d, color in zip(("1", "2", "3", "4"), ["#2a78d6", "#1baf7a", "#d4a017", "#eb6834"]):
        b.bar(x, sh["diameter"][d], bottom=bottom, color=color, width=0.7, edgecolor="white", lw=1,
              label=f"diameter {d}" + ("+" if d == "4" else ""))
        bottom += sh["diameter"][d]
    b.set(xlabel=f"pilots (last: {N_MAX}+)", ylabel="share of the time", ylim=(0, 1))
    b.legend(frameon=False, fontsize=8, loc="lower left")
    for key, color, ls in [("n3 triangle", "#2a78d6", "-"), ("n3 chain", "#eb6834", "-"),
                           ("n4+ complete", "#2a78d6", "--"), ("n4+ tree", "#eb6834", "--")]:
        e = r["effect"].get(key)
        if e is None:
            continue
        lags = np.array(e["lags"])
        c.fill_between(lags, *np.array(e["H_loc_band"]), color=color, alpha=0.15, lw=0)
        c.plot(lags, e["H_loc"], color=color, ls=ls, lw=1.3, label=f"{CLASSES[key]} ({e['flights']})")
    c.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$")
    c.legend(frameon=False, fontsize=7)
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_parquet(LABELS / "groups.parquet")
    intervals = pd.read_parquet(LABELS / "pair_intervals.parquet")
    flights = pd.read_parquet(LABELS / "flights.parquet")
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    g = graphs(groups, intervals)
    g.to_parquet(CACHE / "chain_graphs.parquet", index=False)
    result = {"shares": shares(g), "effect": effect(flights, membership, summary, g, np.random.default_rng(SEED))}
    (OUT / "chains.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "chains.pdf")
    print(json.dumps(result["shares"], indent=0)[:1500], flush=True)
    print({k: (v["flights"], round(v["H"], 3), round(v["H_loc_1000"], 3)) for k, v in result["effect"].items()})
