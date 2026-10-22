"""Hierarchy: who is ahead of whom, in groups of 3 or more, over time, and who leads.

Order. Inside a configuration of 3 pilots or more, each edge (pair interval) points from the pilot
ahead to the one behind when its signed median delay is at least ABREAST s (mechanisms.py);
abreast edges have no direction. Over the stretches of constant edges of chains.py: directed
triangles are transitive (A before B before C, A before C) or cyclic (A before B before C before
A); a graph with every edge directed is acyclic or not, and an acyclic one has one leader (one
member with nobody ahead) or several. Against the same graphs with each direction drawn at random
(N_NULL draws): a cyclic triangle then has probability 1/4.

Switches. Inside the pair intervals of cohesion.py, the delay l*(t) (outside searches) says who is
ahead at each instant: |l*| >= ABREAST s, or abreast. Runs shorter than MIN_RUN s are dropped;
a switch is a change of the pilot ahead. Per hour of contact, and the share of the directed time
of the pilot most often ahead, real against chance contacts lasting at least tau*.

Who leads. In configurations of two named pilots with a clear leader: how often the leader is the
more experienced (more earlier flights in the catalog) or flies the higher wing class (A=1 to
competition=5), among the pairs that differ. Bands: 5-95% of N_BOOT draws of the blocks of the
note (order, who leads) or of the dates (switches).

    uv run python hierarchy.py [--run NAME]   # from this folder, after chains.py and cohesion.py
"""

import json
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cohesion import CACHE as COHESION
from mechanisms import ABREAST, ANONYMISED, roles
from tamsd import FIXES, LABELS, OUT, SEED, save

N_NULL, N_BOOT = 50, 200
MIN_RUN = 60  # s
WING = {"A ou 1": 1, "B ou 1-2": 2, "C ou 2": 3, "D ou 2-3": 4, "CIVL Competition Class": 5,
        "Open Class certification": 5}


def directed(edges):
    """(ahead, behind) of each edge with |ell_med| >= ABREAST; the others are abreast."""
    out = []
    for a, b, ell in edges:
        if ell >= ABREAST:
            out.append((a, b))
        elif ell <= -ABREAST:
            out.append((b, a))
    return out


def order(n, arcs):
    """Whether the graph of `arcs` on members 0..n-1 is acyclic, its number of leaders (members with
    nobody ahead) and its depth (the longest chain of arcs); None for the last two if cyclic."""
    ahead = [0] * n
    nxt = [[] for _ in range(n)]
    for a, b in arcs:
        ahead[b] += 1
        nxt[a].append(b)
    leaders = sum(1 for k in range(n) if ahead[k] == 0)
    depth, level = 0, [0] * n
    queue = [k for k in range(n) if ahead[k] == 0]
    seen = 0
    while queue:
        u = queue.pop()
        seen += 1
        for v in nxt[u]:
            level[v] = max(level[v], level[u] + 1)
            depth = max(depth, level[v])
            ahead[v] -= 1
            if ahead[v] == 0:
                queue.append(v)
    return (True, leaders, depth) if seen == n else (False, None, None)


def triangle_list(edges):
    """The triples of members joined pairwise by an edge."""
    linked = {frozenset((a, b)) for a, b, _ in edges}
    nodes = sorted({x for a, b, _ in edges for x in (a, b)})
    return [(x, y, z) for x, y, z in combinations(nodes, 3)
            if {frozenset((x, y)), frozenset((y, z)), frozenset((x, z))} <= linked]


def triangles(tris, arcs):
    """Among the triangles `tris`, those with three directed edges: (transitive, cyclic) counts."""
    arc = set(arcs)
    t = c = 0
    for x, y, z in tris:
        d = [(u, v) for u, v in ((x, y), (y, x), (y, z), (z, y), (x, z), (z, x)) if (u, v) in arc]
        if len(d) == 3:
            outs = [sum(1 for u, _ in d if u == k) for k in (x, y, z)]
            c, t = (c + 1, t) if outs == [1, 1, 1] else (c, t + 1)
    return t, c


def stretches(confs, intervals):
    """Per stretch of constant edges of every configuration of 3 pilots or more: conf_id, episode,
    n, members, duration, and its edges (a, b, ell_med) as member indices."""
    by_date = dict(tuple(intervals.groupby("date")))
    rows = []
    for conf in confs[confs["n"] >= 3].itertuples():
        day = by_date[conf.date]
        e = day[day["a"].isin(conf.members) & day["b"].isin(conf.members)]
        e = e[(e["t_on"] < conf.t_end) & (e["t_off"] > conf.t_start)]
        idx = {f: k for k, f in enumerate(conf.members)}
        cuts = np.unique(np.clip(np.r_[conf.t_start, conf.t_end, e["t_on"], e["t_off"]], conf.t_start, conf.t_end))
        for a, b in zip(cuts[:-1], cuts[1:]):
            live = e[(e["t_on"] <= a) & (e["t_off"] >= b)]
            edges = [(idx[x], idx[y], ell) for x, y, ell in zip(live["a"], live["b"], live["ell_med"])]
            rows.append((conf.conf_id, conf.episode_id, conf.n, conf.members, b - a, edges))
    return pd.DataFrame(rows, columns=["conf_id", "episode_id", "n", "members", "duration", "edges"])


def measure(st, rng):
    """Per stretch, observed and mean over N_NULL random orientations: directed triangles
    (transitive, cyclic) and, when every edge is directed, acyclic, one leader, depth."""
    rows = []
    for s in st.itertuples():
        arcs = directed(s.edges)
        tris = triangle_list(s.edges)
        t, c = triangles(tris, arcs)
        full = len(arcs) == len(s.edges)
        acyc, leaders, depth = order(s.n, arcs) if full else (None, None, None)
        null = []
        for _ in range(N_NULL):
            flip = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in arcs]
            nt, nc = triangles(tris, flip)
            na, nl, _ = order(s.n, flip) if full else (None, None, None)
            null.append((nc, nt + nc, na, nl == 1 if na else False))
        null = np.array(null, float)
        rows.append((s.conf_id, s.episode_id, s.n, s.duration, t, c, full, acyc, leaders, depth,
                     null[:, 0].mean(), null[:, 1].mean(), null[:, 2].mean() if full else np.nan,
                     null[:, 3].mean() if full else np.nan))
    return pd.DataFrame(rows, columns=["conf_id", "episode_id", "n", "duration", "transitive", "cyclic", "full",
                                       "acyclic", "leaders", "depth", "null_cyclic", "null_triangles",
                                       "null_acyclic", "null_one_leader"])


def order_summary(m, block, rng):
    """Shares weighted by duration with block bands: cyclic among directed triangles, acyclic and
    one leader among the fully directed graphs, observed and at random; and the depth."""
    m = m.assign(block=block.reindex(m["episode_id"]).to_numpy(), w=m["duration"] / 3600)
    full = m[m["full"]]
    acyc = full[full["acyclic"] == True]  # noqa: E712
    ids, k_all = np.unique(m["block"], return_inverse=True)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)

    def share(num, den, rows):
        k = np.searchsorted(ids, rows["block"])
        a, b = np.bincount(k, num, len(ids)), np.bincount(k, den, len(ids))
        draws = (w @ a) / (w @ b)
        return {"value": float(a.sum() / b.sum()), "band": np.percentile(draws, [5, 95]).tolist()}

    tri = m["transitive"] + m["cyclic"]
    out = {"stretches": len(m), "hours": float(m["w"].sum()), "full_share": float(full["w"].sum() / m["w"].sum()),
           "triangles": float((tri * m["w"]).sum()),
           "cyclic": share(m["cyclic"] * m["w"], tri * m["w"], m),
           "cyclic_null": share(m["null_cyclic"] * m["w"], m["null_triangles"] * m["w"], m),
           "acyclic": share(full["acyclic"].astype(float) * full["w"], full["w"], full),
           "acyclic_null": share(full["null_acyclic"] * full["w"], full["w"], full),
           "one_leader": share((acyc["leaders"] == 1) * acyc["w"], acyc["w"], acyc),
           "one_leader_null": share(full["null_one_leader"] * full["w"], full["null_acyclic"] * full["w"], full)}
    out["by_size"] = {}
    for n, g in acyc.assign(k=np.minimum(acyc["n"], 5)).groupby("k"):
        out["by_size"][str(n)] = {"hours": float(g["w"].sum()),
                                  "one_leader": float(((g["leaders"] == 1) * g["w"]).sum() / g["w"].sum()),
                                  "line": float(((g["depth"] == g["n"] - 1) * g["w"]).sum() / g["w"].sum())}
    return out


def runs(states, step):
    """The pilot ahead (+1 or -1) as runs, those shorter than MIN_RUN s dropped and neighbours of one
    sign joined: list of (sign, instants)."""
    s = states[states != 0]
    if not len(s):
        return []
    cut = np.flatnonzero(np.diff(s)) + 1
    r = [(int(x[0]), len(x)) for x in np.split(s, cut)]
    r = [x for x in r if x[1] * step >= MIN_RUN]
    joined = []
    for sign, n in r:
        if joined and joined[-1][0] == sign:
            joined[-1] = (sign, joined[-1][1] + n)
        else:
            joined.append((sign, n))
    return joined


def switches(days, step, tau_star, rng):
    """Per kind (real, chance lasting at least tau*): switches per hour of contact and the median
    share of the directed time of the pilot most often ahead, with date bands."""
    per = {"real": [], "chance": []}
    dominant = {"real": [], "chance": []}
    for path in days:
        r = pickle.loads(path.read_bytes())
        for kind in per:
            n_sw = hours = 0.0
            for _, _, on, off, t, lag, search in r[kind]:
                if off - on < tau_star:
                    continue
                ell = lag[~search].astype(int) * step
                states = np.where(ell >= ABREAST, 1, np.where(ell <= -ABREAST, -1, 0))
                rr = runs(states, step)
                n_sw += max(len(rr) - 1, 0)
                hours += (off - on) / 3600
                total = sum(n for _, n in rr)
                if total:
                    dominant[kind].append(max(sum(n for s, n in rr if s == 1), sum(n for s, n in rr if s == -1)) / total)
            per[kind].append((n_sw, hours))
    w = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=N_BOOT).astype(float)
    out = {}
    for kind, v in per.items():
        v = np.array(v)
        out[kind] = {"intervals": len(dominant[kind]), "hours": float(v[:, 1].sum()),
                     "per_hour": float(v[:, 0].sum() / v[:, 1].sum()),
                     "per_hour_band": np.percentile((w @ v[:, 0]) / (w @ v[:, 1]), [5, 95]).tolist(),
                     "dominant_median": float(np.median(dominant[kind])),
                     "never_switch": float(np.mean(np.array(dominant[kind]) == 1)),
                     "dominant": np.histogram(dominant[kind], bins=np.linspace(0.5, 1, 11))[0].tolist()}
    return out


def experience(flights):
    """Earlier flights of the pilot in the catalog and wing class (1-5, NaN if not rated) of each
    paraglider flight, with its pilot (NaN if anonymised)."""
    catalog = pd.read_csv(FIXES["para"].parent.parent / "catalog/catalog.csv",
                          usecols=["flight_id", "date", "pilot", "wing_class"], dtype=str)
    catalog = catalog[catalog["pilot"].notna() & ~catalog["pilot"].str.contains(ANONYMISED, regex=False)]
    catalog = catalog.drop_duplicates("flight_id").sort_values(["pilot", "date", "flight_id"])
    catalog["earlier"] = catalog.groupby("pilot").cumcount()
    catalog["wing"] = catalog["wing_class"].map(WING)
    f = flights.loc[flights["discipline"] == "para", ["flight", "flight_id"]].astype({"flight_id": str})
    return f.merge(catalog[["flight_id", "pilot", "earlier", "wing"]], on="flight_id").set_index("flight")


def who_leads(confs, pairs, flights, episodes, rng):
    """In configurations of two named pilots with a clear leader: the share in which the leader has
    more earlier flights, or the higher wing class, among the pairs that differ."""
    r = roles(confs, pairs, flights)
    r = r[r["population"] != "abreast"]
    x = experience(flights)
    lead = r[r["population"] == "leader"].set_index("conf")
    follow = r[r["population"] == "follower"].set_index("conf")
    both = lead.join(follow, lsuffix="_l", rsuffix="_f", how="inner")
    both = both[both["flight_l"].isin(x.index) & both["flight_f"].isin(x.index)]
    xl, xf = x.loc[both["flight_l"]], x.loc[both["flight_f"]]
    both = both.assign(pilot_l=xl["pilot"].to_numpy(), pilot_f=xf["pilot"].to_numpy(),
                       earlier_l=xl["earlier"].to_numpy(), earlier_f=xf["earlier"].to_numpy(),
                       wing_l=xl["wing"].to_numpy(), wing_f=xf["wing"].to_numpy())
    both = both[both["pilot_l"] != both["pilot_f"]]
    block = episodes.set_index("episode_id")["block"].reindex(both["episode_l"]).to_numpy()
    ids, k = np.unique(block, return_inverse=True)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)
    out = {"pairs": len(both)}
    for name, a, b in (("experience", "earlier_l", "earlier_f"), ("wing", "wing_l", "wing_f")):
        differ = (both[a] != both[b]) & both[a].notna() & both[b].notna()
        hit = (both[a] > both[b]) & differ
        num, den = np.bincount(k, hit, len(ids)), np.bincount(k, differ, len(ids))
        out[name] = {"pairs": int(differ.sum()), "value": float(num.sum() / den.sum()),
                     "band": np.percentile((w @ num) / (w @ den), [5, 95]).tolist()}
    ratio = np.log10((both["earlier_l"] + 1) / (both["earlier_f"] + 1))
    out["experience"]["median_ratio"] = float(10 ** np.median(ratio))
    return out


def plot(r, path):
    """Left: order in groups of 3+, observed and at random; middle: switches; right: who leads."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    o = r["order"]
    keys = [("cyclic", "cyclic triangles"), ("acyclic", "acyclic graphs"), ("one_leader", "one leader")]
    x = np.arange(len(keys))
    for dx, suffix, color, label in ((-0.18, "", "#eb6834", "observed"), (0.18, "_null", "#2a78d6", "directions at random")):
        v = np.array([o[k + suffix]["value"] for k, _ in keys])
        band = np.array([o[k + suffix]["band"] for k, _ in keys]).T
        a.bar(x + dx, v, width=0.34, color=color, label=label)
        a.errorbar(x + dx, v, yerr=[v - band[0], band[1] - v], fmt="none", ecolor="#333333", lw=1)
    a.set_xticks(x, [lab for _, lab in keys])
    a.set(ylabel="share of the time", ylim=(0, 1))
    a.legend(frameon=False, fontsize=8)
    s = r["switches"]
    for k, (kind, color, label) in enumerate((("real", "#eb6834", "groups"), ("chance", "#2a78d6", r"chance, $\geq\tau^*$"))):
        h = np.array(s[kind]["dominant"], float)
        centres = np.linspace(0.525, 0.975, 10)
        b.step(centres, h / h.sum(), where="mid", color=color, lw=1.4,
               label=f"{label}: {s[kind]['per_hour']:.2f} switches/h")
    b.set(xlabel="share of the directed time of the pilot most often ahead", ylabel="share of the contacts")
    b.legend(frameon=False, fontsize=8)
    wl = r["who_leads"]
    keys = [("experience", "more earlier flights"), ("wing", "higher wing class")]
    v = np.array([wl[k]["value"] for k, _ in keys])
    band = np.array([wl[k]["band"] for k, _ in keys]).T
    c.bar(np.arange(2), v, width=0.5, color="#1baf7a")
    c.errorbar(np.arange(2), v, yerr=[v - band[0], band[1] - v], fmt="none", ecolor="#333333", lw=1)
    c.axhline(0.5, color="#999999", ls=":", lw=1)
    c.set_xticks(np.arange(2), [lab for _, lab in keys])
    c.set(ylabel="share of the pairs where the leader has", ylim=(0.4, 0.7))
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    confs = pd.read_parquet(LABELS / "configurations.parquet")
    intervals = pd.read_parquet(LABELS / "pair_intervals.parquet")
    flights = pd.read_parquet(LABELS / "flights.parquet")
    episodes = pd.read_parquet(LABELS / "episodes.parquet")
    block = episodes.set_index("episode_id")["block"]
    m = measure(stretches(confs, intervals), rng)
    days = sorted(p for p in (COHESION / "all" / "days").glob("*.pkl") if not p.name.startswith("."))
    p = json.loads((LABELS / "days" / "params.json").read_text())
    tau_star = json.loads((LABELS / "calibration.json").read_text())["tau_star"]
    result = {"order": order_summary(m, block, rng), "switches": switches(days, p["step"], tau_star, rng),
              "who_leads": who_leads(confs, intervals, flights, episodes, rng)}
    (OUT / "hierarchy.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "hierarchy.pdf")
    print(json.dumps({k: v for k, v in result["order"].items() if k != "by_size"}, indent=0), result["order"]["by_size"],
          {k: {kk: vv for kk, vv in v.items() if kk != "dominant"} for k, v in result["switches"].items()},
          result["who_leads"], sep="\n", flush=True)
