"""Repeated partners: do pilots fly in a group with the same people more often than chance?

Paraglider pilots are the names of the catalog (anonymised ones left out). Two pilots are partners
on a day when their flights share a pair interval; each pair counts once per day (a pair-day).
Chance keeps who flies in a group where and when: within each pool (the site-days of blocks.sites,
joined when a pair interval links two of them) the partners are redrawn by a configuration model,
each pilot keeping its number of partners that day (N_NULL draws; self and double pairs are
redrawn). Statistics: the share of pair-days on pairs that meet on more than one day, the pairs
meeting on 2 days or more, and the share of pair-days of two pilots of the same club (the club of
each flight in the catalog).

    uv run python partners.py [--run NAME]    # from this folder, about 2 min
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from blocks import sites, with_takeoffs
from mechanisms import ANONYMISED
from tamsd import FIXES, LABELS, OUT, SEED, save

N_NULL, TRIES = 100, 20
DAYS_MAX = 10


def named(flights):
    """Pilot and club of each paraglider flight with a named pilot."""
    catalog = pd.read_csv(FIXES["para"].parent.parent / "catalog/catalog.csv",
                          usecols=["flight_id", "pilot", "club"], dtype=str).drop_duplicates("flight_id")
    catalog = catalog[catalog["pilot"].notna() & ~catalog["pilot"].str.contains(ANONYMISED, regex=False)]
    f = flights.loc[flights["discipline"] == "para", ["flight", "flight_id"]].astype({"flight_id": str})
    return f.merge(catalog, on="flight_id").set_index("flight")


def pair_days(intervals, flights):
    """One row per day and pair of named pilots with a pair interval: date, p, q (p < q), same club,
    pool (the joined site-days)."""
    who = named(flights)
    site = pd.Series(sites(flights["lat0"].to_numpy(), flights["lon0"].to_numpy(), flights["date"].to_numpy()),
                     index=flights["flight"].to_numpy())
    e = intervals[intervals["a"].isin(who.index) & intervals["b"].isin(who.index)]
    pa, pb = who.loc[e["a"], "pilot"].to_numpy(), who.loc[e["b"], "pilot"].to_numpy()
    ca, cb = who.loc[e["a"], "club"].to_numpy(), who.loc[e["b"], "club"].to_numpy()
    sa, sb = site.loc[e["a"]].to_numpy(), site.loc[e["b"]].to_numpy()
    d = pd.DataFrame({"date": e["date"].to_numpy(), "p": np.minimum(pa, pb), "q": np.maximum(pa, pb),
                      "same_club": (ca == cb) & pd.notna(ca), "sa": sa, "sb": sb})
    d = d[d["p"] != d["q"]].drop_duplicates(["date", "p", "q"]).reset_index(drop=True)
    n = int(max(d["sa"].max(), d["sb"].max())) + 1
    d["pool"] = connected_components(coo_matrix((np.ones(len(d)), (d["sa"], d["sb"])), shape=(n, n)), directed=False)[1][d["sa"]]
    return d.drop(columns=["sa", "sb"])


def rewire(pool, rng):
    """The pairs of one pool redrawn with each pilot's number of partners kept: a list of (p, q), p < q."""
    stubs = np.r_[pool["p"].to_numpy(), pool["q"].to_numpy()]
    if len(pool) == 1:
        return list(zip(pool["p"], pool["q"]))
    for _ in range(TRIES):
        s = rng.permutation(stubs)
        a, b = s[0::2], s[1::2]
        pairs = set(zip(np.minimum(a, b), np.maximum(a, b)))
        if (a != b).all() and len(pairs) == len(a):
            return list(pairs)
    return [(x, y) for x, y in pairs if x != y]


def statistics(days):
    """Share of pair-days on pairs met on 2 days or more, number of such pairs, share of pair-days of
    the same club, and the pairs by number of days (1..DAYS_MAX, the last pooled)."""
    per_pair = days.groupby(["p", "q"]).size()
    k = np.minimum(per_pair.to_numpy(), DAYS_MAX)
    return {"repeat_share": float((per_pair[per_pair >= 2]).sum() / per_pair.sum()),
            "repeat_pairs": int((per_pair >= 2).sum()), "same_club": float(days["same_club"].mean()),
            "days_hist": np.bincount(k, minlength=DAYS_MAX + 1)[1:].tolist()}


def partners(intervals, flights, rng):
    days = pair_days(intervals, flights)
    who = named(flights)
    club = who.groupby("pilot")["club"].agg(lambda c: c.mode().iat[0] if c.notna().any() else None)
    observed = statistics(days)
    pools = [g for _, g in days.groupby(["date", "pool"])]
    draws = []
    for _ in range(N_NULL):
        pairs = [(d, p, q) for g in pools for p, q in rewire(g, rng) for d in (g["date"].iat[0],)]
        null = pd.DataFrame(pairs, columns=["date", "p", "q"])
        cp, cq = club.reindex(null["p"]).to_numpy(), club.reindex(null["q"]).to_numpy()
        null["same_club"] = (cp == cq) & pd.notna(cp)
        draws.append(statistics(null))
    # the observed same-club share with each pilot's main club, as in the null
    cp, cq = club.reindex(days["p"]).to_numpy(), club.reindex(days["q"]).to_numpy()
    observed["same_club_main"] = float(((cp == cq) & pd.notna(cp)).mean())
    out = {"pair_days": len(days), "pilots": int(pd.unique(days[["p", "q"]].to_numpy().ravel()).size),
           "pairs": int(days.groupby(["p", "q"]).ngroups), "pools": len(pools),
           "max_days": int(days.groupby(["p", "q"]).size().max()), "observed": observed, "null": {}}
    for key in ("repeat_share", "repeat_pairs", "same_club"):
        v = np.array([d[key] for d in draws], float)
        out["null"][key] = {"mean": float(v.mean()), "band": np.percentile(v, [5, 95]).tolist()}
    out["null"]["days_hist"] = np.mean([d["days_hist"] for d in draws], axis=0).tolist()
    return out


def plot(r, path):
    """Left: pairs by number of days together, observed and at random; right: the three shares."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    k = np.arange(1, DAYS_MAX + 1)
    a.plot(k, r["observed"]["days_hist"], color="#eb6834", marker="o", ms=5, lw=1.4, label="observed")
    a.plot(k, r["null"]["days_hist"], color="#2a78d6", marker="s", ms=5, lw=1.4, ls="--", label="partners at random")
    a.set(yscale="log", xlabel=f"days together (last: {DAYS_MAX}+)", ylabel="pairs of pilots")
    a.set_xticks(k)
    a.legend(frameon=False, fontsize=8)
    keys = [("repeat_share", "pair-days on\nrepeated pairs"), ("same_club", "pair-days of\nthe same club")]
    x = np.arange(len(keys))
    obs = [r["observed"]["repeat_share"], r["observed"]["same_club_main"]]
    null = np.array([r["null"][key]["mean"] for key, _ in keys])
    band = np.array([r["null"][key]["band"] for key, _ in keys]).T
    b.bar(x - 0.18, obs, width=0.34, color="#eb6834", label="observed")
    b.bar(x + 0.18, null, width=0.34, color="#2a78d6", label="partners at random")
    b.errorbar(x + 0.18, null, yerr=[null - band[0], band[1] - null], fmt="none", ecolor="#333333", lw=1)
    b.set_xticks(x, [lab for _, lab in keys])
    b.set(ylabel="share", ylim=(0, 1))
    b.legend(frameon=False, fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    result = partners(pd.read_parquet(LABELS / "pair_intervals.parquet"), flights, np.random.default_rng(SEED))
    (OUT / "partners.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "partners.pdf")
    print(json.dumps({k: v for k, v in result.items() if k not in ("null",)}, indent=0)[:1200],
          {k: v for k, v in result["null"].items() if k != "days_hist"}, flush=True)
