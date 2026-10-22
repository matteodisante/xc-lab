"""How groups form and break, who flies ahead, and when in the day flights are in a group.

Entries and exits. A flight enters a group where a stretch of group time (tamsd.stretches) starts
after a stretch alone, and leaves where one ends before a stretch alone; group time that starts
at the departure or ends at the landing is counted apart. The phase at that instant is that of
the Vilpellet run holding it (1 Hz flights), against the share of time in each phase of the same
flights alone and in a group (runs placed by their middle, as in survival.py).

Leaders and followers. In a configuration of two pilots the signed median delay of their pair
interval (ell_med, positive when the second flight follows the first, from the instants outside
a search) tells who flies ahead; |ell_med| < ABREAST s counts as abreast. Leader and follower are
compared within each configuration (decompose.effect with the configuration as stratum): the
budget and performance statistics and the TA-MSD of the pieces of 2 pilots (size.py). Whether
some pilots lead more often than chance: the dispersion of the lead fractions of the pilots with
at least MIN_ROLES roles, against N_PERM draws that swap the two roles of every configuration
with probability 1/2.

Hours. The time of solo flights, of the group flights alone and in a group, over the local hour
(Europe/Paris), sampled every SAMPLE_S s.

Bands: 5-95% of N_BOOT draws of the blocks of the note (decompose.N_BOOT, decompose.block_counts).

    uv run python mechanisms.py [--run NAME]    # after size.py and performance.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget
import decompose
import size
from phase_msd import RUNS
from survival import assign
from tamsd import CACHE, COLORS, FIXES, LABELS, OUT, SEED, read_pieces, save, stretches

ABREAST = 15  # s
MIN_ROLES, N_PERM = 10, 1000
SAMPLE_S = 300
PHASES = ("climb", "search", "transition", "none")
ANONYMISED = "$2A$07$FFVL"


def events(spans):
    """Entries and exits of group time: flight, kind (entry, exit, at departure, at landing), t on
    the flight clock."""
    s = spans.sort_values(["flight", "a"], kind="stable")
    before = s.groupby("flight")["population"].shift(1)
    after = s.groupby("flight")["population"].shift(-1)
    g = (s["population"] == "group").to_numpy()
    parts = [s[g & before.eq("group_solo").to_numpy()].assign(kind="entry", t=lambda d: d["a"]),
             s[g & before.isna().to_numpy()].assign(kind="at departure", t=lambda d: d["a"]),
             s[g & after.eq("group_solo").to_numpy()].assign(kind="exit", t=lambda d: d["b"]),
             s[g & after.isna().to_numpy()].assign(kind="at landing", t=lambda d: d["b"])]
    return pd.concat(parts, ignore_index=True)[["flight", "kind", "t"]]


def phase_at(points, runs):
    """The phase of the run of the flight that holds each point (flight, t), "none" outside every run."""
    p = points.reset_index(drop=True).assign(order=lambda d: np.arange(len(d))).sort_values("t", kind="stable")
    r = runs[["flight", "t_start", "t_end", "phase"]].sort_values("t_start", kind="stable")
    m = pd.merge_asof(p, r, left_on="t", right_on="t_start", by="flight", direction="backward")
    phase = np.where(m["t"].to_numpy() <= m["t_end"].to_numpy(), m["phase"].to_numpy(object), "none")
    return pd.Series(phase, index=m["order"].to_numpy()).sort_index().to_numpy()


def shares(labels, weights, block, counts, categories=PHASES):
    """Share of each category among rows (`labels` with `weights`), point and band, resampling the blocks."""
    out = {}
    for c in categories:
        num, den = np.bincount(block, weights * (labels == c)), np.bincount(block, weights)
        values = [num.sum() / den.sum()] + [(k[: len(num)] @ num) / (k[: len(den)] @ den) for k in counts]
        out[c] = decompose.described(values)
    return out


def entries_and_exits(flights, spans, runs, all_runs, block, counts):
    """Phase at entry and exit, read on every run of the flight (`all_runs`, including those that
    start before the departure), against the time shares of the same flights alone and in a group
    (`runs`, placed in the stretches by their middle)."""
    labelled = flights.loc[flights["labelled"] & (flights["discipline"] == "para"), "flight"]
    ev = events(spans[spans["flight"].isin(labelled)])
    ev["phase"] = phase_at(ev, all_runs)
    b = block.reindex(ev["flight"]).to_numpy()
    departure = (flights.set_index("flight")["departure"] - flights.set_index("flight")["utc0"]).reindex(ev["flight"])
    entry = (ev["kind"] == "entry").to_numpy()
    out = {"counts": ev["kind"].value_counts().to_dict(),
           "entries within 10 min of the departure": float(((ev["t"].to_numpy() - departure.to_numpy())[entry] < 600).mean())}
    for kind in ("entry", "exit"):
        m = (ev["kind"] == kind).to_numpy()
        out[kind] = shares(ev["phase"].to_numpy(object)[m], np.ones(m.sum()), b[m], counts)
    grouped = runs[runs["flight"].isin(ev["flight"].unique())]
    for pop, name in (("group_solo", "time alone"), ("group", "time in a group")):
        r = grouped[grouped["population"] == pop]
        out[name] = shares(r["phase"].to_numpy(object), r["n_fixes"].to_numpy(float), block.reindex(r["flight"]).to_numpy(),
                           counts, PHASES[:3])
    return out


def roles(configurations, pairs, flights):
    """Leader, follower or abreast for each member of every configuration of two pilots, as
    stretches on the flight clock: flight, population, episode, conf, a, b, delay (ell_med)."""
    two = configurations[configurations["n"] == 2].copy()
    two[["x", "y"]] = np.vstack(two["members"].to_numpy())
    m = two.merge(pairs, left_on=["x", "y"], right_on=["a", "b"])
    m = m[(m["t_on"] <= m["t_start"]) & (m["t_end"] <= m["t_off"])]
    ell = m["ell_med"].to_numpy()
    lead = np.where(ell >= ABREAST, m["x"], np.where(ell <= -ABREAST, m["y"], -1))
    utc0 = flights.set_index("flight")["utc0"]
    rows = []
    for who, other in (("x", "y"), ("y", "x")):
        f = m[who].to_numpy()
        role = np.where(lead < 0, "abreast", np.where(lead == f, "leader", "follower"))
        rows.append(pd.DataFrame({"flight": f, "population": role, "episode": m["episode_id"].to_numpy(),
                                  "conf": m["conf_id"].to_numpy(), "partner": m[other].to_numpy(),
                                  "a": m["t_start"].to_numpy() - utc0.reindex(f).to_numpy(),
                                  "b": m["t_end"].to_numpy() - utc0.reindex(f).to_numpy(), "delay": np.abs(ell)}))
    return pd.concat(rows, ignore_index=True)


def by_middle(table, spans):
    """Rows of `table` (flight, t_mid) placed in the stretch of `spans` holding t_mid, with its columns."""
    t = table.drop(columns=[c for c in ("population",) if c in table]).assign(t_start=table["t_mid"], t_end=table["t_mid"])
    return assign(t, spans)


def leaders(flights, configurations, pairs, runs, terms, pieces, info, block, counts):
    """Leader minus follower within each configuration of two pilots, and the delays."""
    r = roles(configurations, pairs, flights)
    clear = r[r["population"] != "abreast"]
    out = {"configurations": int(r["conf"].nunique()), "abreast": float((r["population"] == "abreast").mean()),
           "delay quantiles (s)": np.nanpercentile(r["delay"], [25, 50, 75, 90]).tolist()}
    budget_rows = by_middle(runs.drop(columns=["population", "episode", "a", "b"], errors="ignore"), clear)
    perf_rows = by_middle(terms[terms["population"] == "group"], clear)
    effects = {}
    for table, stats in ((budget_rows, decompose.BUDGET), (perf_rows, decompose.PERFORMANCE)):
        values = decompose.effect(decompose.ratio_units(table, "conf", info, block, stats), "leader", "follower", counts)
        effects.update({stat: decompose.described(values[:, j]) for j, stat in enumerate(stats)})
    two = {k: v[pieces["population"] == "n2"] for k, v in pieces.items()}
    placed = by_middle(pd.DataFrame({"flight": two["flight"], "t_mid": two["start"] + two["duration"] / 2,
                                     "piece": np.arange(len(two["flight"]))}), clear)
    p = {k: v[placed["piece"].to_numpy()] for k, v in two.items()} | {"population": placed["population"].to_numpy(),
                                                                         "conf": placed["conf"].to_numpy()}
    values = decompose.effect(decompose.msd_units(p, "conf", info, block), "leader", "follower", counts)
    effects.update({stat: decompose.described(v) for stat, v in decompose.msd_statistics(values).items()})
    out["leader - follower, same configuration"] = effects
    print("leader - follower", {k: round(v["value"], 4) for k, v in effects.items()}, flush=True)
    return out, r


def habitual(roles_table, flights, rng):
    """Whether some pilots lead more than chance: dispersion of the lead fractions of the pilots
    with at least MIN_ROLES clear roles, against N_PERM random swaps of the roles of every
    configuration. Pilots are names of the catalog; anonymised ones are left out."""
    catalog = pd.read_csv(FIXES["para"].parent.parent / "catalog/catalog.csv", usecols=["flight_id", "pilot"], dtype=str)
    pilot = flights[["flight", "flight_id"]].astype({"flight_id": str}).merge(catalog, on="flight_id", how="left")
    pilot = pilot.set_index("flight")["pilot"]
    r = roles_table[roles_table["population"] != "abreast"].assign(pilot=lambda d: pilot.reindex(d["flight"]).to_numpy())
    r = r[r["pilot"].notna() & ~r["pilot"].astype(str).str.contains(ANONYMISED, regex=False)]
    both = r.groupby("conf")["pilot"].transform("size") == 2  # both members named
    r = r[both]
    counts = r.groupby("pilot").size()
    keep = counts.index[counts >= MIN_ROLES]
    r = r[r["pilot"].isin(keep)].sort_values(["conf", "population"])
    lead = (r["population"] == "leader").to_numpy(float)
    conf = pd.factorize(r["conf"])[0]
    who = pd.factorize(r["pilot"])[0]
    n = np.bincount(who)

    def dispersion(lead):
        k = np.bincount(who, lead)
        return float(np.mean((k - n / 2) ** 2 / (n / 4)))

    observed = dispersion(lead)
    null = []
    for _ in range(N_PERM):  # swap the two roles of each configuration with probability 1/2
        flip = rng.random(conf.max() + 1) < 0.5
        null.append(dispersion(np.where(flip[conf], 1 - lead, lead)))
    null = np.array(null)
    fractions = np.bincount(who, lead) / n
    return {"pilots": int(len(n)), "roles": int(n.sum()), "configurations": int(conf.max() + 1),
            "dispersion": observed, "null mean": float(null.mean()), "null 95%": float(np.percentile(null, 95)),
            "p": float((null >= observed).mean()), "lead fractions": fractions.tolist(), "roles per pilot": n.tolist()}


def hours(spans, flights):
    """Time per local hour of solo flights, of the group flights alone, and in a group (shares), and
    the fraction of the group flights' time spent in a group."""
    a, b = spans["a"].to_numpy(float), spans["b"].to_numpy(float)
    n = np.maximum(np.ceil((b - a - SAMPLE_S / 2) / SAMPLE_S), 0).astype(np.int64)  # samples a + S/2 + kS < b
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    utc = np.repeat(flights.set_index("flight")["utc0"].reindex(spans["flight"]).to_numpy() + a + SAMPLE_S / 2, n)
    hour = decompose.local_hour(utc + k * SAMPLE_S)
    pop = np.repeat(spans["population"].to_numpy(), n)
    out = {}
    for p in ("solo", "group_solo", "group"):
        out[p] = (np.bincount(hour[pop == p], minlength=24) / (pop == p).sum()).tolist()
    g, a = (np.bincount(hour[pop == p], minlength=24) for p in ("group", "group_solo"))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fraction in a group"] = (g / (g + a)).tolist()
    return out


def tidy(axes):
    for a in np.atleast_1d(axes):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)


def plot_phases(result, path):
    """The phase at the instant of entering and leaving a group, against the time shares."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    series = (("entry", "entering a group", "#2a78d6", "o"), ("exit", "leaving a group", "#eb6834", "o"),
              ("time alone", "share of the time, same flights alone", "#8a8a86", "s"),
              ("time in a group", "share of the time, in a group", "#3d3d3a", "D"))
    cats = PHASES[:3]
    for k, (key, label, color, marker) in enumerate(series):
        x = np.arange(len(cats)) + (k - 1.5) * 0.14
        v = np.array([result["phases"][key][c]["value"] for c in cats]) * 100
        band = np.array([result["phases"][key][c]["band"] for c in cats]) * 100
        ax.errorbar(x, v, yerr=[v - band[:, 0], band[:, 1] - v], fmt=marker, color=color, ms=6, capsize=2, label=label)
    ax.set_xticks(range(len(cats)), list(cats))
    ax.set(ylabel="%", title="Paragliders: phase at the instant of entering or leaving a group")
    ax.title.set_fontsize(9.5)
    ax.legend(frameon=False, fontsize=7.5)
    tidy(ax)
    save(fig, path)


def plot_hours(result, path):
    """Left: the time of each population over the local hour. Right: the share of the group flights'
    time spent in a group, by hour."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    h = np.arange(24)
    for pop, label in (("solo", "solo flights"), ("group_solo", "group flights, alone"), ("group", "group flights, in a group")):
        axes[0].plot(h, np.array(result["hours"][pop]) * 100, color=COLORS[pop], marker="o", ms=3, lw=1.2, label=label)
    axes[0].set(xlim=(7.5, 20.5), xlabel="local hour", ylabel="% of the time of the population",
                title="When the time is flown")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].plot(h, np.array(result["hours"]["fraction in a group"]) * 100, color=COLORS["group"], marker="o", ms=3, lw=1.2)
    axes[1].set(xlim=(7.5, 20.5), ylim=(0, None), xlabel="local hour", ylabel="%",
                title="Share of the group flights' time spent in a group")
    tidy(axes)
    save(fig, path)


def plot_habitual(result, path, rng):
    """The lead fractions of the pilots, against those of roles drawn at random (binomial, same
    number of roles per pilot, mean of 200 draws)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    hab = result["habitual"]
    bins = np.linspace(0, 1, 21)
    ax.hist(hab["lead fractions"], bins=bins, color="#2a78d6", alpha=0.85, label="observed")
    n = np.array(hab["roles per pilot"])
    null = np.mean([np.histogram(rng.binomial(n, 0.5) / n, bins)[0] for _ in range(200)], axis=0)
    ax.step(bins, np.r_[null, null[-1]], where="post", color="#3d3d3a", lw=1.4, label="roles at random")
    ax.set(xlabel="fraction of a pilot's roles as the one ahead", ylabel="pilots",
           title=f"{hab['pilots']} pilots with at least {MIN_ROLES} roles in pairs")
    ax.title.set_fontsize(9.5)
    ax.legend(frameon=False, fontsize=7.5)
    tidy(ax)
    save(fig, path)


if __name__ == "__main__":
    flights = pd.read_parquet(LABELS / "flights.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary)
    block = summary.set_index("flight")["block"]
    counts = decompose.block_counts(int(block.max()) + 1, np.random.default_rng(SEED))
    runs = budget.phase_runs(flights, spans).assign(t_mid=lambda r: (r["t_start"] + r["t_end"]) / 2)
    all_runs = pd.read_parquet(RUNS["para"], columns=["flight_id", "t_start", "t_end", "phase"]).merge(
        flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]], on="flight_id")
    result = {"phases": entries_and_exits(flights, spans, runs, all_runs, block, counts)}
    print("phases", json.dumps(result["phases"]["counts"]), flush=True)
    info, _, _ = decompose.flight_table(decompose.with_takeoffs(flights), summary)
    pieces = size.tamsd_pieces("para")
    terms = pd.read_parquet(CACHE / "performance_terms.parquet")
    result["leaders"], table = leaders(flights, pd.read_parquet(LABELS / "configurations.parquet"),
                                       pd.read_parquet(LABELS / "pair_intervals.parquet"), runs, terms, pieces, info,
                                       block, counts)
    result["habitual"] = habitual(table, flights, np.random.default_rng(SEED))
    result["hours"] = hours(spans, flights)
    (OUT / "mechanisms.json").write_text(json.dumps(result, indent=1))
    plot_phases(result, OUT / "mechanisms_phases.pdf")
    plot_hours(result, OUT / "mechanisms_hours.pdf")
    plot_habitual(result, OUT / "mechanisms_habitual.pdf", np.random.default_rng(SEED))
    name = "leader - follower, same configuration"
    decompose.forest({name: result["leaders"][name]}, OUT / "mechanisms_leaders.pdf", contrasts={name: None},
                     labels={name: "leader - follower"}, colors={name: "#c24a1d"},
                     title="Paragliders: the pilot ahead minus the one behind, same configuration of 2 pilots",
                     per_row=4)
