"""Glide: in glide, how does a group stretch along the route and fan out across it? The scale of one group.

Population: the groups of breathing.py (n >= 2, all members 1 Hz flights with phases, at least
MIN_GROUP s), in sizes 2 and 3+, with the same grid instants, passage times x, route direction u,
L_par, L_perp, L_z and climbing share f. Null: the chance contacts of the surrogate pairs of
cohesion.py (two pilots who never met, at the same place) lasting at least tau*, both flights with
phases, as groups of 2: the first pilot and the copy of the second shifted by the r of the run
(cohesion.py stores the contacts, the run stores r). Their delays, positions and climb runs are
read on the contact instants; the pilot ahead (smaller x) is the head.

Glides. A glide is a maximal run of instants with f = 0 (no member in a climb run) lasting at least
MIN_GLIDE_S, with f > 0 just before and just after it: it lies between the group leaving a thermal
and entering the next. Its instants are taken every SAMPLE_S from its start; s = (t - start) / length
lies in [0, 1), and the minutes from the start and to the end are kept as well.

Along the route. L_par ~ v_head sigma_l (members on the head's track, each one delay behind), so per
glide the changes over DT of log L_par are split into those of log sigma_l (sigma_l: RMS of x, the
delays) and of log v_head (the head's displacement over the last DT along u, over DT; its window
must lie inside the glide), for each DT in DTS. Shares of the variance of the changes about their
glide mean: sigma_l, v_head, twice their covariance, and the rest (what the product leaves out);
and, as the factors can be noisier than their product, the share of each factor in the variance
of the change, Cov(y, factor) / Var(y), which sums to 1 with the rest (attr_*).

Across the route. d, signed distance of a follower at t from the head's track at the point where
the head passed (the head's position at t - (x_i - x_head), as in breathing.follow): positive on
the left of the head's velocity there (chord of the head's track over +- VEL_S around the passage).
d is kept where the passage window lies inside the glide, and per instant as the mean of d^2 over
the followers. Per bin of s, of the minutes from the start and to the end: the RMS of d (pooled
over the instants), the median over the instants of the RMS of d, and the medians of L_perp and
L_z, by size and for the chance pairs.

Bands: 5-95% of N_BOOT draws of the dependency blocks of the note (episodes) for the groups; the
chance pairs belong to no episode, so their blocks are their dates.

    uv run python glide.py [--run NAME] [--sample N]    # from this folder, after cohesion.py
"""

import argparse
import json
import pickle
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from breathing import MIN_GROUP, N_BOOT, NO_RUNS, ROUTE_S, band, boot_median, follow, geometry, observe, route, stat
from cohesion import CACHE as COHESION, contacts, dense, grid
from rotation import climb_runs, climbing, positions
from tamsd import LABELS, OUT, SEED, save

MIN_GLIDE_S = 120  # shortest glide kept, s
SAMPLE_S = 30  # spacing of the instants of a glide, s
DTS = (30, 60, 120)  # times over which the changes of log L_par and the head's speed are taken, s
VEL_S = 10  # half-window of the head's velocity at a passage point, s
N_S = 10  # bins of the normalised time s
N_MIN = 10  # bins of one minute from the start and to the end
KINDS = ("pairs", "3+", "chance")
LABEL = {"pairs": "pairs", "3+": "3+ pilots", "chance": "chance pairs"}
COLORS = {"pairs": "#8e5ad7", "3+": "#eb6834", "chance": "#2a78d6"}
AXES = {"s": (0, N_S), "from_start": (1, N_MIN), "to_end": (2, N_MIN)}  # column of an instant, bins
PARTS = ("sigma_l", "v_head", "covariance", "rest", "correlation", "attr_sigma_l", "attr_v_head", "attr_rest")


def glides(f, step):
    """The glides of a series of climbing shares: the first instant and the end instant (excluded)
    of the maximal runs of f = 0 lasting at least MIN_GLIDE_S with f > 0 before and after."""
    state = (f == 0).astype(int)
    edge = np.flatnonzero(np.diff(state)) + 1
    first, end = np.r_[0, edge], np.r_[edge, len(f)]
    keep = (state[first] == 1) & (first > 0) & (end < len(f)) & ((end - first) * step >= MIN_GLIDE_S)
    return list(zip(first[keep].tolist(), end[keep].tolist()))


def cross_track(xyz, x, step):
    """The signed distance d (T, n) in m of each member at t from the head's track at the point
    where the head passed (follow), positive on the left of the head's velocity there; 0 for the
    head, NaN without an order, a fix or a head track around the passage. `xyz` (n, T, 3), `x` (T, n)."""
    n, steps = xyz.shape[:2]
    past = follow(xyz, x, step)
    velocity = follow(xyz, x, step, VEL_S) - follow(xyz, x, step, -VEL_S)
    left = np.cross(past / np.linalg.norm(past, axis=-1, keepdims=True), velocity)
    d = ((xyz - past) * left).sum(-1) / np.linalg.norm(left, axis=-1)
    d[x.argmin(axis=1), np.arange(steps)] = 0.0
    return d.T


def variance_parts(log_l, log_s, log_v, lag=1):
    """The sums over the steps of one glide, from the series sampled every SAMPLE_S, of the squares
    and products about the glide mean of the changes over `lag` samples of log L_par (y), log
    sigma_l (a) and log v_head (b), over the steps where the three exist: [steps, aa, bb, ab, yy, ay,
    by]; zeros with fewer than 2."""
    a, b, y = (q[lag:] - q[:-lag] for q in (log_s, log_v, log_l))
    ok = np.isfinite(a + b + y)
    a, b, y = a[ok], b[ok], y[ok]
    if len(a) < 2:
        return np.zeros(7)
    a, b, y = a - a.mean(), b - b.mean(), y - y.mean()
    return np.array([len(a), a @ a, b @ b, a @ b, y @ y, a @ y, b @ y])


def one_series(xyz, z, x, f, step):
    """The glides of one group or chance pair (`xyz` (n, T, 3), `z` (n, T), `x` (T, n), `f` (T,)),
    each as (instants, moments, length in s): the instants every SAMPLE_S as the columns s, minutes
    from the start, minutes to the end, L_perp, L_z, mean d^2 over the followers (m, 6), and the
    moments of variance_parts for each DT of DTS (len(DTS), 7)."""
    found = glides(f, step)
    if not found:
        return []
    n, steps = z.shape
    k, h = SAMPLE_S // step, ROUTE_S // (2 * step)
    np.seterr(divide="ignore", invalid="ignore")
    L = geometry(xyz, z, h)
    flat, u = route(xyz, h)
    d2 = cross_track(xyz, x, step) ** 2
    delay = (x - x.min(axis=1, keepdims=True)) / step
    sigma = np.sqrt((x ** 2).mean(axis=1))
    head, t = x.argmin(axis=1), np.arange(steps)
    speeds = []
    for dt in DTS:
        m = dt // step
        v = np.full(steps, np.nan)
        v[m:] = (flat(xyz[head[m:], t[m:]] - xyz[head[m:], t[:-m]], m, steps) * u[m:]).sum(-1) / dt
        speeds.append(v)
    out = []
    for a, e in found:
        i = np.arange(a, e, k)
        w = VEL_S / step
        inside = ~(delay[i] > 0) | ((i[:, None] - delay[i] - w >= a) & (i[:, None] - delay[i] + w <= e - 1))
        mean_d2 = np.where(inside.all(axis=1), d2[i].sum(axis=1) / (n - 1), np.nan)
        moments = np.array([variance_parts(np.log(L[0, i]), np.log(sigma[i]), np.log(np.where((i - dt // step >= a) & (v[i] > 0), v[i], np.nan)),
                                           dt // SAMPLE_S) for dt, v in zip(DTS, speeds)])
        instants = np.column_stack([(i - a) / (e - a), (i - a) * step / 60, (e - i) * step / 60, L[1, i], L[2, i], mean_d2])
        out.append((instants, moments, (e - a) * step))
    return out


def chance_series(on, off, series, r, flight_i, flight_j, row_i, row_j, step, tracks, runs):
    """The chance pair of a surrogate contact [on, off) as a group of 2: the copy of j shifted by r
    and i (cohesion.series stores the delay by which i follows the copy). Returns the positions and
    altitudes (2, T, 4), x (T, 2) and f (T,)."""
    t = on + step * np.arange((off - on) // step)
    p = np.stack([tracks.at(("shift", row_j), t)[:, :4], tracks.at(row_i, t)[:, :4]])
    x = positions(2, [(1, 0)], dense(*series, on, off, step)[:, None])
    both = [climbing(SimpleNamespace(t_start=on + r, t_end=off + r, members=(flight_j,)), step, runs),
            climbing(SimpleNamespace(t_start=on, t_end=off, members=(flight_i,)), step, runs)]
    return p, x, np.concatenate(both, axis=1).mean(axis=1)


def measure(groups, flights, dates, step, p, tau_star, per_series=one_series):
    """What `per_series(xyz, z, x, f, step)` returns for each group of `groups` and each chance contact
    of the `dates`: a list of (kind, block, result), a block being an episode block for a group and a
    date for a chance pair."""
    runs = climb_runs(flights, set(flights.loc[flights["labelled"] & flights["date"].isin(dates), "flight"]))
    archive = grid.archive()
    by_date = dict(tuple(archive[archive["date"].isin(dates)].groupby("date")))
    flights_of = dict(tuple(flights[flights["date"].isin(dates)].groupby("date")))
    by_day = dict(tuple(groups.groupby("date")))
    out = []
    for number, date in enumerate(dates):
        mine = flights_of[date]
        labelled = dict(zip(mine["flight_id"], mine["labelled"]))
        cached = pickle.loads((COHESION / "all" / "days" / f"{date}.pkl").read_bytes())
        chance = [c for c in cached["chance"] if c[3] - c[2] >= tau_star and labelled.get(c[0], False) and labelled.get(c[1], False)]
        day = by_day.get(date)
        if day is None and not chance:
            continue
        flight_of = dict(zip(mine["flight_id"], mine["flight"]))
        wanted = {f for m in (day["members"] if day is not None else []) for f in m}
        wanted_ids = {c[k] for c in chance for k in (0, 1)}
        kept = mine[mine["flight"].isin(wanted) | mine["flight_id"].isin(wanted_ids)]
        a = by_date[date].merge(kept[["discipline", "flight_id"]], on=["discipline", "flight_id"])
        table, g = grid.day(a, {d: NO_RUNS for d in grid.DERIVED}, p["step"], p["h_max"], p["r_dep"])
        tracks = contacts.Tracks(g, p["step"])
        row_of = {flight_of[fid]: k for k, fid in enumerate(table["flight_id"])}
        for group in (day.itertuples() if day is not None else []):
            xyz, x = observe(group, cached["real"], flight_of, step, tracks, row_of)
            f = climbing(group, step, runs).mean(axis=1)
            out.append(("pairs" if len(group.members) == 2 else "3+", group.block, per_series(xyz[..., :3], xyz[..., 3], x, f, step)))
        if chance:
            run = pickle.loads((LABELS / "days" / f"{date}.pkl").read_bytes())
            ids = run["table"]["flight_id"].to_numpy()
            shift = {(ids[i], ids[j]): r for i, j, r in zip(run["null_pairs"]["i"], run["null_pairs"]["j"], run["null_pairs"]["r"])}
            row = {fid: k for k, fid in enumerate(table["flight_id"])}
            for c in chance:
                r = shift[c[0], c[1]]
                tracks.shifted(row[c[1]], r, ("shift", row[c[1]]))
                xyz, x, f = chance_series(c[2], c[3], c[4:], r, flight_of[c[0]], flight_of[c[1]], row[c[0]], row[c[1]], step, tracks, runs)
                out.append(("chance", date, per_series(xyz[..., :3], xyz[..., 3], x, f, step)))
        if number % 100 == 0:
            print(f"{number}/{len(dates)} dates, {len(out)} units", flush=True)
    return out


def shares(m):
    """The PARTS of the variance of the changes of log L_par in the moments (..., 7) of
    variance_parts: the shares of sigma_l, v_head, twice their covariance and the rest, the
    correlation of the two changes, and the shares Cov(y, factor) / Var(y) of sigma_l, v_head and
    the rest."""
    n, aa, bb, ab, yy, ay, by = np.moveaxis(m, -1, 0)
    return np.stack([aa / yy, bb / yy, 2 * ab / yy, 1 - (aa + bb + 2 * ab) / yy, ab / np.sqrt(aa * bb),
                     ay / yy, by / yy, 1 - (ay + by) / yy], axis=-1)


def profile(inst, owner, blocks, w):
    """Per axis of the glide time and per bin: the RMS of d, the median of the per-instant RMS of d
    and the medians of L_perp and L_z of the instants `inst` (m, 6), with their draws (N_BOOT,
    bins), and the number of instants with d."""
    out = {}
    for axis, (col, bins) in AXES.items():
        at = np.floor(inst[:, col] * (N_S if axis == "s" else 1)).astype(int)
        res = {"d_rms": [], "d_med": [], "L_perp": [], "L_z": [], "n": []}
        for b in range(bins):
            here = at == b
            ok = here & np.isfinite(inst[:, 5])
            s, c = (np.bincount(owner[ok], v, blocks) for v in (inst[ok, 5], np.ones(ok.sum())))
            res["d_rms"].append((np.sqrt(s.sum() / c.sum()), np.sqrt((w @ s) / (w @ c))))
            res["n"].append(int(ok.sum()))
            res["d_med"].append(boot_median(np.sqrt(inst[ok, 5]), owner[ok], w) if ok.any() else (np.nan, np.full(N_BOOT, np.nan)))
            for name, j in (("L_perp", 3), ("L_z", 4)):
                ok = here & np.isfinite(inst[:, j])
                res[name].append(boot_median(inst[ok, j], owner[ok], w) if ok.any() else (np.nan, np.full(N_BOOT, np.nan)))
        out[axis] = {name: ([r[0] for r in v], np.array([r[1] for r in v]).T) if name != "n" else v for name, v in res.items()}
    return out


def summarize(units, rng):
    """The glides, the variance shares and the profiles by kind, with their bands. The draws of the
    groups are shared by the two sizes."""
    np.seterr(divide="ignore", invalid="ignore")
    out, prof = {"glides": {}, "along": {}, "profile": {}, "chance_ratio": {}}, {}
    for population in (("pairs", "3+"), ("chance",)):
        sel = [u for u in units if u[0] in population]
        ids, key = np.unique([str(u[1]) for u in sel], return_inverse=True)
        w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)
        for kind in population:
            mine = [(k, u[2]) for k, u in zip(key, sel) if u[0] == kind]
            pieces = [(k, g) for k, gl in mine for g in gl]
            owner = np.array([k for k, _ in pieces], int)
            by_block = np.zeros((len(ids), len(DTS), 7))
            np.add.at(by_block, owner, np.array([g[1] for _, g in pieces]).reshape(-1, len(DTS), 7))
            sh, dr = shares(by_block.sum(0)), shares(np.tensordot(w, by_block, axes=(1, 0)))
            out["along"][kind] = {str(dt): {"steps": float(by_block[:, i, 0].sum())} | {
                name: stat(float(sh[i, j]), dr[:, i, j]) for j, name in enumerate(PARTS)}
                for i, dt in enumerate(DTS)}
            seconds = np.array([g[2] for _, g in pieces])
            out["glides"][kind] = {"units": len(mine), "glides": len(pieces), "hours": float(seconds.sum() / 3600),
                                   "median_min": float(np.median(seconds) / 60)}
            inst = np.concatenate([g[0] for _, g in pieces])
            owner_i = np.repeat(owner, [len(g[0]) for _, g in pieces])
            prof[kind] = profile(inst, owner_i, len(ids), w)
            out["profile"][kind] = {axis: {"centers": ((np.arange(AXES[axis][1]) + 0.5) / (N_S if axis == "s" else 1)).tolist(),
                                           "n_d": r["n"]} | {name: [stat(v, d) for v, d in zip(r[name][0], r[name][1].T)]
                                                              for name in ("d_rms", "d_med", "L_perp", "L_z")}
                                    for axis, r in prof[kind].items()}
    for axis in AXES:  # pairs of the groups against the chance pairs
        value = np.array(prof["pairs"][axis]["d_rms"][0]) / np.array(prof["chance"][axis]["d_rms"][0])
        draws = prof["pairs"][axis]["d_rms"][1] / prof["chance"][axis]["d_rms"][1]
        out["chance_ratio"][axis] = [stat(v, d) for v, d in zip(value, draws.T)]
    return out


def plot(r, path):
    """Left: shares of the variance of the changes of log L_par over DT in sigma_l, in v_head and
    in the rest (Cov(y, factor) / Var(y)), against DT, for the groups; middle: RMS cross-track
    offset against the normalised time in the glide, groups and chance pairs; right: median L_z
    against it."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    styles = {"attr_sigma_l": ("-", "o", r"$\sigma_\ell$"), "attr_v_head": ("--", "s", r"$v_{head}$"), "attr_rest": (":", "v", "rest")}
    for i, kind in enumerate(KINDS):
        if kind != "chance":
            for part, (ls, marker, _) in styles.items():
                rows = [r["along"][kind][str(dt)][part] for dt in DTS]
                v, bd = np.array([x["value"] for x in rows]), np.array([x["band"] for x in rows]).T
                a.errorbar(np.arange(len(DTS)) + (i - 0.5) * 0.08, v, yerr=[v - bd[0], bd[1] - v], color=COLORS[kind], ls=ls,
                           marker=marker, ms=4, lw=1.1, capsize=2)
            a.plot([], [], color=COLORS[kind], label=LABEL[kind])
        for ax, name in ((b, "d_rms"), (c, "L_z")):
            p = r["profile"][kind]["s"]
            ax.fill_between(p["centers"], *np.array([x["band"] for x in p[name]]).T, color=COLORS[kind], alpha=0.15, lw=0)
            ax.plot(p["centers"], [x["value"] for x in p[name]], color=COLORS[kind], ls="--" if kind == "chance" else "-",
                    marker="o", ms=3, lw=1.2, label=LABEL[kind])
    for part, (ls, marker, label) in styles.items():
        a.plot([], [], color="#333333", ls=ls, marker=marker, ms=4, label=label)
    a.axhline(0, color="#999999", lw=0.6)
    a.set_xticks(range(len(DTS)), [str(dt) for dt in DTS])
    a.set(xlabel="time $\\Delta$ of the changes (s)", ylabel=r"share of the variance of $\Delta\log L_\parallel$")
    a.legend(frameon=False, fontsize=7, ncol=2)
    b.set(xlabel="time in the glide, $s$", ylabel="RMS cross-track offset $d$ (m)", ylim=(0, None))
    b.legend(frameon=False, fontsize=8)
    c.set(xlabel="time in the glide, $s$", ylabel=r"median $L_z$ (m)", ylim=(0, None))
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0)")
    args = parser.parse_args()
    params = json.loads((LABELS / "days" / "params.json").read_text())
    step = params["step"]
    tau_star = json.loads((LABELS / "calibration.json").read_text())["tau_star"]
    flights = pd.read_parquet(LABELS / "flights.parquet")
    groups = pd.read_parquet(LABELS / "groups.parquet")
    labelled = flights.set_index("flight")["labelled"]
    groups = groups[groups["t_end"] - groups["t_start"] >= MIN_GROUP]
    groups = groups[groups["members"].map(lambda m: bool(labelled.loc[list(m)].all()))].reset_index(drop=True)
    if args.sample:
        dates = np.random.default_rng(0).choice(sorted(groups["date"].unique()), args.sample, replace=False)
        groups = groups[groups["date"].isin(dates)].reset_index(drop=True)
    else:
        dates = sorted(f.stem for f in (COHESION / "all" / "days").glob("*.pkl") if not f.name.startswith("."))
    block = pd.read_parquet(LABELS / "episodes.parquet").set_index("episode_id")["block"]
    groups["block"] = block.reindex(groups["episode_id"]).to_numpy()
    units = measure(groups, flights, sorted(dates), step, params, tau_star)
    result = summarize(units, np.random.default_rng(SEED))
    result["params"] = {"min_glide_s": MIN_GLIDE_S, "sample_s": SAMPLE_S, "dts": list(DTS), "vel_s": VEL_S, "route_s": ROUTE_S, "tau_star": tau_star}
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"glide_sample{args.sample}" if args.sample else "glide"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps({k: result[k] for k in ("glides", "along")}), flush=True)
