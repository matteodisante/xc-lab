"""Breathing: does a group contract and stretch, and is that more than kinematics? The scale of one group.

Population: the groups of n >= 2 whose members are all 1 Hz flights with phases and that last at
least MIN_GROUP s (the longest lag of the autocorrelations), in sizes 2, 3, 4, 5+, on the grid
instants of the group.

Per instant (every instant with all the members on the grid of grid.py):
- c, the centroid of the members (Earth-centred), and the horizontal frame (east, north) at c.
- u, the route direction: the horizontal displacement of c between t - ROUTE_S / 2 and
  t + ROUTE_S / 2, both inside the group. Circling makes the instantaneous direction meaningless,
  so the route is the window.
- L_par, L_perp: RMS over the members of the horizontal offsets from c along u and across it;
  L_z: RMS of the altitude offsets from their mean.
- sigma_l: RMS of the passage times x_i of rotation.positions (s) where the delays connect all
  the members.
- Follow the leader, the length of constant delays: member i is predicted where the head (the
  smallest x) was at t - (x_i - x_head), its track interpolated linearly on the grid; NaN before
  the group starts or without an order. L_par_pred, L_perp_pred: the same RMS offsets of the
  predicted positions from their centroid, along the same u. ratio = L_par / L_par_pred.
- f, the climbing share of the members (rotation.climbing).

Measures, by size, each quantity on its own valid instants:
- By class of f (0, (0, 0.5), [0.5, 1), 1): the medians of L_par, L_perp, L_z, sigma_l, L_par_pred,
  L_perp_pred, ratio, and their ratio to the class f = 0. Under "prediction", on the instants
  where both L_par and L_par_pred exist: their medians, the median of the ratio and the correlation
  of the logs of the two, over all and within each class. If the observed length tracks the
  prediction the breathing is the kinematics of constant delays; the residual is a real change.
- Entries: the instant at which f first reaches 0.5 after f = 0 lasting at least MIN_RUN s and
  staying below 0.5 since. The medians of the same quantities from -PRE_S to POST_S around it,
  every SUB_S, their ratio to the mean of the medians before the entry (not for the ratio), and
  the correlation of the logs of L_par and L_par_pred across the entries at each offset.
- Autocorrelation of L_par, sigma_l and f up to ACF_MIN min, every SUB_S: per group the series
  minus its mean, over the pairs of instants where both exist, and the first lag at or below 0.
  The mean removal makes the long lags slightly negative.

Medians and sums use the instants every SUB_S. Bands: 5-95% of N_BOOT draws of the dependency
blocks of the note (episodes).

    uv run python breathing.py [--run NAME] [--sample N]    # from this folder, after cohesion.py
"""

import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cohesion import CACHE as COHESION, contacts, grid
from hierarchy import MIN_RUN
from rotation import climb_runs, climbing, group_delays, positions
from tamsd import LABELS, OUT, SEED, save

ROUTE_S = 600  # window of the route direction, s
MIN_GROUP = 30 * 60  # shortest group kept, s
SUB_S = 30  # spacing of the instants that enter the medians and the autocorrelations, s
ACF_MIN = 30  # longest lag of the autocorrelations, min
PRE_S, POST_S = 300, 600  # window around an entry, s
N_BOOT = 200  # draws of the blocks
SIZES = ("2", "3", "4", "5+", "all")
CLASSES = ("f = 0", "0 < f < 0.5", "0.5 <= f < 1", "f = 1")
QUANTITIES = ("L_par", "L_perp", "L_z", "sigma_l", "L_par_pred", "L_perp_pred", "ratio", "f")
Q = {name: i for i, name in enumerate(QUANTITIES)}
SERIES = ("L_par", "sigma_l", "f")  # the autocorrelated quantities
COLORS = {"2": "#8e5ad7", "3": "#eb6834", "4": "#2a78d6", "5+": "#1baf7a", "all": "#333333"}
NO_RUNS = pd.DataFrame(columns=["flight_id", "segment_id", "t_start", "t_end", "state"])


def geometry(xyz, z, h, predicted=None):
    """The horizontal spread of the members along and across the route of one group. `xyz` (n, T, 3)
    are the Earth-centred positions of the members, `z` (n, T) their altitudes, NaN without a fix;
    `h` is half of ROUTE_S in instants; `predicted` (n, T, 3) positions to measure on the same
    centroid procedure and the same route. Returns L_par, L_perp, L_z and, with `predicted`,
    L_par_pred, L_perp_pred: (3 or 5, T), NaN where a member has no fix or where an end of the
    route window falls outside the group."""
    n, steps = z.shape
    c = xyz.mean(axis=0)
    lon, lat = np.arctan2(c[:, 1], c[:, 0]), np.arctan2(c[:, 2], np.hypot(c[:, 0], c[:, 1]))
    east = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros(steps)])
    north = np.column_stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    flat = lambda d, a, b: np.stack([(d * east[a:b]).sum(-1), (d * north[a:b]).sum(-1)], -1)
    rows = 3 if predicted is None else 5
    if steps <= 2 * h:
        return np.full((rows, steps), np.nan)
    shift = flat(c[2 * h:] - c[:-2 * h], h, steps - h)  # displacement over the window, in the frame at its middle
    u = np.full((steps, 2), np.nan)
    u[h:steps - h] = shift / np.linalg.norm(shift, axis=1)[:, None]

    def spread(p):
        offset = flat(p - p.mean(axis=0), 0, steps)  # (n, T, 2)
        along = (offset * u).sum(-1)
        across = offset[..., 0] * -u[:, 1] + offset[..., 1] * u[:, 0]
        return np.sqrt((along ** 2).mean(0)), np.sqrt((across ** 2).mean(0))

    out = [*spread(xyz), np.sqrt(((z - z.mean(0)) ** 2).mean(0))]
    return np.array(out + ([] if predicted is None else list(spread(predicted))))


def follow(xyz, x, step):
    """The positions (n, T, 3) of the members if each followed the head of the instant along its
    track with its delay: member i at t is where the head, the member of the smallest x, was at
    t - (x_i - x_head), `x` (T, n) in s, `xyz` (n, T, 3). Linear between grid instants; NaN where
    that time precedes the group, where the track has no fix, and where x is NaN."""
    n, steps = xyz.shape[:2]
    head = x.argmin(axis=1)
    j = np.arange(steps)[:, None] - (x - x.min(axis=1, keepdims=True)) / step
    ok = j > -1e-6  # rounding of a delay that reaches the first instant
    j = np.where(ok, np.maximum(j, 0), 0)
    lo = np.floor(j).astype(int)
    w = (j - lo)[..., None]
    a, b = xyz[head[:, None], lo], xyz[head[:, None], np.minimum(lo + 1, steps - 1)]
    out = np.where(w == 0, a, (1 - w) * a + w * b)
    out[~ok] = np.nan
    return out.transpose(1, 0, 2)


def entries(f, step):
    """The instants at which f first reaches 0.5 after f = 0 lasting at least MIN_RUN s and staying
    below 0.5 since."""
    state = np.where(f == 0, 0, np.where(f < 0.5, 1, 2))
    first = np.r_[0, np.flatnonzero(np.diff(state)) + 1]
    length = np.diff(np.r_[first, len(f)])
    out, armed = [], False
    for s, a, m in zip(state[first], first, length):
        if s == 0 and m * step >= MIN_RUN:
            armed = True
        elif s == 2 and armed:
            out.append(a)
            armed = False
    return np.array(out, int)


def autocovariance(y, lags):
    """Per lag 0..lags (instants), the sums over the pairs of instants where both y exist of the
    product of the series minus its mean, and of the mean of their squares: (lags + 1, 2)."""
    ok = np.isfinite(y)
    z = np.where(ok, y - y[ok].mean(), 0.0) if ok.any() else np.zeros(len(y))
    m = len(y)
    pad = lambda a: np.r_[a[m - 1:m + lags], np.zeros(max(0, lags + 1 - (len(a) - m + 1)))]
    num = pad(np.correlate(z, z, "full"))
    den = (pad(np.correlate(ok.astype(float), z ** 2, "full")) + pad(np.correlate(z ** 2, ok.astype(float), "full"))) / 2
    return np.column_stack([num, den])


def one_group(group, real, flight_of, step, runs, tracks, row_of):
    """The series of one group at its grid instants: the QUANTITIES as an array (8, T)."""
    n = len(group.members)
    t = group.t_start + step * np.arange((group.t_end - group.t_start) // step)
    p = np.stack([tracks.at(row_of[f], t)[:, :4] if row_of.get(f) in tracks.values else np.full((len(t), 4), np.nan)
                  for f in group.members])
    edges, delays = group_delays(group, real, flight_of, step)
    x = positions(n, edges, delays) if len(edges) else np.full((len(t), n), np.nan)
    L = geometry(p[..., :3], p[..., 3], ROUTE_S // (2 * step), follow(p[..., :3], x, step))
    ratio = np.divide(L[0], L[3], out=np.full(len(t), np.nan), where=L[3] > 0)
    f = climbing(group, step, runs).mean(axis=1)
    return np.array([L[0], L[1], L[2], np.sqrt((x ** 2).mean(axis=1)), L[3], L[4], ratio, f])


def summarize_group(q, step):
    """The pieces of one group that the summary pools: its series every SUB_S (8, m), the profile
    around its entries (entries, offsets, 8), and the autocovariances of SERIES (3, lags + 1, 2)."""
    k = SUB_S // step
    series = q[:, ::k]
    offsets = np.arange(-PRE_S, POST_S + 1, SUB_S) // step
    at = entries(q[Q["f"]], step)[:, None] + offsets[None, :]
    inside = (at >= 0) & (at < q.shape[1])
    profile = np.where(inside[..., None], q.T[np.clip(at, 0, q.shape[1] - 1)], np.nan)
    acf = np.array([autocovariance(series[Q[s]], ACF_MIN * 60 // SUB_S) for s in SERIES])
    return series, profile, acf


def measure(groups, flights, step, p):
    """Per group of `groups`: its series, entry profiles and autocovariances (summarize_group)."""
    runs = climb_runs(flights, {f for m in groups["members"] for f in m})
    archive = grid.archive()
    by_date = dict(tuple(archive[archive["date"].isin(groups["date"])].groupby("date")))
    flights_of = dict(tuple(flights[flights["date"].isin(groups["date"])].groupby("date")))
    out = []
    for number, (date, day) in enumerate(groups.groupby("date", sort=False)):
        real = pickle.loads((COHESION / "all" / "days" / f"{date}.pkl").read_bytes())["real"]
        mine = flights_of[date]
        flight_of = dict(zip(mine["flight_id"], mine["flight"]))
        kept = mine[mine["flight"].isin({f for m in day["members"] for f in m})]
        a = by_date[date].merge(kept[["discipline", "flight_id"]], on=["discipline", "flight_id"])
        table, g = grid.day(a, {d: NO_RUNS for d in grid.DERIVED}, p["step"], p["h_max"], p["r_dep"])
        tracks = contacts.Tracks(g, p["step"])
        row_of = {flight_of[fid]: k for k, fid in enumerate(table["flight_id"])}
        for group in day.itertuples():
            out.append(summarize_group(one_group(group, real, flight_of, step, runs, tracks, row_of), step))
        if number % 100 == 0:
            print(f"{number} dates, {len(out)}/{len(groups)} groups", flush=True)
    return out


def boot_median(x, k, w):
    """The median of x and of its draws: `k` is the block of each value, `w` the multiplicity of the
    blocks in each draw (draws, blocks)."""
    o = np.argsort(x)
    x, k = x[o], k[o]
    draws = np.empty(len(w))
    for d, wd in enumerate(w):
        c = np.cumsum(wd[k])
        draws[d] = x[min(np.searchsorted(c, c[-1] / 2), len(x) - 1)] if c[-1] > 0 else np.nan
    return float(np.median(x)), draws


def corr_log(x, y, k, w):
    """The correlation of x and y (positive values) in logs, and its draws; sums by block."""
    a, b = np.log(x), np.log(y)
    s = np.array([np.bincount(k, v, w.shape[1]) for v in (np.ones_like(a), a, b, a * a, b * b, a * b)])
    def r(t):
        n, sa, sb, saa, sbb, sab = t
        return (n * sab - sa * sb) / np.sqrt((n * saa - sa ** 2) * (n * sbb - sb ** 2))
    return float(r(s.sum(axis=1))), np.array([r(s @ wd) for wd in w])


def band(draws):
    return np.nanpercentile(draws, [5, 95]).tolist() if np.isfinite(draws).any() else [np.nan, np.nan]


def stat(value, draws):
    return {"value": value, "band": band(draws)}


def first_zero(r, lags):
    """The first lag at which the autocorrelation r is at most 0, or NaN."""
    i = np.flatnonzero(r <= 0)
    return float(lags[i[0]]) if len(i) else np.nan


def summarize(pieces, groups, block, step, rng):
    """The medians, the entry profiles and the autocorrelations by size, with their bands. `pieces`
    are in the order of `groups`."""
    n = groups["n"].to_numpy()
    cls = {"2": n == 2, "3": n == 3, "4": n == 4, "5+": n >= 5, "all": n >= 2}
    np.seterr(divide="ignore", invalid="ignore")
    ids, kg = np.unique(block, return_inverse=True)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)
    lags = np.arange(ACF_MIN * 60 // SUB_S + 1) * SUB_S / 60
    offsets = np.arange(-PRE_S, POST_S + 1, SUB_S) / 60
    out = {"groups": {}, "by_f": {}, "prediction": {}, "entries": {}, "acf": {}, "classes": list(CLASSES)}
    for size, sel in cls.items():
        idx = np.flatnonzero(sel)
        q = np.concatenate([pieces[g][0] for g in idx], axis=1)
        owner = np.concatenate([np.full(pieces[g][0].shape[1], kg[g]) for g in idx])
        f = q[Q["f"]]
        klass = np.select([f == 0, f < 0.5, f < 1], [0, 1, 2], 3)
        out["groups"][size] = {"groups": int(len(idx)), "hours": float(len(owner) * SUB_S / 3600),
                               "geometry_share": float(np.isfinite(q[Q["L_par"]]).mean()),
                               "sigma_share": float(np.isfinite(q[Q["sigma_l"]]).mean()),
                               "prediction_share": float(np.isfinite(q[Q["L_par_pred"]]).mean())}
        # medians by class of f
        med, draws = {}, {}
        for name in QUANTITIES[:-1]:
            for c in range(4):
                x = q[Q[name], klass == c]
                ok = np.isfinite(x)
                med[name, c], draws[name, c] = boot_median(x[ok], owner[klass == c][ok], w) if ok.any() else (np.nan, np.full(N_BOOT, np.nan))
        rows = {"n": [int((klass == c).sum()) for c in range(4)]}
        for name in QUANTITIES[:-1]:
            rows[name] = [stat(med[name, c], draws[name, c]) for c in range(4)]
            if name != "ratio":
                rows[name + "_over_gliding"] = [stat(med[name, c] / med[name, 0], draws[name, c] / draws[name, 0]) for c in range(4)]
        for name in ("sigma_l", "L_par_pred"):
            rows[name + "_share"] = [float(np.isfinite(q[Q[name], klass == c]).sum() / max(rows["n"][c], 1)) for c in range(4)]
        out["by_f"][size] = rows
        # observed against predicted length, on the instants where both exist
        both = np.isfinite(q[Q["L_par"]]) & np.isfinite(q[Q["L_par_pred"]]) & (q[Q["L_par"]] > 0) & (q[Q["L_par_pred"]] > 0)
        pred = {"instants": int(both.sum())}
        for label, mask in [("all", both)] + [(CLASSES[c], both & (klass == c)) for c in range(4)]:
            if mask.sum() < 30:
                continue
            r, dr = corr_log(q[Q["L_par"], mask], q[Q["L_par_pred"], mask], owner[mask], w)
            pred[label] = {"instants": int(mask.sum()), "corr_log": stat(r, dr)}
            for name in ("L_par", "L_par_pred", "ratio"):
                pred[label][name] = stat(*boot_median(q[Q[name], mask], owner[mask], w))
        out["prediction"][size] = pred
        # entries
        prof = np.concatenate([pieces[g][1] for g in idx])
        eo = np.concatenate([np.full(len(pieces[g][1]), kg[g]) for g in idx])
        ent = {"entries": int(len(prof)), "offsets_min": offsets.tolist()}
        if len(prof):
            pre = offsets < 0
            for name, j in Q.items():
                cols = [boot_median(prof[ok, i, j], eo[ok], w) if ok.any() else (np.nan, np.full(N_BOOT, np.nan))
                        for i, ok in ((i, np.isfinite(prof[:, i, j])) for i in range(len(offsets)))]
                med, dr = np.array([c[0] for c in cols]), np.array([c[1] for c in cols])
                ent[name] = [stat(med[i], dr[i]) for i in range(len(offsets))]
                if name not in ("f", "ratio"):
                    ent[name + "_over_before"] = [stat(med[i] / med[pre].mean(), dr[i] / dr[pre].mean(axis=0))
                                                  for i in range(len(offsets))]
            both = np.isfinite(prof[..., Q["L_par"]]) & np.isfinite(prof[..., Q["L_par_pred"]]) & \
                (prof[..., Q["L_par"]] > 0) & (prof[..., Q["L_par_pred"]] > 0)
            ent["corr_log"] = [stat(*corr_log(prof[both[:, i], i, Q["L_par"]], prof[both[:, i], i, Q["L_par_pred"]], eo[both[:, i]], w))
                               if both[:, i].sum() > 2 else stat(np.nan, np.full(N_BOOT, np.nan)) for i in range(len(offsets))]
        out["entries"][size] = ent
        # autocorrelations
        acf = np.concatenate([pieces[g][2][None] for g in idx])  # (groups, 3, lags + 1, 2)
        by_block = np.zeros((len(ids),) + acf.shape[1:])
        np.add.at(by_block, kg[idx], acf)
        total = np.tensordot(w, by_block, axes=(1, 0))
        rho, rho_draws = by_block.sum(axis=0)[..., 0] / by_block.sum(axis=0)[..., 1], total[..., 0] / total[..., 1]
        out["acf"][size] = {"lags_min": lags.tolist()}
        for j, name in enumerate(SERIES):
            out["acf"][size][name] = [stat(rho[j, i], rho_draws[:, j, i]) for i in range(len(lags))]
            out["acf"][size][name + "_first_zero_min"] = stat(first_zero(rho[j], lags),
                                                              np.array([first_zero(r, lags) for r in rho_draws[:, j]]))
    return out


def plot(r, path):
    """Left: median L_par (solid) and its prediction by constant delays (dashed) in each class of f,
    each relative to its own value at f = 0; middle: the same around the entry of a group in a
    thermal, relative to before it; right: autocorrelation of L_par (solid), sigma_l (dashed) and f (dotted,
    all sizes) against the lag."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    style = {"L_par": "-", "L_par_pred": "--"}
    names = {"L_par": r"$L_\parallel$ observed", "L_par_pred": r"$L_\parallel$ constant delays"}
    for size in SIZES[:4]:
        for name in style:
            rows = r["by_f"][size][name + "_over_gliding"]
            v, bd = np.array([x["value"] for x in rows]), np.array([x["band"] for x in rows]).T
            a.errorbar(np.arange(4) + (SIZES.index(size) - 1.5) * 0.08, v, yerr=[v - bd[0], bd[1] - v], color=COLORS[size],
                       ls=style[name], marker="o", ms=3, lw=1.1, capsize=2)
            e = r["entries"][size]
            if e["entries"]:
                t = e["offsets_min"]
                rows = e[name + "_over_before"]
                b.fill_between(t, *np.array([x["band"] for x in rows]).T, color=COLORS[size], alpha=0.12, lw=0)
                b.plot(t, [x["value"] for x in rows], color=COLORS[size], ls=style[name], lw=1.2)
        acf = r["acf"][size]
        for name, ls in (("L_par", "-"), ("sigma_l", "--")):
            c.plot(acf["lags_min"], [x["value"] for x in acf[name]], color=COLORS[size], ls=ls, lw=1.2)
    acf = r["acf"]["all"]
    c.plot(acf["lags_min"], [x["value"] for x in acf["f"]], color="#333333", ls=":", lw=1.4, label="$f$, all sizes")
    c.plot([], [], color="#333333", ls="-", label=r"$L_\parallel$")
    c.plot([], [], color="#333333", ls="--", label=r"$\sigma_\ell$")
    c.legend(frameon=False, fontsize=8)
    for size in SIZES[:4]:
        a.plot([], [], color=COLORS[size], label=f"{size} pilots")
    for name, ls in style.items():
        a.plot([], [], color="#333333", ls=ls, label=names[name])
    a.axhline(1, color="#999999", lw=0.6)
    a.set_xticks(range(4), ["$f=0$", "$0<f<.5$", r"$.5\leq f<1$", "$f=1$"])
    a.set(xlabel="climbing share of the members", ylabel="median relative to $f=0$")
    a.legend(frameon=False, fontsize=8)
    b.axhline(1, color="#999999", lw=0.6)
    b.axvline(0, color="#999999", lw=0.6)
    b.set(xlabel="time from the group entering a thermal (min)", ylabel="median relative to before the entry")
    c.axhline(0, color="#999999", lw=0.6)
    c.set(xlabel="lag (min)", ylabel="autocorrelation")
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0)")
    args = parser.parse_args()
    params = json.loads((LABELS / "days" / "params.json").read_text())
    step = params["step"]
    flights = pd.read_parquet(LABELS / "flights.parquet")
    groups = pd.read_parquet(LABELS / "groups.parquet")
    labelled = flights.set_index("flight")["labelled"]
    groups = groups[groups["t_end"] - groups["t_start"] >= MIN_GROUP]
    groups = groups[groups["members"].map(lambda m: bool(labelled.loc[list(m)].all()))].reset_index(drop=True)
    if args.sample:
        dates = np.random.default_rng(0).choice(sorted(groups["date"].unique()), args.sample, replace=False)
        groups = groups[groups["date"].isin(dates)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    assert groups["date"].is_monotonic_increasing  # measure() goes date by date
    pieces = measure(groups, flights, step, params)
    block = pd.read_parquet(LABELS / "episodes.parquet").set_index("episode_id")["block"]
    result = summarize(pieces, groups, block.reindex(groups["episode_id"]).to_numpy(), step, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"breathing_sample{args.sample}" if args.sample else "breathing"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps(result["groups"]), flush=True)
