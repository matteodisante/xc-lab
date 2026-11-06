"""Headings: in glide, does the follower turn where the head turned or when the head turned?

Population: the glides of glide.py (groups of 2 and of 3+, all members 1 Hz flights with phases,
and the chance pairs of the surrogate contacts), taken on the 5 s grid. Every pair of members of a
group (of the two of a chance pair) is a pair-glide.

Quantity. In the glide, the horizontal velocity of each member is the central displacement over the
neighbouring instants, in the east-north frame of the centroid; its heading theta is the angle of
that velocity, and dtheta = theta - the direction of the centroid displacement from the first to
the last instant of the glide, wrapped to (-pi, pi]. The pair (i, j) is oriented so that i is
ahead: l = the median over the glide of x_j - x_i, with x the passage times of the group, is at
least 0. The correlation C(tau) = corr(dtheta_i(t), dtheta_j(t + tau)) is taken over the instants
where both exist inside the glide, at least MIN_GLIDE_S of them, each series centred on that
overlap; tau > 0 when j follows i. A follower who retraces the head's track peaks at tau = l; one
who turns when the head turns peaks at a short tau whatever l. The same is done on dv, the
horizontal speed (its mean is removed by the correlation). Per pair-glide the centred sums are kept,
so that the curve of a class is pooled as sum Sxy / sqrt(sum Sxx sum Syy); its maximum is refined
by a parabola through the grid maximum and its neighbours, and its value C(l) at the median l of
the class is read by linear interpolation (retracing gives C(l) its largest value; the maximum of a
curve without a peak would only pick the highest noise). Per kind, the OLS slope of the per-pair
tau_max on l.

Null: the chance pairs, two pilots who never met at the same place (glide.chance_series), in the
same classes of l as the real pairs.

Constants, measured on the real pair-glides of --sample 200: TAU_MAX is their 95th percentile of l
rounded up to the grid; L_EDGES their quartiles of l, the same for every kind.

Bands: 5-95% of N_BOOT draws of the dependency blocks of the note (episodes) for the groups, whose
draws the two sizes share; the chance pairs belong to no episode, so their blocks are their dates.

    uv run python headings.py [--run NAME] [--sample N]    # from this folder, after cohesion.py
"""

import argparse
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import glide
from breathing import MIN_GROUP, N_BOOT, ROUTE_S, boot_median, route, stat
from cohesion import CACHE as COHESION
from glide import COLORS, KINDS, LABEL, MIN_GLIDE_S, glides
from tamsd import LABELS, OUT, SEED, save

TAU_MAX = 120  # largest |tau|, s: 95th percentile of l over the real pair-glides (120.0 s), rounded up to the grid
L_EDGES = (10.0, 30.0, 65.0)  # edges of the classes of l, s: quartiles of l over the real pair-glides (10, 30, 65 s)
SIGNALS = ("heading", "speed")


def wrap(a):
    """The angle `a` brought to (-pi, pi]."""
    return np.pi - np.mod(np.pi - a, 2 * np.pi)


def kinematics(xyz, flat, step):
    """The heading and the speed (n, T) of the members, `xyz` (n, T, 3), from the central
    displacement over the neighbouring instants in the frame `flat` of the route; NaN at the ends
    and without a fix."""
    n, steps = xyz.shape[:2]
    v = np.full((n, steps, 2), np.nan)
    v[:, 1:-1] = flat(xyz[:, 2:] - xyz[:, :-2], 1, steps - 1) / (2 * step)
    return np.arctan2(v[..., 1], v[..., 0]), np.hypot(v[..., 0], v[..., 1])


def lagged_sums(x, y, k, step):
    """For the lags -k..k (in grid steps) of the series `x`, `y` (s, L) of one pair-glide: the number
    n of instants where x(t) and y(t + lag) both exist and the centred sums [Sxy, Sxx, Syy]
    (s, 3, 2k + 1) on that overlap; both zero where the overlap is shorter than MIN_GLIDE_S."""
    length = x.shape[1]
    shifted = sliding_window_view(np.pad(y, ((0, 0), (k, k)), constant_values=np.nan), length, axis=1)
    ok = np.isfinite(x)[:, None] & np.isfinite(shifted)
    ok = ok.all(axis=0)  # the signals exist together
    n = ok.sum(axis=1)
    n = np.where(n * step >= MIN_GLIDE_S, n, 0)
    keep = ok & (n > 0)[:, None]
    a, b = np.where(keep, x[:, None], 0.0), np.where(keep, shifted, 0.0)
    m = np.maximum(n, 1)
    sx, sy = a.sum(-1), b.sum(-1)
    return n, np.stack([(a * b).sum(-1) - sx * sy / m, (a * a).sum(-1) - sx ** 2 / m, (b * b).sum(-1) - sy ** 2 / m], axis=1)


def correlation(s):
    """The Pearson correlation (..., lags) from the sums (..., 3, lags) [Sxy, Sxx, Syy]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return s[..., 0, :] / np.sqrt(s[..., 1, :] * s[..., 2, :])


def one_series(xyz, z, x, f, step):
    """The pair-glides of one group or chance pair (`xyz` (n, T, 3), `z` (n, T), `x` (T, n), `f`
    (T,)): a list of (l, n, S, tau_max) with l the delay of the pair in s (the first member ahead),
    n the overlap at each lag, S the sums (2 SIGNALS, 3, 2k + 1) in float32 and tau_max (2,) the lag
    of the per-pair maximum in s (NaN without one)."""
    found = glides(f, step)
    if not found:
        return []
    n, k = len(xyz), TAU_MAX // step
    flat, _ = route(xyz, 1)
    theta, speed = kinematics(xyz, flat, step)
    c = xyz.mean(axis=0)
    out = []
    for a, e in found:
        live = np.flatnonzero(np.isfinite(c[a:e, 0])) + a
        if len(live) < 2:
            continue
        mid = live[len(live) // 2]
        d = flat((c[live[-1]] - c[live[0]])[None], mid, mid + 1)[0]
        signal = np.stack([wrap(theta[:, a:e] - np.arctan2(d[1], d[0])), speed[:, a:e]])
        for i, j in itertools.combinations(range(n), 2):
            both = np.isfinite(x[a:e, [i, j]]).all(axis=1)
            if both.sum() * step < MIN_GLIDE_S:
                continue
            l = float(np.median(x[a:e][both, j] - x[a:e][both, i]))
            if l < 0:
                i, j, l = j, i, -l
            count, sums = lagged_sums(signal[:, i], signal[:, j], k, step)
            c_lag = correlation(sums)
            top = np.where(np.isfinite(c_lag), c_lag, -np.inf).argmax(axis=-1)
            tau = np.where(np.isfinite(c_lag).any(axis=-1), (top - k) * step, np.nan)
            out.append((l, count.astype(np.int16), sums.astype(np.float32), tau))
    return out


def peak(c, step):
    """The lag in s of the maximum of the curves `c` (..., 2k + 1), refined by a parabola through the
    grid maximum and its neighbours, and the value there; NaN for a curve without a value."""
    n = c.shape[-1]
    m = np.where(np.isfinite(c), c, -np.inf).argmax(axis=-1)
    at = lambda i: np.take_along_axis(c, np.clip(i, 0, n - 1)[..., None], -1)[..., 0]
    y0, lo, hi = at(m), at(m - 1), at(m + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = lo - 2 * y0 + hi
        fit = (m > 0) & (m < n - 1) & (den < 0)
        shift = np.where(fit, 0.5 * (lo - hi) / np.where(fit, den, 1), 0.0)
        top = np.where(fit, y0 - (lo - hi) ** 2 / (8 * np.where(fit, den, 1)), y0)
    seen = np.isfinite(c).any(axis=-1)
    return np.where(seen, (m + shift - n // 2) * step, np.nan), np.where(seen, top, np.nan)


def slope(l, t, k, w):
    """The OLS slope of t on l and its draws; `k` is the block of each pair-glide, `w` the multiplicity
    of the blocks in each draw (draws, blocks)."""
    ok = np.isfinite(t)
    l, t, k = l[ok], t[ok], k[ok]
    s = np.array([np.bincount(k, v, w.shape[1]) for v in (np.ones_like(l), l, t, l * l, l * t)])
    r = lambda n, sl, st, sll, slt: (n * slt - sl * st) / (n * sll - sl ** 2)
    return float(r(*s.sum(axis=1))), r(*(s @ w.T))


def median(x, k, w):
    """boot_median, NaN without values."""
    return boot_median(x, k, w) if len(x) else (np.nan, np.full(len(w), np.nan))


def summarize(units, rng, step):
    """The classes of l and the slopes by kind, with their bands."""
    np.seterr(divide="ignore", invalid="ignore")
    lags = np.arange(-(TAU_MAX // step), TAU_MAX // step + 1) * step
    out = {"tau": lags.tolist(), "classes": {}, "slope": {}}
    k = TAU_MAX // step
    for population in (("pairs", "3+"), ("chance",)):
        sel = [u for u in units if u[0] in population]
        ids, key = np.unique([str(u[1]) for u in sel], return_inverse=True)
        w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)
        for kind in population:
            recs = [(b, r) for b, u in zip(key, sel) if u[0] == kind for r in u[2]]
            owner = np.array([b for b, _ in recs], int)
            l = np.array([r[0] for _, r in recs])
            tau = np.array([r[3] for _, r in recs])  # (N, 2)
            cls = np.searchsorted(L_EDGES, l, side="right")
            out["slope"][kind] = {s: stat(*slope(l, tau[:, q], owner, w)) for q, s in enumerate(SIGNALS)}
            out["classes"][kind] = []
            for c in range(len(L_EDGES) + 1):
                here = np.flatnonzero(cls == c)
                present, inv = np.unique(owner[here], return_inverse=True)
                by_block = np.zeros((len(present), 2, 3, 2 * k + 1))
                for i, b in zip(here, inv):
                    by_block[b] += recs[i][1][2]
                total, draws = by_block.sum(axis=0), np.tensordot(w[:, present], by_block, axes=(1, 0))
                curve, curves = correlation(total), correlation(draws)
                t_max, c_max = peak(curve, step)
                t_draw, c_draw = peak(curves, step)
                entry = {"n": int(len(here)), "l_median": stat(*median(l[here], owner[here], w))}
                at_l = lambda c: np.interp(entry["l_median"]["value"], lags, c)
                for q, s in enumerate(SIGNALS):
                    entry[s] = {"curve": [stat(v, d) for v, d in zip(curve[q], curves[:, q].T)],
                                "tau_max": stat(float(t_max[q]), t_draw[:, q]), "C_max": stat(float(c_max[q]), c_draw[:, q]),
                                "C_at_l": stat(float(at_l(curve[q])), np.array([at_l(d) for d in curves[:, q]])),
                                "pair_tau_max_median": stat(*median(tau[here, q][np.isfinite(tau[here, q])],
                                                                    owner[here][np.isfinite(tau[here, q])], w))}
                out["classes"][kind].append(entry)
    real = np.array([r[0] for u in units if u[0] != "chance" for r in u[2]])
    out["measured"] = {"l_p95": float(np.percentile(real, 95)), "l_quartiles": np.percentile(real, [25, 50, 75]).tolist(), "n_real": len(real)}
    return out


def plot(r, path):
    """Left: the pooled heading correlation of the pairs against tau - l, l the median delay of each
    class: a follower who turns where the head turned peaks at 0 in every class, one who turns when
    the head turns at -l (ticks). Right: the pooled heading correlation at tau = l against l, by kind."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    tau = np.array(r["tau"])
    rows = r["classes"]["pairs"]
    for row, color in zip(rows, plt.get_cmap("Purples")(np.linspace(0.45, 1.0, len(rows)))):
        l, curve = row["l_median"]["value"], row["heading"]["curve"]
        a.fill_between(tau - l, *np.array([x["band"] for x in curve]).T, color=color, alpha=0.25, lw=0)
        a.plot(tau - l, [x["value"] for x in curve], color=color, lw=1.4, label=rf"$\ell \approx {l:.0f}$ s")
        a.plot(2 * [-l], [0, 0.04], color=color, lw=2, transform=a.get_xaxis_transform())
    a.axvline(0, color="#999999", lw=0.8)
    a.set(xlabel=r"$\tau - \ell$ (s)", ylabel=r"pooled $C(\tau)$ of the heading, pairs")
    a.legend(frameon=False, fontsize=8)
    for kind in KINDS:
        rows = r["classes"][kind]
        x = [c["l_median"]["value"] for c in rows]
        y = np.array([c["heading"]["C_at_l"]["value"] for c in rows])
        lo, hi = np.array([c["heading"]["C_at_l"]["band"] for c in rows]).T
        b.errorbar(x, y, yerr=[y - lo, hi - y], color=COLORS[kind], ls="--" if kind == "chance" else "-", marker="o", ms=4,
                   lw=1.2, capsize=2, label=LABEL[kind])
    b.axhline(0, color="#999999", lw=0.5)
    b.set(xlabel=r"median delay $\ell$ of the class (s)", ylabel=r"pooled $C(\ell)$ of the heading", xlim=(0, None))
    b.legend(frameon=False, fontsize=8)
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
    units = glide.measure(groups, flights, sorted(dates), step, params, tau_star, per_series=one_series)
    result = summarize(units, np.random.default_rng(SEED), step)
    result["params"] = {"tau_max": TAU_MAX, "l_edges": list(L_EDGES), "min_glide_s": MIN_GLIDE_S, "tau_star": tau_star, "route_s": ROUTE_S}
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"headings_sample{args.sample}" if args.sample else "headings"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps({k: result[k] for k in ("measured", "slope")}), flush=True)
