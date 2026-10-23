"""Cohesion: how two pilots in contact keep together, against two pilots who meet by chance.

Per date, with the grid of the note (grid.day), the pair intervals of the run and its null model
(the same surrogate pairs, same seed), for the near pairs (takeoffs within r_s):

- Delay. Inside each contact, the delay l*(t) of the closest correspondence at each grid instant
  (contacts.scan); instants in a search are left out, as for ell_med. Real: the pair intervals
  lasting at least tau*. Chance: the contacts of the surrogate pairs, all of them and those
  lasting at least tau*, conditioned as the real ones.
- Separation. From the start of the first contact of each pair, real and surrogate, the
  horizontal distance at the same instant every SEP_STEP s up to SEP_MAX s, while both fly; also
  for the pairs whose first contact lasts at least tau*.

Summary: D(s) = <(l(t+s) - l(t))^2> over the pairs of instants of one contact, real by the size
of its configuration at t, and its Ornstein-Uhlenbeck fit D(s) = 2 sigma^2 (1 - exp(-k s)) + c
(c: the noise of l*), with the diffusivity 2 k sigma^2; the drift <l(t+DRIFT_S) - l(t) | l(t)> /
DRIFT_S made symmetric (the order of a pair is arbitrary), its slope -k for |l| <= SLOPE_S, and its
mean over all instants (offset); the median and quartiles of the distance after the first contact.
Bands: 5-95% of N_BOOT draws of the dates.

    uv run python cohesion.py [--jobs J] [--sample N]   # from this folder; full run about 20 min
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from tamsd import LABELS, OUT, SEED, save

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ideas" / "group-identification"))
import contacts  # noqa: E402
import grid  # noqa: E402
import run  # noqa: E402

SEP_STEP, SEP_MAX = 300, 3 * 3600  # s
SEPS = ("real", "chance", "real_long", "chance_long")
LAGS = np.array([1, 2, 4, 6, 9, 12, 18, 24, 36, 48, 60, 90, 120, 180, 240, 360])  # grid steps
DRIFT_S = 60
SLOPE_S = 80  # |l| up to which the drift is fitted by -k l
SIZES = ("2", "3", "4+")
N_BOOT = 200
CACHE = LABELS / "analyses" / "cohesion"


def series(tracks, i, j, on, off, p):
    """Grid instants of [on, off) at which i has a correspondence with j, the delay of the closest
    one in grid steps, and whether either pilot is in search there."""
    t = np.arange(on, off, tracks.step)
    ti, lag, search, _ = contacts.scan(tracks, i, j, t, p["delta"], p["eps_h"], p["eps_z"],
                                       np.cos(np.radians(p["theta_max"])))
    return ti.astype(np.int64), (lag // tracks.step).astype(np.int8), search


def separation(tracks, i, j, t0, span):
    """Horizontal distance at the same instant every SEP_STEP s from t0; NaN where either does not fly."""
    t = t0 + np.arange(0, SEP_MAX + 1, SEP_STEP)
    d = np.linalg.norm(tracks.at(i, t)[:, :3] - tracks.at(j, t)[:, :3], axis=1)
    d[(t < span[0]) | (t > span[1])] = np.nan
    return d.astype(np.float32)


def one_date(date, flights, runs, p):
    """Delay series of the real pair intervals (at least tau*) and of every surrogate contact, and
    the separation after the first contact (with whether that contact lasts at least tau*), for
    the near pairs of one date."""
    out = {"date": date, "real": [], "chance": [], "sep": {"real": [], "chance": []}}
    table, g = grid.day(flights, runs, p["step"], p["h_max"], p["r_dep"])
    if g is None:
        return out
    tracks = contacts.Tracks(g, p["step"])
    cached = pickle.loads((Path(p["run_days"]) / f"{date}.pkl").read_bytes())
    assert (cached["table"]["flight_id"].to_numpy() == table["flight_id"].to_numpy()).all()
    dep, land, fid = (table[c].to_numpy() for c in ("departure", "landing", "flight_id"))
    iv = cached["intervals"]
    iv = iv[iv["d_dec"] < p["r_s"]].sort_values("t_on")
    for (a, b), pair in iv.groupby(["a", "b"], sort=False):
        first = pair.iloc[0]
        d = separation(tracks, a, b, int(first["t_on"]), (max(dep[a], dep[b]), min(land[a], land[b])))
        out["sep"]["real"].append((d, first["t_off"] - first["t_on"] >= p["tau_star"]))
        for row in pair[pair["t_off"] - pair["t_on"] >= p["tau_star"]].itertuples():
            out["real"].append((fid[a], fid[b], row.t_on, row.t_off, *series(tracks, a, b, row.t_on, row.t_off, p)))
    rng = np.random.default_rng(int(date.replace("-", "")))  # the seed of run.one_date: the same surrogates
    for i, j, r in contacts.surrogates(table, rng, p, tracks.step):
        copy, instants, span = contacts.surrogate(table, g, tracks, i, j, r, p)
        found = contacts.pair_intervals(tracks, copy, i, instants, span, p) if instants is not None else []
        if not found:
            continue
        d = separation(tracks, copy, i, found[0][0], span)
        out["sep"]["chance"].append((d, found[0][1] - found[0][0] >= p["tau_star"]))
        for on, off, _ in found:
            out["chance"].append((fid[i], fid[j], on, off, *series(tracks, copy, i, on, off, p)))
    return out


def dense(t, lag, search, on, off, step):
    """The delay in seconds on every grid instant of [on, off), NaN without a correspondence or in search."""
    x = np.full((off - on) // step, np.nan)
    keep = ~search
    x[(t[keep] - on) // step] = lag[keep].astype(float) * step
    return x


def accumulate(x, cls, n_cls, step):
    """Per class of the instant t: sums of (l(t+s) - l(t))^2 and their counts at each lag of LAGS,
    and the drift sums by l(t) at DRIFT_S."""
    sums, counts = np.zeros((n_cls, len(LAGS))), np.zeros((n_cls, len(LAGS)))
    edges = np.arange(-125, 126, 10)
    drift, dcount = np.zeros((n_cls, len(edges) - 1)), np.zeros((n_cls, len(edges) - 1))
    for k, lag in enumerate(LAGS):
        if lag >= len(x):
            break
        d = x[lag:] - x[:-lag]
        ok = np.isfinite(d)
        np.add.at(sums[:, k], cls[:-lag][ok], d[ok] ** 2)
        np.add.at(counts[:, k], cls[:-lag][ok], 1)
    m = DRIFT_S // step
    if m < len(x):
        d = x[m:] - x[:-m]
        ok = np.isfinite(d)
        b = np.clip(np.searchsorted(edges, x[:-m][ok]) - 1, 0, len(edges) - 2)
        np.add.at(drift, (cls[:-m][ok], b), d[ok] / DRIFT_S)
        np.add.at(dcount, (cls[:-m][ok], b), 1)
    return sums, counts, drift, dcount


def ou(s, sigma2, k, c):
    return 2 * sigma2 * (1 - np.exp(-k * s)) + c


def fit(s, d):
    """sigma^2 (s^2), k (1/s) and c (s^2) of the Ornstein-Uhlenbeck fit; NaN if it fails."""
    try:
        return curve_fit(ou, s, d, p0=(d[-1] / 2, 1 / 300, d[0] / 2), sigma=d, bounds=(0, np.inf), maxfev=20000)[0]
    except (RuntimeError, ValueError):
        return np.full(3, np.nan)


def size_class(n):
    return np.where(n >= 4, 2, np.where(n == 3, 1, 0))


def summarize(days, flights, membership, step, tau_star, rng):
    """Per date: D(s) and drift sums of real (by size), chance long and chance, and the separation
    histograms; then the totals, the OU fits and the bands over dates."""
    gid = flights.set_index(["date", "flight_id"])["flight"]
    mb = membership[membership["state"] == "G"].sort_values(["flight", "t_start"])
    starts = dict(tuple(mb.groupby("flight")[["t_start", "t_end", "n"]]))
    per_date = {k: [] for k in ("real", "chance_long", "chance")}
    seps = {k: [] for k in SEPS}  # per date: distances, pairs x instants
    counts_iv = {"real": 0, "chance": 0, "chance_long": 0}
    for path in days:
        r = pickle.loads(path.read_bytes())
        acc = {"real": [np.zeros((3, len(LAGS))), np.zeros((3, len(LAGS))), np.zeros((3, 25)), np.zeros((3, 25))],
               "chance_long": [np.zeros((1, len(LAGS))), np.zeros((1, len(LAGS))), np.zeros((1, 25)), np.zeros((1, 25))],
               "chance": [np.zeros((1, len(LAGS))), np.zeros((1, len(LAGS))), np.zeros((1, 25)), np.zeros((1, 25))]}
        for a, _, on, off, t, lag, search in r["real"]:
            x = dense(t, lag, search, on, off, step)
            m = starts[gid[(r["date"], a)]]
            grid_t = np.arange(on, off, step)
            k = np.clip(np.searchsorted(m["t_start"].to_numpy(), grid_t, side="right") - 1, 0, len(m) - 1)
            cls = size_class(m["n"].to_numpy()[k])
            for z, y in zip(acc["real"], accumulate(x, cls, 3, step)):
                z += y
            counts_iv["real"] += 1
        for _, _, on, off, t, lag, search in r["chance"]:
            x = dense(t, lag, search, on, off, step)
            cls = np.zeros(len(x), int)
            kinds = ["chance"] + (["chance_long"] if off - on >= tau_star else [])
            for kind in kinds:
                for z, y in zip(acc[kind], accumulate(x, cls, 1, step)):
                    z += y
                counts_iv[kind] += 1
        for kind in per_date:
            per_date[kind].append(acc[kind])
        n_sep = SEP_MAX // SEP_STEP + 1
        for kind in ("real", "chance"):
            d = np.array([x for x, _ in r["sep"][kind]], np.float32).reshape(-1, n_sep)
            long = np.array([y for _, y in r["sep"][kind]], bool)
            seps[kind].append(d)
            seps[f"{kind}_long"].append(d[long])
    s = LAGS * step
    w = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=N_BOOT).astype(float)
    out = {"lags_s": s.tolist(), "intervals": counts_iv, "D": {}, "fit": {}, "drift": {}, "sep": {}}
    rows = {"real": [("2", 0), ("3", 1), ("4+", 2), ("all", None)], "chance_long": [("all", 0)], "chance": [("all", 0)]}
    for kind, items in rows.items():
        sums = np.array([a[0] for a in per_date[kind]])  # dates x classes x lags
        cnts = np.array([a[1] for a in per_date[kind]])
        dsum = np.array([a[2] for a in per_date[kind]])
        dcnt = np.array([a[3] for a in per_date[kind]])
        for name, c in items:
            pick = (lambda z: z.sum(axis=1)) if c is None else (lambda z, c=c: z[:, c])
            S, C, DS, DC = pick(sums), pick(cnts), pick(dsum), pick(dcnt)
            d = S.sum(0) / C.sum(0)
            draws = np.array([fit(s, (wi @ S) / (wi @ C)) for wi in w])
            f = fit(s, d)
            key = f"{kind}/{name}"
            out["D"][key] = {"value": d.tolist(), "band": np.nanpercentile((w @ S) / (w @ C), [5, 95], axis=0).tolist(),
                             "pairs": C.sum(0).tolist()}
            out["fit"][key] = {"sigma": float(np.sqrt(f[0])), "k": float(f[1]), "c": float(f[2]),
                               "sigma_band": np.nanpercentile(np.sqrt(draws[:, 0]), [5, 95]).tolist(),
                               "k_band": np.nanpercentile(draws[:, 1], [5, 95]).tolist(),
                               "q": float(2 * f[0] * f[1]),
                               "q_band": np.nanpercentile(2 * draws[:, 0] * draws[:, 1], [5, 95]).tolist()}
            # the order of a pair is arbitrary: the drift at l and minus the drift at -l
            SS, SC = DS - DS[:, ::-1], DC + DC[:, ::-1]
            ell = np.arange(-120, 121, 10.0)
            near = np.abs(ell) <= SLOPE_S

            def slope(f_ell, n):
                return float(-(n * ell * f_ell)[near].sum() / (n * ell ** 2)[near].sum())

            sym = SS.sum(0) / SC.sum(0)
            sym_draws = (w @ SS) / (w @ SC)
            n_bins = SC.sum(0)
            out["drift"][key] = {"l_s": ell.tolist(), "value": sym.tolist(),
                                 "band": np.nanpercentile(sym_draws, [5, 95], axis=0).tolist(), "count": n_bins.tolist(),
                                 "offset": float(DS.sum() / DC.sum()), "k": slope(sym, n_bins),
                                 "k_band": np.percentile([slope(d, n_bins) for d in sym_draws], [5, 95]).tolist()}
    for kind in SEPS:
        d = np.concatenate(seps[kind])
        date = np.repeat(np.arange(len(days)), [len(x) for x in seps[kind]])
        draws = np.array([quartiles(d[np.repeat(np.arange(len(d)), wi[date].astype(int))]) for wi in w[:50]])
        q = quartiles(d)
        out["sep"][kind] = {"s_min": (np.arange(d.shape[1]) * SEP_STEP / 60).tolist(),
                            "pairs": np.isfinite(d).sum(0).tolist(), "q25": q[0].tolist(), "median": q[1].tolist(),
                            "q75": q[2].tolist(), "median_band": np.nanpercentile(draws[:, 1], [5, 95], axis=0).tolist()}
    return out


def quartiles(d):
    """25th, 50th and 75th percentiles over pairs at each instant, NaN with fewer than 20 pairs."""
    if not len(d):
        return np.full((3, d.shape[1]), np.nan)
    q = np.nanpercentile(d, [25, 50, 75], axis=0)
    q[:, np.isfinite(d).sum(0) < 20] = np.nan
    return q


def plot(r, path):
    """Left: D(s) with the OU fits; middle: drift of the delay; right: median separation after the first contact."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    s = np.array(r["lags_s"])
    styles = {"real/2": ("#eb6834", "-", "o", "groups, 2 pilots"), "real/3": ("#c4472a", "-", "s", "groups, 3"),
              "real/4+": ("#8e2e1c", "-", "^", "groups, 4+"),
              "chance_long/all": ("#2a78d6", "-", "D", r"chance, $\geq\tau^*$"),
              "chance/all": ("#2a78d6", ":", "v", "chance, all")}
    for key, (color, ls, marker, label) in styles.items():
        d, f = r["D"][key], r["fit"][key]
        a.fill_between(s / 60, *d["band"], color=color, alpha=0.15, lw=0)
        a.plot(s / 60, d["value"], color=color, ls=ls, marker=marker, ms=4, lw=1.2,
               label=f"{label}: $1/k$ = {1 / f['k'] / 60:.1f} min")
        dr = r["drift"][key]
        ok = np.array(dr["count"]) >= 200
        x = np.array(dr["l_s"])[ok]
        b.fill_between(x, *np.array(dr["band"])[:, ok], color=color, alpha=0.15, lw=0)
        b.plot(x, np.array(dr["value"])[ok], color=color, ls=ls, marker=marker, ms=4, lw=1.2)
    a.set(xscale="log", yscale="log", xlabel="lag $s$ (min)", ylabel=r"$D(s)$ (s$^2$)")
    a.legend(frameon=False, fontsize=7)
    b.axhline(0, color="#999999", lw=0.6)
    b.set(xlabel=r"delay $\ell$ (s)", ylabel=rf"symmetric drift $\langle\Delta\ell\rangle/\Delta$, $\Delta$ = {DRIFT_S} s")
    for key, color, ls, label in [("real", "#eb6834", "-", "real pairs"), ("chance", "#2a78d6", "-", "chance"),
                                  ("real_long", "#eb6834", "--", r"real, first contact $\geq\tau^*$"),
                                  ("chance_long", "#2a78d6", "--", r"chance, first contact $\geq\tau^*$")]:
        sp = r["sep"][key]
        t = np.array(sp["s_min"])
        c.fill_between(t, *np.array(sp["median_band"]) / 1000, color=color, alpha=0.15, lw=0)
        c.plot(t, np.array(sp["median"]) / 1000, color=color, ls=ls, lw=1.4, label=label)
    c.set(yscale="log", xlabel="time since the first contact (min)", ylabel="median distance (km)")
    c.legend(frameon=False, fontsize=7)
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0), cached apart")
    args = parser.parse_args()
    p = json.loads((LABELS / "days" / "params.json").read_text())
    p["tau_star"] = json.loads((LABELS / "calibration.json").read_text())["tau_star"]
    p["run_days"] = str(LABELS / "days")
    folder = CACHE / (f"sample{args.sample}" if args.sample else "all") / "days"
    folder.mkdir(parents=True, exist_ok=True)
    flights = pd.read_parquet(LABELS / "flights.parquet")
    every = sorted(f.stem for f in (LABELS / "days").glob("*.pkl") if not f.name.startswith("."))
    dates = sorted(np.random.default_rng(0).choice(every, args.sample, replace=False)) if args.sample else every
    run.compute(grid.archive(), dates, folder, args.jobs, job=one_date, params=p)
    result = summarize([folder / f"{d}.pkl" for d in dates], flights, pd.read_parquet(LABELS / "membership.parquet"),
                       p["step"], p["tau_star"], np.random.default_rng(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"cohesion_sample{args.sample}" if args.sample else "cohesion"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print({k: (round(1 / v["k"] / 60, 2), round(v["sigma"], 1)) for k, v in result["fit"].items()}, flush=True)
