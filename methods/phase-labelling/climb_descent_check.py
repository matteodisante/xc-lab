"""Do the climb runs of the refit (and of Vilpellet) climb, and are they aligned with the track?

Two samples of viewer flights: 3000 logged at 1 s that have both label sets, and 2000 logged slower that
have the refit only. `extract` keeps their fixes in --cache, `analyse` writes out/climb_descent_check.{json,pdf}.
Time offset: the shift s such that the climb indicator at time t - s best matches the viewer's v_z at t
(s < 0: the labels lead the track).

    python climb_descent_check.py extract [--slow] --cache DIR
    python climb_descent_check.py analyse --cache DIR
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DER = Path("/Volumes/SSD_DATA/xc-viewer-data/paragliders/derived")
GROUND = Path("/Users/matteodisante/local_master_thesis/xc-lab/analyses/groups/out/d120_eh500_ez300_th45_g120_q0.1/open_declared/ground.parquet")
SEED = 2026
TAG = {False: "", True: "_slow"}
SHIFTS = np.arange(-120, 121)
BINS = [0, 30, 60, 120, 300, 600, 1e9]
CODE = {"climb": 0, "search": 1, "transition": 2}
GROUND_FLIGHTS = Path("/Volumes/SSD_DISANTE/xc-lab/group-identification/runs/d120_eh500_ez300_th45_g120_q0.1/flights.parquet")
BIN_NAMES = ["<30 s", "30-60 s", "60-120 s", "120-300 s", "300-600 s", ">600 s"]


def sample_ids(n, slow=False):
    """Viewer flights with refit and Vilpellet runs (all logged at 1 s), or, with slow, with refit runs
    only and a logger interval above 1 s."""
    meta = pd.read_parquet(DER / "flights_meta.parquet", columns=["flight_id", "drop_stage", "dt_native_s"])
    ok = set(meta.flight_id[meta.drop_stage.isna() & ((meta.dt_native_s > 1) if slow else True)])
    for k in (("refit",) if slow else ("refit", "vilpellet")):
        col = pq.read_table(DER / f"segmentation/{k}/phase_segments.parquet", columns=["flight_id"]).column(0)
        ok &= set(pc.unique(col).to_pylist())
    ids = np.array(sorted(ok, key=int))
    return np.sort(np.random.default_rng(SEED).choice(ids, size=n, replace=False))


def extract(n, cache, slow=False):
    ids = sample_ids(n, slow)
    value_set = pa.array(ids)
    pf = pq.ParquetFile(DER / "fixes.parquet")
    parts, t0 = [], time.time()
    for g in range(pf.num_row_groups):
        tab = pf.read_row_group(g, columns=["flight_id", "segment_id", "t", "z", "v_z", "interpolated"])
        tab = tab.filter(pc.is_in(tab["flight_id"], value_set=value_set))
        if tab.num_rows:
            parts.append(tab)
        if g % 100 == 0:
            print(g, pf.num_row_groups, round(time.time() - t0), flush=True)
    pq.write_table(pa.concat_tables(parts), cache / f"sample_fixes{TAG[slow]}.parquet")



def runs_of(kind, ids):
    t = pq.read_table(DER / f"segmentation/{kind}/phase_segments.parquet",
                      columns=["flight_id", "phase", "t_start", "t_end"], filters=[("flight_id", "in", list(ids))]).to_pandas()
    return t.sort_values(["flight_id", "t_start"], kind="stable").reset_index(drop=True)


class Fixes:
    """The fixes of the sample, sorted by flight and time; key = flight number * 1e6 + t + 2e5."""

    def __init__(self, path):
        col = lambda name: pq.read_table(path, columns=[name])[name].combine_chunks()
        fid = pc.cast(col("flight_id"), pa.int32()).to_numpy()
        t = col("t").to_numpy()
        order = np.lexsort((t, fid))
        ids_int, flight = np.unique(fid[order], return_inverse=True)
        del fid
        self.ids = np.array([str(i) for i in ids_int])
        self.code = {k: i for i, k in enumerate(self.ids)}
        self.t = t[order].astype(np.float64)
        del t
        self.z = col("z").to_numpy()[order]
        self.vz = col("v_z").to_numpy()[order]
        del order
        self.key = flight * 1e6 + self.t + 2e5
        self.bounds = np.r_[0, np.cumsum(np.bincount(flight, minlength=len(self.ids)))]


def ground_classes(ids):
    g = pd.read_parquet(GROUND, columns=["flight", "class"])
    m = pd.read_parquet(GROUND_FLIGHTS, columns=["flight", "flight_id"]).merge(g, on="flight")
    m = m.assign(flight_id=m.flight_id.astype(np.int64)).drop_duplicates("flight_id").set_index("flight_id")["class"]
    return pd.Series([m.get(int(k), "not in the ground table") for k in ids], index=ids)


def climb_stats(runs, fix):
    """Net height change, mean vertical speed and duration of every climb run, from the viewer's fixes."""
    r = runs[runs.phase == "climb"]
    c = r.flight_id.map(fix.code).to_numpy() * 1e6 + 2e5
    lo = np.searchsorted(fix.key, c + r.t_start.to_numpy() - 1e-3, "left")
    hi = np.searchsorted(fix.key, c + r.t_end.to_numpy() + 1e-3, "right") - 1
    ok = hi > lo
    out = r[ok].copy()
    lo, hi = lo[ok], hi[ok]
    out["dz"] = fix.z[hi].astype(float) - fix.z[lo]
    out["dur"] = fix.t[hi] - fix.t[lo]
    out["vz"] = out.dz / out.dur
    out["dbin"] = pd.cut(out.dur, BINS, labels=BIN_NAMES, right=False)
    return out, int((~ok).sum())


def share_table(df, by):
    res = {}
    for k, d in df.groupby(by, observed=True):
        res[str(k)] = dict(runs=len(d), time_h=float(d.dur.sum() / 3600), share_runs_descent=float((d.dz < 0).mean()),
                           share_time_descent=float(d.dur[d.dz < 0].sum() / d.dur.sum()),
                           share_runs_vz_below_m05=float((d.vz < -0.5).mean()),
                           share_time_vz_below_m05=float(d.dur[d.vz < -0.5].sum() / d.dur.sum()),
                           median_vz=float(d.vz.median()), time_weighted_mean_vz=float(d.dz.sum() / d.dur.sum()))
    return res


def join(runs, fix):
    """Phase code of every viewer fix, as xc_viewer.data.label_from_runs gives it (-1: no run)."""
    c = runs.flight_id.map(fix.code).to_numpy() * 1e6 + 2e5
    start, end = c + runs.t_start.to_numpy(), c + runs.t_end.to_numpy()
    i = np.searchsorted(start - 1e-3, fix.key, "right") - 1
    j = np.maximum(i, 0)
    inside = (i >= 0) & (fix.key <= end[j] + 1e-3)
    return np.where(inside, runs.phase.map(CODE).to_numpy()[j], -1).astype(np.int8)


def fix_stats(lab, fix):
    out = {}
    for name, c in list(CODE.items()) + [("unlabelled", -1)]:
        m = lab == c
        v = fix.vz[m]
        v = v[~np.isnan(v)]
        out[name] = dict(share_of_fixes=float(m.mean()), mean_vz=float(v.mean()) if len(v) else None,
                         share_vz_below_0=float((v < 0).mean()) if len(v) else None,
                         share_vz_below_m05=float((v < -0.5).mean()) if len(v) else None)
    return out


def correlations(t, vz, start, end, climb):
    """Pearson correlation of the climb indicator, shifted by each of SHIFTS seconds, with v_z on a 1-s grid."""
    grid = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
    if len(grid) < 600:
        return None
    v = np.interp(grid, t, np.nan_to_num(vz))
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = (t[i] - t[i - 1]) <= 10
    r = np.searchsorted(start, grid, "right") - 1
    lab = (r >= 0) & (grid <= end[np.maximum(r, 0)] + 1e-3)
    cl = lab & climb[np.maximum(r, 0)]
    n = len(grid)
    out = np.full(len(SHIFTS), np.nan)
    for k, s in enumerate(SHIFTS):
        sl_v, sl_c = (slice(s, n), slice(0, n - s)) if s >= 0 else (slice(0, n + s), slice(-s, n))
        m = near[sl_v] & lab[sl_c]
        if m.sum() < 600:
            continue
        x, y = v[sl_v][m], cl[sl_c][m].astype(float)
        if x.std() > 0 and y.std() > 0:
            out[k] = ((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std())
    return out


def alignment(runs, fix):
    rows = []
    for f, g in runs.groupby("flight_id"):
        c = fix.code[f]
        a, b = fix.bounds[c], fix.bounds[c + 1]
        corr = correlations(fix.t[a:b], fix.vz[a:b], g.t_start.to_numpy(), g.t_end.to_numpy(), (g.phase == "climb").to_numpy())
        if corr is None or np.isnan(corr).all():
            continue
        j = int(np.nanargmax(corr))
        rows.append((f, int(SHIFTS[j]), corr[j], corr[SHIFTS == 0][0]))
    al = pd.DataFrame(rows, columns=["flight_id", "offset", "corr_best", "corr_0"])
    al["gain"] = al.corr_best - al.corr_0
    return al


def agreement(runs_a, runs_b, fix):
    """Per flight: share of viewer fixes labelled by both with the same phase."""
    a, b = join(runs_a, fix), join(runs_b, fix)
    flight = np.repeat(np.arange(len(fix.ids)), np.diff(fix.bounds))
    both = (a >= 0) & (b >= 0)
    n = np.bincount(flight, weights=both, minlength=len(fix.ids))
    same = np.bincount(flight, weights=both & (a == b), minlength=len(fix.ids))
    return pd.Series(same / np.maximum(n, 1), index=fix.ids).where(n > 0)


def _timebase_flight(item):
    """Shift in [-30, 30] s of our pre-processed track (in the time base of the saved runs) that best matches the
    viewer's heights, by 10-s height differences."""
    import refit

    f, t, z, shift, path, disc = item
    flight, _ = refit.read_flight(path, disc, refit.config())
    if flight is None:
        return f, None
    to, zo = flight["t"] - shift, flight["xyz"][2]
    keep = np.r_[True, np.diff(to) > 0]
    to, zo = to[keep], zo[keep]
    k = max(2, int(round(10 / np.median(np.diff(t)))))
    err = []
    for o in range(-30, 31):
        m = (t + o >= to[0]) & (t + o <= to[-1])
        if m.sum() < 50:
            err.append(np.inf)
            continue
        a, b = z[m].astype(float), np.interp(t[m] + o, to, zo)
        err.append(np.median(np.abs((a[k:] - a[:-k]) - (b[k:] - b[:-k]))))
    return f, int(np.argmin(err)) - 30


def timebase(cache, meta, slow, n):
    """Per flight of an evenly spaced subsample: the best shift between our track and the viewer's, in seconds."""
    import refit

    fx = pq.read_table(cache / f"sample_fixes{TAG[slow]}.parquet", columns=["flight_id", "t", "z", "interpolated"]).to_pandas()
    fx = fx[~fx.interpolated]
    ids = sorted(fx.flight_id.unique(), key=int)
    ids = ids[:: max(1, len(ids) // n)][:n]
    cat = refit.catalogue("para").set_index("flight_id").local_path
    tracks = refit.DISCIPLINES["para"]["tracks"]
    items = [(f, g.t.to_numpy(float), g.z.to_numpy(float), meta.ground_phase_start_s[f], tracks / cat[int(f)], "para")
             for f, g in fx[fx.flight_id.isin(ids)].groupby("flight_id")]
    del fx
    with ProcessPoolExecutor(3) as pool:
        res = dict(pool.map(_timebase_flight, items, chunksize=2))
    return pd.Series({f: v for f, v in res.items() if v is not None})


def analyse_sample(cache, slow):
    fix = Fixes(cache / f"sample_fixes{TAG[slow]}.parquet")
    kinds = ("refit",) if slow else ("refit", "vilpellet")
    runs = {k: runs_of(k, fix.ids) for k in kinds}
    gclass = ground_classes(fix.ids)
    res, tab = {}, {}
    for k in kinds:
        d, dropped = climb_stats(runs[k], fix)
        d["gclass"] = d.flight_id.map(gclass)
        lab = join(runs[k], fix)
        al = alignment(runs[k], fix)
        res[k] = dict(climb_runs=len(d), climb_runs_with_under_two_fixes=dropped, all=share_table(d.assign(a="all"), "a")["all"],
                      by_duration=share_table(d, "dbin"), by_ground_class=share_table(d, "gclass"),
                      fixes_by_phase=fix_stats(lab, fix), alignment=align_summary(al))
        tab[k] = (d, al)
    if not slow:
        ag = agreement(runs["refit"], runs["vilpellet"], fix)
        res["agreement_refit_vs_vilpellet"] = dict(median=float(ag.median()), mean=float(ag.mean()))
        tab["agree"] = ag
    meta = pd.read_parquet(DER / "flights_meta.parquet", columns=["flight_id", "ground_phase_start_s", "dt_native_s", "n_segments",
                                                                  "was_resampled", "n_fix_clean", "duration_flight_s"]).set_index("flight_id")
    return res, tab, meta, fix


def align_summary(al):
    big = al.offset.abs() >= 10
    return dict(flights=len(al), offset_median=float(al.offset.median()), offset_q05=float(al.offset.quantile(0.05)),
                offset_q95=float(al.offset.quantile(0.95)), share_zero=float((al.offset == 0).mean()),
                share_abs_ge_3=float((al.offset.abs() >= 3).mean()), share_abs_ge_10=float(big.mean()),
                share_abs_ge_30=float((al.offset.abs() >= 30).mean()),
                corr_at_best_median=float(al.corr_best.median()), corr_at_zero_median=float(al.corr_0.median()),
                gain_median=float(al.gain.median()), gain_q95=float(al.gain.quantile(0.95)),
                share_gain_ge_0p05=float((al.gain >= 0.05).mean()), share_gain_ge_0p1=float((al.gain >= 0.1).mean()))


def features(al, meta, agree):
    """Compare the flights with |offset| >= 10 s to the others."""
    d = al.set_index("flight_id").join(meta)
    if agree is not None:
        d["agreement"] = agree
    d["big"] = d.offset.abs() >= 10
    out = {}
    for big, g in d.groupby("big"):
        out["abs_offset_ge_10" if big else "abs_offset_lt_10"] = dict(
            flights=len(g), corr_at_best_median=float(g.corr_best.median()), n_segments_mean=float(g.n_segments.mean()),
            share_3_or_more_segments=float((g.n_segments >= 3).mean()), was_resampled_share=float(g.was_resampled.mean()),
            n_fix_clean_median=float(g.n_fix_clean.median()), ground_phase_start_s_median=float(g.ground_phase_start_s.median()),
            share_ground_phase_start_s_above_300=float((g.ground_phase_start_s > 300).mean()),
            share_ground_phase_start_s_zero=float((g.ground_phase_start_s == 0).mean()),
            dt_native_s_mean=float(g.dt_native_s.mean()),
            **({"agreement_median": float(g.agreement.median())} if agree is not None else {}))
    if agree is not None:
        bins = pd.cut(d.offset.abs(), [-1, 3, 9, 29, 1e9], labels=["0-3 s", "4-9 s", "10-29 s", ">=30 s"])
        out["agreement_by_abs_offset"] = {str(k): dict(flights=len(g), median=float(g.agreement.median()))
                                          for k, g in d.groupby(bins, observed=True)}
    return out


def examples(d, al, k=6):
    """A few flight ids for each case."""
    pos = d[(d.dz < -50) & (d.dur >= 120)].nsmallest(k * 3, "dz").drop_duplicates("flight_id").head(k)
    by_flight = d.groupby("flight_id").apply(lambda g: pd.Series(dict(
        climb_h=g.dur.sum() / 3600, share_time_descent=g.dur[g.dz < 0].sum() / g.dur.sum())), include_groups=False)
    worst = by_flight[by_flight.climb_h >= 0.3].nlargest(k, "share_time_descent")
    mis = al[al.offset.abs() >= 10].nlargest(k, "gain")
    return dict(
        long_descending_climb_runs=[dict(flight_id=r.flight_id, t_start=float(r.t_start), t_end=float(r.t_end), net_dz_m=round(float(r.dz), 1))
                                    for r in pos.itertuples()],
        flights_most_climb_time_in_descent=[dict(flight_id=f, share=round(float(r.share_time_descent), 3)) for f, r in worst.iterrows()],
        largest_offset_gain=[dict(flight_id=r.flight_id, offset_s=int(r.offset), corr_gain=round(float(r.gain), 3)) for r in mis.itertuples()])


def figure(res, tab, tb, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = dict(refit="C0", vilpellet="C1", slow="C2")
    lab = dict(refit="refit, 1 s loggers", vilpellet="Vilpellet, 1 s loggers", slow="refit, slower loggers")
    sets = dict(refit=tab["fast"]["refit"], vilpellet=tab["fast"]["vilpellet"], slow=tab["slow"]["refit"])
    r = dict(refit=res["fast"]["refit"], vilpellet=res["fast"]["vilpellet"], slow=res["slow"]["refit"])
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    a = ax[0, 0]
    for k, (d, _) in sets.items():
        a.hist(d.vz.clip(-3, 5), bins=np.linspace(-3, 5, 81), histtype="step", density=True, color=col[k], label=lab[k])
    a.axvline(0, color="k", lw=0.5)
    a.set(xlabel="mean vertical speed of a climb run (m/s, clipped)", ylabel="density", title="a  climb runs")
    a.legend(fontsize=7)
    a = ax[0, 1]
    w = 0.27
    for i, k in enumerate(sets):
        v = [r[k]["by_duration"].get(b, {}).get("share_runs_descent", np.nan) for b in BIN_NAMES]
        a.bar(np.arange(len(BIN_NAMES)) + (i - 1) * w, v, w, color=col[k], label=lab[k])
    a.set(xticks=range(len(BIN_NAMES)), xticklabels=BIN_NAMES, ylabel="share of climb runs with net descent", title="b  by duration")
    a.tick_params(axis="x", labelsize=7)
    a = ax[0, 2]
    classes = ["plains", "hills", "low mountains", "high mountains", "unknown", "not in the ground table"]
    for i, k in enumerate(sets):
        v = [r[k]["by_ground_class"].get(c, {}).get("share_runs_descent", np.nan) for c in classes]
        a.bar(np.arange(len(classes)) + (i - 1) * w, v, w, color=col[k])
    a.set(xticks=range(len(classes)), xticklabels=[c.replace(" ", "\n") for c in classes], title="c  by ground class")
    a.tick_params(axis="x", labelsize=6)
    a = ax[1, 0]
    for k, (_, al) in sets.items():
        a.hist(al.offset.clip(-60, 60), bins=np.arange(-60.5, 61.5), histtype="step", color=col[k], label=lab[k])
    a.set(yscale="log", xlabel="best time offset (s, clipped at +-60)", ylabel="flights", title="d  climb indicator vs v_z")
    a = ax[1, 1]
    g = res["fast"]["features"]["agreement_by_abs_offset"]
    a.bar(list(g), [v["median"] for v in g.values()], color="C3")
    for i, v in enumerate(g.values()):
        a.text(i, v["median"], f"n={v['flights']}", ha="center", va="bottom", fontsize=7)
    a.set(ylim=(0.8, 1.0), xlabel="|offset| of the refit labels", ylabel="median share of fixes, refit = Vilpellet", title="e  agreement vs offset")
    a = ax[1, 2]
    for k, c in (("fast", "C0"), ("slow", "C2")):
        a.hist(tb[k].clip(-10, 10), bins=np.arange(-10.5, 11.5), histtype="step", color=c, label=f"{k} loggers, n={len(tb[k])}")
    a.set(yscale="log", xlabel="shift of our track against the viewer's heights (s)", ylabel="flights", title="f  time base check")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["extract", "analyse"])
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--slow", action="store_true")
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    if a.step == "extract":
        extract(a.n, a.cache, a.slow)
        return
    res, tab, tb = {}, {}, {}
    for key, slow in (("fast", False), ("slow", True)):
        r, t, meta, fix = analyse_sample(a.cache, slow)
        refit_al = t["refit"][1]
        r["features"] = features(refit_al, meta, t.get("agree"))
        if not slow:
            r["features_vilpellet"] = features(t["vilpellet"][1], meta, t.get("agree"))
        r["examples"] = examples(*t["refit"])
        del fix
        res[key], tab[key] = r, t
        tb[key] = timebase(a.cache, meta, slow, 150 if not slow else 300)
        s = tb[key]
        r["timebase_check"] = dict(flights=len(s), share_zero=float((s == 0).mean()), share_abs_1=float((s.abs() == 1).mean()),
                                   share_abs_ge_2=float((s.abs() >= 2).mean()), examples_abs_ge_2=[f for f in s[s.abs() >= 2].index[:6]])
    nflights = {k: len(v["refit"][0].flight_id.unique()) for k, v in tab.items()}
    out = dict(seed=SEED, samples=dict(fast=dict(flights=3000, note="viewer flights logged at 1 s with refit and Vilpellet runs"),
                                       slow=dict(flights=2000, note="viewer flights logged slower, refit runs only (Vilpellet labels only 1-s flights)")),
               convention="offset s: the climb indicator at time t - s is best correlated with the viewer's v_z at t; s < 0 means the labels lead the track",
               **res)
    OUT.mkdir(exist_ok=True)
    (OUT / "climb_descent_check.json").write_text(json.dumps(out, indent=1, default=str))
    figure(res, tab, tb, OUT / "climb_descent_check.pdf")


if __name__ == "__main__":
    main()
