"""Group or zone? Group and solo flights compared over every lag, flying over the same ground.

The difference between group time (G) and solo flights (S) could come from where groups fly: if
the zones where groups form also make solo pilots progress in a straight line, the two would
differ little once compared there. Strata, from coarse to fine, of the middle of each piece of
track (zone: cell of CELL_KM km, from the takeoff and the east and north of the piece):

    none             all flights pooled
    zone             the cell
    zone x month     the cell and the month of the flight
    zone x hour      the cell and the local hour (Europe/Paris) of the middle of the piece
    site-day x hour  the flights of one day taking off within 3 km, and the local hour

In every stratum with both populations the solo flights are reweighted to the mix of group flights
over the strata (direct standardisation): MSD_S|G = sum_s n_G,s MSD_S,s / sum_s n_G,s, where n_G,s
are the flights (units: flight and stratum) of G in s and MSD_S,s the ensemble MSD of the solo
units of s. The group is taken on the same strata, so both curves share the support. The same for
A, the solo time of the group flights. From each curve H_loc over every lag and the fitted H
(tamsd.fit_h, tamsd.local_h) and their differences with G. Bands: 5-95% of N_BOOT draws of blocks
that join the episodes and the site-days (decompose.block_counts), the same draws everywhere.

    uv run python zones.py [--run NAME]    # after tamsd.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import decompose
from blocks import with_takeoffs
from tamsd import CACHE, COLORS, LABELS, LAGS, OUT, SEED, fit_h, local_h, read_pieces, save

CELL_KM = 25
EARTH_KM = 6371.0
STRATA = {"none": None, "zone": "zone", "zone x month": "zone_month", "zone x hour": "zone_hour",
          "site-day x hour": "site_hour"}
CONTRASTS = {"S": ("group", "solo"), "A": ("group", "group_solo")}
RAMP = ("#9a9893", "#f7c59f", "#f2a065", "#d4561f", "#8c2f0f")  # grey, then orange, light to dark: finer strata


def zone_of(p, takeoff):
    """The cell of CELL_KM km, as an integer code, of the middle of every piece of `p` (dict with
    flight, mid_e, mid_n): the takeoff (lat0, lon0 by flight in `takeoff`) plus the east and north
    offsets, on a plane centred on France. The conversion is approximate (the east-west error is
    below 7 km at 300 km from the takeoff), enough for cells of this size."""
    t = takeoff.reindex(p["flight"])
    lat0, lon0 = t["lat0"].to_numpy(float), t["lon0"].to_numpy(float)
    lat = lat0 + np.degrees(p["mid_n"] / EARTH_KM / 1000)
    lon = lon0 + np.degrees(p["mid_e"] / EARTH_KM / 1000 / np.cos(np.radians(lat0)))
    x, y = (lon - 2.5) * 111.32 * np.cos(np.radians(46.5)), (lat - 46.5) * 111.2
    return (np.floor(x / CELL_KM).astype(np.int64) + 500) * 1000 + np.floor(y / CELL_KM).astype(np.int64) + 500


def with_strata(p, flights):
    """`p` (dict of arrays) with the columns zone, zone_month and zone_hour."""
    info = flights.set_index("flight")
    zone = zone_of(p, info[["lat0", "lon0"]])
    utc = info["utc0"].reindex(p["flight"]).to_numpy() + p["start"] + p["duration"] / 2
    month = pd.to_datetime(info["utc0"].reindex(p["flight"]).to_numpy(), unit="s", utc=True).month.to_numpy()
    return p | {"zone": zone, "zone_month": zone * 100 + month, "zone_hour": zone * 100 + decompose.local_hour(utc)}


def standardized(units, P, Q, counts):
    """On the strata with both populations: the ensemble MSD of P, that of Q reweighted to the mix of
    P, that of Q on the same strata without reweighting, and the share of the units of P kept, for
    the point estimate and each draw of `counts` (arrays draws x lags, draws x lags, draws x lags, draws)."""
    s = pd.factorize(units.stratum)[0]
    n_s = s.max() + 1
    isP, isQ = units.population == P, units.population == Q
    mp, mq, mo, share = [], [], [], []
    for d in range(len(counts) + 1):
        w = np.ones(len(s)) if d == 0 else counts[d - 1, units.block].astype(float)
        nP, nQ = (np.bincount(s[m], weights=w[m], minlength=n_s) for m in (isP, isQ))
        sP, sQ = (np.column_stack([np.bincount(s[m], weights=w[m] * v, minlength=n_s) for v in units.values[m].T])
                  for m in (isP, isQ))
        ok = (nP > 0) & (nQ > 0)
        with np.errstate(invalid="ignore", divide="ignore"):  # NaN curves where no stratum has both
            mp.append(sP[ok].sum(axis=0) / nP[ok].sum())
            mq.append(((nP[ok] / nQ[ok])[:, None] * sQ[ok]).sum(axis=0) / nP[ok].sum())
            mo.append(sQ[ok].sum(axis=0) / nQ[ok].sum())
        share.append(nP[ok].sum() / nP.sum())
    return np.array(mp), np.array(mq), np.array(mo), np.array(share)


def curve(msd, lags=LAGS):
    """H_loc at every lag and the fitted H of each row of `msd` (draws x lags)."""
    return np.array([local_h(lags, m) for m in msd]), np.array([fit_h(lags, m) for m in msd])


def band(draws):
    """Point estimate (first row) and 5-95% band (the other rows), per column."""
    return draws[0].tolist(), np.percentile(draws[1:], [5, 95], axis=0).tolist()


def compare(pieces, flights, info, block, counts):
    """For every stratification and contrast, G and the standardised other population: ensemble MSD,
    H_loc over every lag, fitted H, their differences with their bands, and the share of G kept."""
    p = with_strata(pieces, flights)
    out = {}
    for name, key in STRATA.items():
        units = decompose.msd_units(p, key, info, block)
        out[name] = {}
        for c, (P, Q) in CONTRASTS.items():
            mp, mq, mo, share = standardized(units, P, Q, counts)
            hp, fp = curve(mp)
            hq, fq = curve(mq)
            ho, fo = curve(mo)
            entry = {"share of G kept": band(share[:, None])[0][0], "lags": LAGS.tolist()}
            for label, h, f, m in (("G", hp, fp, mp), ("other", hq, fq, mq)):
                v, b = band(h)
                entry[label] = {"msd": m[0].tolist(), "H_loc": v, "H_loc_band": b, "H": f[0],
                                "H_band": np.percentile(f[1:], [5, 95]).tolist()}
            v, b = band(hp - hq)
            entry["dH_loc"], entry["dH_loc_band"] = v, b
            entry["dH"] = float((fp - fq)[0])
            entry["dH_band"] = np.percentile((fp - fq)[1:], [5, 95]).tolist()
            v, b = band(hp - ho)  # the same strata, the other population not reweighted
            entry["dH_loc_support"], entry["dH_loc_support_band"] = v, b
            entry["dH_support"] = float((fp - fo)[0])
            entry["dH_support_band"] = np.percentile((fp - fo)[1:], [5, 95]).tolist()
            out[name][c] = entry
        print(name, {c: round(out[name][c]["dH"], 4) for c in CONTRASTS}, "share kept",
              round(out[name]["S"]["share of G kept"], 3), flush=True)
    return out


def figure_curves(result, path):
    """H_loc(tau) of G and of the standardised S and A, one panel per stratification."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex=True, sharey=True)
    for ax, (name, r) in zip(axes.flat, result.items()):
        lags = np.array(r["S"]["lags"])
        for label, color, entry in (("G, group time", COLORS["group"], r["S"]["G"]),
                                    ("S, solo flights", COLORS["solo"], r["S"]["other"]),
                                    ("A, same flights alone", COLORS["group_solo"], r["A"]["other"])):
            lo, hi = entry["H_loc_band"]
            ax.fill_between(lags, lo, hi, color=color, alpha=0.2, lw=0)
            ax.plot(lags, entry["H_loc"], color=color, marker="o", ms=3, lw=1.2, label=label)
        ax.set_xscale("log")
        ax.set_title(f"{name}  (share of G kept: {100 * r['S']['share of G kept']:.0f}%)", fontsize=9)
        ax.axhline(1, color="#999999", lw=0.6, ls=":")
    axes.flat[-1].axis("off")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel(r"$\tau$ (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$H_{\rm loc}(\tau)$")
    fig.suptitle("Paragliders: H_loc over every lag, solo flights reweighted to the zones of the group "
                 "(5-95% bands)", fontsize=10)
    save(fig, path)


def figure_differences(result, path):
    """G minus the standardised S, and G minus the standardised A: H_loc(tau) differences over every lag."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, c, title in zip(axes, ("S", "A"), ("G minus S, same zones", "G minus A, same zones")):
        for (name, r), color in zip(result.items(), RAMP):
            lags = np.array(r[c]["lags"])
            lo, hi = r[c]["dH_loc_band"]
            ax.fill_between(lags, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(lags, r[c]["dH_loc"], color=color, marker="o", ms=3, lw=1.3, label=name)
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=title)
    axes[0].set_ylabel(r"$\Delta H_{\rm loc}(\tau)$")
    axes[0].legend(frameon=False, fontsize=7.5, title="compared within", title_fontsize=7.5)
    save(fig, path)


if __name__ == "__main__":
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    info, block, n_blocks = decompose.flight_table(flights, summary)
    pieces = read_pieces(CACHE / "pieces.npz")["para"]
    counts = decompose.block_counts(n_blocks, np.random.default_rng(SEED))
    result = compare(pieces, flights, info, block, counts)
    (OUT / "zones.json").write_text(json.dumps(result, indent=1))
    figure_curves(result, OUT / "zones_curves.pdf")
    figure_differences(result, OUT / "zones_differences.pdf")
