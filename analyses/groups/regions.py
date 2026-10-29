"""Group, solo and the same flights alone in the zones where most flights have group time.

A zone is a cell of CELL_KM x CELL_KM km of zones.plane that holds the middle of the pieces, whichever the
takeoff. The N_ZONES cells with the most flights that have a piece of group time in them are kept. In each,
the pieces of tamsd.py (paragliders) give MSD and H_loc over the lags of tamsd.LAGS for solo flights, group
time and the same flights alone, with equal weight per flight and the bootstrap bands of tamsd.py. A zone
is named by the takeoff most flown by its group flights.

    uv run python regions.py [--run NAME]    # after tamsd.py; writes out/<run>/regions.json and regions.pdf
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curves
import tamsd
import zones
from blocks import with_takeoffs
from tamsd import CACHE, COLORS, LABELS, OUT, SEED

CELL_KM = 50
N_ZONES = 6
MIN_FLIGHTS = 20  # flights a population needs in a zone to be drawn
LABEL = {"solo": "S", "group": "G", "group_solo": "A"}


def centre(code):
    """Latitude, longitude and south-west corner (km on zones.plane) of the cell `code`."""
    x0, y0 = (code // 1000 - 500) * CELL_KM, (code % 1000 - 500) * CELL_KM
    x, y = x0 + CELL_KM / 2, y0 + CELL_KM / 2
    return 46.5 + y / 111.2, 2.5 + x / (111.32 * np.cos(np.radians(46.5))), x0, y0


def choose(p, code):
    """The N_ZONES cells with the most flights of group time, the busiest first."""
    group = p["population"] == "group"
    count = pd.Series(p["flight"][group]).groupby(code[group]).nunique()
    return count.sort_values(ascending=False, kind="stable").head(N_ZONES)


if __name__ == "__main__":
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    catalog = pd.read_csv(tamsd.FIXES["para"].parent.parent / "catalog/catalog.csv", usecols=["flight_id", "takeoff"], dtype=str).rename(columns={"takeoff": "site_name"})
    site = flights.astype({"flight_id": str}).merge(catalog, how="left", on="flight_id").set_index("flight")["site_name"]
    p = tamsd.read_pieces(CACHE / "pieces.npz")["para"]
    code = zones.zone_of(p, flights.set_index("flight")[["lat0", "lon0"]], CELL_KM)
    rng = np.random.default_rng(SEED)
    result = {"cell_km": CELL_KM, "min_flights": MIN_FLIGHTS, "zones": []}
    for cell, n_group in choose(p, code).items():
        here = code == cell
        group_here = here & (p["population"] == "group")
        names = site.reindex(p["flight"][group_here]).dropna()
        lat, lon, x0, y0 = centre(int(cell))
        pops = tamsd.summarize({k: v[here] for k, v in p.items()}, summary, rng)
        result["zones"].append({"code": int(cell), "name": names.mode().iat[0] if len(names) else "?", "lat": lat, "lon": lon,
                                "x0_km": x0, "y0_km": y0, "group flights": int(n_group), "populations": pops})
        print(result["zones"][-1]["name"], {k: v["flights"] for k, v in pops.items()})
    (OUT / "regions.json").write_text(json.dumps(result, indent=1))
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), constrained_layout=True)
    for ax, z in zip(axes.flat, result["zones"]):
        drawn = [curves.series(r, f"{LABEL[k]} ({r['flights']:,})", COLORS[k])
                 for k, r in z["populations"].items() if r["flights"] >= MIN_FLIGHTS]
        curves.plot(ax, drawn)
        ax.set_title(f"{z['name']}: {z['lat']:.1f}°N {z['lon']:.1f}°E", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.savefig(OUT / "regions.pdf", bbox_inches="tight")
    fig.savefig(OUT / "regions.png", dpi=200, bbox_inches="tight")
