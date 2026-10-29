"""The paragliders that fly open.

Declared open: the pilot declared a distance flight on the FFVL catalogue (free distance or distance with 1,
2 or 3 points). Straight open: declared open, and the distance between the first and the last fix of the
flight, its takeoff and its landing, is at least MIN_OPENING of the path flown (flights_meta path_km). The
second keeps the flights that do not fold back on themselves; closure.py compares the declared type with
the track.

Run any script of the analyses with --declared-open or --open (straight open) to keep only these flights:
tamsd.read_flights() drops the others from the flight table, so the stretches, the pieces and the controls
come from them only. The groups are those of the run, whoever the companions; the isolation of a control
still looks at every flight of the archive.
"""

import numpy as np
import pandas as pd

DECLARED_OPEN = {"Dist libre", "Dist 1 pt", "Dist 2 pts", "Dist 3 pts"}
MIN_OPENING = 0.5  # takeoff to landing over the path flown


def opening(labels, ends, derived):
    """Per flight of the run `labels`: whether it is declared open and its opening, takeoff to landing (first
    and last fix, `ends` of closure.read_ends) over the path flown; `derived` is the folder of the fixes."""
    flights = pd.read_parquet(labels / "flights.parquet", columns=["flight", "flight_id"]).astype({"flight_id": str})
    catalog = pd.read_csv(derived.parent / "catalog/catalog.csv", usecols=["flight_id", "flight_type"], dtype={"flight_id": str})
    meta = pd.read_parquet(derived / "flights_meta.parquet", columns=["flight_id", "path_km"]).astype({"flight_id": str})
    f = flights.merge(pd.read_parquet(ends), on="flight_id").merge(catalog, on="flight_id").merge(meta, on="flight_id")
    ends_km = np.hypot(f["e1"] - f["e0"], f["n1"] - f["n0"]) / 1000
    return pd.DataFrame({"flight": f["flight"], "declared_open": f["flight_type"].isin(DECLARED_OPEN),
                         "opening": ends_km / f["path_km"]})


def open_flights(labels, ends, derived, straight):
    """The flight numbers of the declared open flights, and with `straight` only those with an opening of at
    least MIN_OPENING."""
    o = opening(labels, ends, derived)
    keep = o["declared_open"] & ((o["opening"] >= MIN_OPENING) if straight else True)
    return o.loc[keep, "flight"].to_numpy()
