"""The paragliders that fly open: the distance between the first and the last fix of the flight, its
takeoff and its landing, is more than OPEN_KM km. closure.py compares this with the type the pilot
declared.

Run any script of the analyses with --open to keep only these flights: tamsd.read_flights() drops the
others from the flight table, so the stretches, the pieces and the controls come from open flights only.
The groups are those of the run, whoever the companions; the isolation of a control still looks at
every flight of the archive. Caches go to <run>/analyses_open and results to out/<run>/open.
"""

import numpy as np
import pandas as pd

OPEN_KM = 6  # km: agrees with the declared open and closed flights in 91% of the paragliders


def open_flights(labels, ends):
    """The flight numbers (column flight of the flight table of the run `labels`) of the paragliders
    that fly open, from the first and last fixes in `ends` (closure.read_ends)."""
    flights = pd.read_parquet(labels / "flights.parquet", columns=["flight", "flight_id"]).astype({"flight_id": str})
    f = flights.merge(pd.read_parquet(ends), on="flight_id")
    return f.loc[np.hypot(f["e1"] - f["e0"], f["n1"] - f["n0"]) / 1000 > OPEN_KM, "flight"].to_numpy()
