"""The flights of the analyses by wing level: paragliders in a group, and paragliders with no contact.

Level by the wing class of the FFVL catalog: experienced C, D or CIVL Competition Class (CCC),
beginners A or B. Group: flights with group time. Solo: flights with no contact of any length,
i.e. no group time and no row in the raw intervals of the run. The pieces, the equal weight per
flight and the bands are those of group-identification/tamsd.py, which this module puts on the path.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "group-identification"))
from tamsd import CACHE, COLORS, LABELS, SEED, read_pieces, save  # noqa: E402

OUT = Path(__file__).with_name("out") / LABELS.name
CATALOG = Path("/Volumes/SSD_DATA/xc-viewer-data/paragliders/catalog/catalog.csv")
LEVELS = {"experienced": {"C ou 2": "C", "D ou 2-3": "D", "CIVL Competition Class": "CCC"},
          "beginners": {"A ou 1": "A", "B ou 1-2": "B"}}  # level -> catalog class -> label
POPULATIONS = {"group": "group flights, group time", "solo": "flights with no contact"}


def wing_class(flights):
    """Catalog wing class of each paraglider flight, indexed by flight number."""
    ids = flights[flights["discipline"] == "para"].set_index("flight")["flight_id"]
    catalog = pd.read_csv(CATALOG, usecols=["flight_id", "wing_class"], dtype={"flight_id": str})
    return ids.map(catalog.set_index("flight_id")["wing_class"])


def alone(summary, raw):
    """Flights with no contact of any length."""
    touched = np.union1d(raw["a"].to_numpy(), raw["b"].to_numpy())
    return summary.loc[(summary["n_entries"] == 0) & ~summary["flight"].isin(touched), "flight"].to_numpy()


def keep(flight, population, classes, alone_flights, level):
    """Per row (piece or run, with its flight and population): whether it belongs to a flight of
    `level`, in group time or in a flight with no contact."""
    in_level = np.isin(flight, classes.index[classes.isin(LEVELS[level])])
    solo = (population == "solo") & np.isin(flight, alone_flights)
    return in_level & ((population == "group") | solo)


def load():
    """The flight summary, the wing classes and the flights with no contact of the run."""
    flights = pd.read_parquet(LABELS / "flights.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    return summary, wing_class(flights), alone(summary, pd.read_parquet(LABELS / "intervals_raw.parquet"))
