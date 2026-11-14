"""The flights that fly open, for both disciplines.

Declared open: the pilot declared a distance flight on the FFVL catalogue (DECLARED_OPEN: free distance or
distance with 1, 2 or 3 points). Opening: the distance between the first and the last fix of the flight (its
takeoff and its landing) over the path flown (flights_meta path_km). Open: declared open and an opening of more
than MIN_OPENING, which drops the declared open flights that come back to their takeoff.
"""

import numpy as np

DECLARED_OPEN = {"Dist libre", "Dist 1 pt", "Dist 2 pts", "Dist 3 pts"}
MIN_OPENING = 0.1


def opening(ends_km, path_km):
    """Takeoff to landing over the path flown; NaN where the path is not positive."""
    ends_km, path_km = np.asarray(ends_km, float), np.asarray(path_km, float)
    return np.where(path_km > 0, ends_km / np.where(path_km > 0, path_km, 1), np.nan)


def declared(flight_type):
    """Whether the catalogue type is a declared distance flight."""
    return np.isin(np.asarray(flight_type, dtype=object), list(DECLARED_OPEN))


def is_open(flight_type, ends_km, path_km):
    """Declared open and an opening of more than MIN_OPENING."""
    return declared(flight_type) & (opening(ends_km, path_km) > MIN_OPENING)
