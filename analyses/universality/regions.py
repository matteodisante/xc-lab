"""The four regions of the comparison, two lowland and two mountain.

The boxes (lon west, lon east, lat south, lat north, degrees; bounds included) are those of REGIONAL_BOXES of the
earlier project (soaring-anomalous-transport, soaring/analysis/regions.py), copied here. As there, a flight belongs
to a region when its takeoff, the first fix (lat0, lon0 of the frame of the flight), lies in the box and its
terrain class (terrain.py, the median ground elevation under its points) is one of the classes of the region:
plains or hills for the lowland regions, low or high mountains for the mountain ones. The boxes do not overlap.
"""

import numpy as np

BOXES = {"Alps": (5.4, 10.0, 43.8, 46.6), "Pyrenees": (-1.9, 3.3, 42.0, 43.5),
         "Channel Coast": (-1.8, 2.0, 48.3, 51.2), "Champagne-Lorraine": (2.1, 6.2, 48.0, 50.6)}
CLASSES = {"Alps": ("low mountains", "high mountains"), "Pyrenees": ("low mountains", "high mountains"),
           "Channel Coast": ("plains", "hills"), "Champagne-Lorraine": ("plains", "hills")}
NONE = ""


def region_of(lat, lon, cls):
    """The region of each flight from its takeoff latitude and longitude and its terrain class; NONE outside."""
    lat, lon, cls = np.asarray(lat, float), np.asarray(lon, float), np.asarray(cls, dtype=object)
    out = np.full(lat.shape, NONE, dtype=object)
    for name, (west, east, south, north) in BOXES.items():
        out[(west <= lon) & (lon <= east) & (south <= lat) & (lat <= north) & np.isin(cls, CLASSES[name])] = name
    return out
