"""The cleaned flights of the old repository (soaring-anomalous-transport cleaning) as items for flights.py.

fixes.parquet (derived/, per discipline) holds the cleaned fixes of every kept flight: time t (s), E, N, z in a
local east-north-up frame centred on (lat0, lon0, alt0) of flights_meta.parquet, and the segment of the fix.
`stream` yields the flights of chosen ids from a range of row groups, so a run can be cut into chunks;
`lambert` gives their positions in Lambert-93.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import flights as fl

DISK = Path("/Volumes/SSD_DISANTE")
FOLDERS = {"paragliders": "paragliders/ffvl_cfd_igc", "hang_gliders": "hang_gliders/delta_cfd_igc"}
COLUMNS = ["flight_id", "segment_id", "t", "E", "N", "z"]


def derived(discipline):
    return DISK / FOLDERS[discipline] / "derived"


def frames(discipline):
    """{flight_id: (lat0, lon0, alt0)} of the flights the cleaning kept."""
    m = pd.read_parquet(derived(discipline) / "flights_meta.parquet", columns=["flight_id", "lat0", "lon0", "alt0", "drop_reason"])
    m = m[m["drop_reason"].isna()].dropna(subset=["lat0", "lon0"]).drop_duplicates("flight_id").fillna({"alt0": 0.0})
    return dict(zip(m["flight_id"].astype(str), zip(m["lat0"], m["lon0"], m["alt0"])))


def n_row_groups(discipline):
    return pq.ParquetFile(derived(discipline) / "fixes.parquet").metadata.num_row_groups


def _flight(df, frame, discipline):
    return {"flight": str(df["flight_id"].iat[0]), "discipline": discipline, "t": df["t"].to_numpy(float),
            "seg": df["segment_id"].to_numpy(int), "E": df["E"].to_numpy(float), "N": df["N"].to_numpy(float),
            "z": df["z"].to_numpy(float), "frame": frame}


def stream(discipline, ids, a, b):
    """Flights of `ids` that start in the row groups a <= i < b of fixes.parquet, one dict each (see _flight).
    Rows of a flight are contiguous; a flight that crosses a boundary belongs to the chunk where it starts."""
    f = pq.ParquetFile(derived(discipline) / "fixes.parquet")
    n, wanted, frame = f.metadata.num_row_groups, pa.array(sorted(ids)), frames(discipline)
    skip = None  # a flight that started in the previous chunk: its rows at the start of row group a are not ours
    if a > 0:
        skip = f.read_row_group(a - 1, columns=["flight_id"])["flight_id"][-1].as_py()
    carry = None  # rows of the last flight of the previous row group, held until the flight ends
    for i in range(a, n):
        t = f.read_row_group(i, columns=COLUMNS)
        first, last = t["flight_id"][0].as_py(), t["flight_id"][-1].as_py()
        df = t.filter(pc.is_in(t["flight_id"], value_set=wanted)).to_pandas()
        df["flight_id"] = df["flight_id"].astype(str)
        if skip is not None:
            if first == skip:
                df = df[df["flight_id"] != skip]
            if last != skip or first != skip:  # the flight ended here
                skip = None
        cont = carry is not None and first == carry["flight_id"].iat[0]  # the held flight goes on in this row group
        if i >= b:  # past the chunk: only finish the flight in progress
            if carry is None:
                return
            if cont:
                carry = pd.concat([carry, df[df["flight_id"] == first]])
                if last == first:  # the flight fills the whole row group
                    continue
            yield from _emit([carry], frame, discipline)
            return
        if carry is not None:
            if cont:
                df = pd.concat([carry, df])
            else:
                yield from _emit([carry], frame, discipline)
            carry = None
        if len(df) and last in ids:
            keep = df["flight_id"] == last
            carry, df = df[keep], df[~keep]
        yield from _emit([g for _, g in df.groupby("flight_id", sort=False)], frame, discipline)
    if carry is not None:
        yield from _emit([carry], frame, discipline)


def _emit(groups, frame, discipline):
    for g in groups:
        fr = frame.get(str(g["flight_id"].iat[0]))
        if fr is not None and len(g) > 1:
            yield _flight(g, fr, discipline)


def lambert(item):
    """(n, 2) Lambert-93 positions of the fixes of a flight item."""
    from common.legacy import LocalFrame, enu_to_geodetic

    lat, lon = enu_to_geodetic(item["E"], item["N"], item["z"], LocalFrame(*item["frame"]))
    return np.column_stack(fl.lambert93(np.asarray(lat), np.asarray(lon)))
