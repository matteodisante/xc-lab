"""Everything xc-lab borrows from the frozen ``soaring`` repository.

xc-dynamics does not yet have the Vilpellet segmenter or the preprocessing it expects,
so the lab takes both from ``soaring`` at tag ``legacy-final``, checked out read-only in
``../soaring-legacy``. This is the only module that imports ``soaring``: when a piece is
ported to xc-dynamics, its import changes here and the methods and analyses keep working.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd
from soaring.analysis import igc
from soaring.analysis.config import load_preproc_config
from soaring.analysis.derived import stream_flights
from soaring.analysis.preproc.enu import LocalFrame, geodetic_to_ecef
from soaring.analysis.preproc.pipeline import FlightResult, run_flight
from soaring.analysis.segmentation.vilpellet import (
    UNCLASSIFIED,
    VILPELLET_STATES,
    VilpelletTrack,
    load_vilpellet_config,
    segment_flight,
)
from soaring.reporting.disciplines import DISCIPLINES
from soaring.viewer.geodesy import enu_to_geodetic

__all__ = [
    "UNCLASSIFIED",
    "VILPELLET_STATES",
    "FlightResult",
    "LocalFrame",  # the origin of a cleaned flight's E, N: lat0, lon0, alt0 of flights_meta
    "VilpelletTrack",
    "clean_flight",
    "enu_to_geodetic",  # E, N, z of a cleaned flight back to latitude and longitude
    "geodetic_to_ecef",
    "segment",
    "stream_flights",  # one whole flight at a time from soaring's fixes.parquet on SSD_DISANTE
]

_preproc_config = cache(load_preproc_config)
_vilpellet_config = cache(load_vilpellet_config)


def clean_flight(igc_path: str | Path, discipline: str = "paragliders") -> FlightResult:
    """Parse one ``.igc`` file and run soaring's preprocessing on it.

    ``result.kept`` is False when a cleaning gate dropped the flight, and
    ``result.meta.drop_reason`` says why. The cleaned fixes are ``result.fixes``.
    """
    path = Path(igc_path)
    return run_flight(
        igc.parse_igc(path),
        _preproc_config(),
        source=DISCIPLINES[discipline].source,
        flight_id=path.stem,
        discipline=discipline,
    )


def segment(fixes: pd.DataFrame, discipline: str = "paragliders") -> VilpelletTrack:
    """Label every fix of one cleaned flight with Vilpellet's segmenter.

    The ``phase`` column of ``track.fixes`` holds one of ``VILPELLET_STATES``, or
    ``UNCLASSIFIED`` where the segment was too short to decode.
    """
    return segment_flight(fixes, _vilpellet_config(), discipline=discipline)
