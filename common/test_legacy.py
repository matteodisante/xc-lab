from pathlib import Path

import pytest
import soaring

from common.legacy import UNCLASSIFIED, VILPELLET_STATES, clean_flight, segment

LAB = Path(__file__).resolve().parents[1]
SAMPLE = Path(
    "/Volumes/SSD_DATA/xc-viewer-data/paragliders/raw/igc/2021-2022/"
    "2021-09-01_20308851.igc"
)


def test_soaring_is_the_frozen_worktree():
    # soaring reads its YAML configs relative to its own checkout, so this is also
    # the check that the configs come from legacy-final.
    checkout = Path(soaring.__file__).resolve().parents[2]
    assert checkout == (LAB.parent / "soaring-legacy").resolve()


@pytest.mark.skipif(not SAMPLE.exists(), reason="SSD_DATA is not mounted")
def test_one_flight_is_cleaned_and_segmented():
    result = clean_flight(SAMPLE)
    assert result.kept, result.meta.drop_reason

    track = segment(result.fixes)
    phases = set(track.fixes["phase"])
    assert track.eligible
    assert phases <= {*VILPELLET_STATES, UNCLASSIFIED}
    assert "climb" in phases
