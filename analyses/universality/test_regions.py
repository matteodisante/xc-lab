import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import regions  # noqa: E402


def test_boxes_are_those_of_the_earlier_project():
    assert regions.BOXES == {"Alps": (5.4, 10.0, 43.8, 46.6), "Pyrenees": (-1.9, 3.3, 42.0, 43.5),
                             "Channel Coast": (-1.8, 2.0, 48.3, 51.2), "Champagne-Lorraine": (2.1, 6.2, 48.0, 50.6)}


def test_a_flight_needs_the_box_and_the_class():
    lat = [45.0, 45.0, 42.5, 49.5, 49.5, 49.0, 49.0, 45.0, 30.0]
    lon = [6.5, 6.5, 0.5, 0.0, 0.0, 4.0, 4.0, 3.0, 0.0]
    cls = ["high mountains", "plains", "low mountains", "plains", "high mountains", "hills", "unknown", "low mountains",
           "hills"]
    assert list(regions.region_of(lat, lon, cls)) == ["Alps", "", "Pyrenees", "Channel Coast", "", "Champagne-Lorraine", "", "",
                                                      ""]


def test_box_edges_are_included_and_the_boxes_do_not_overlap():
    assert regions.region_of([43.8, 46.6], [5.4, 10.0], ["low mountains"] * 2).tolist() == ["Alps", "Alps"]
    assert regions.region_of([48.3, 48.3], [2.0, 2.1], ["hills"] * 2).tolist() == ["Channel Coast", "Champagne-Lorraine"]
    boxes = list(regions.BOXES.values())
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]
