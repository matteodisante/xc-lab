import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import openflights  # noqa: E402


def test_declared_open_types():
    kinds = ["Dist libre", "Dist 1 pt", "Dist 2 pts", "Dist 3 pts", "triangle", "triangle FAI", "Quadrilatère",
             "Aller-Retour", "Marche et Vol", None]
    assert openflights.declared(kinds).tolist() == [True] * 4 + [False] * 6


def test_opening_is_takeoff_to_landing_over_the_path():
    assert openflights.opening([5.0, 0.0, 3.0], [50.0, 10.0, 0.0])[:2].tolist() == [0.1, 0.0]
    assert np.isnan(openflights.opening([3.0], [0.0])[0])


def test_open_needs_a_declared_type_and_an_opening_above_the_minimum():
    assert openflights.MIN_OPENING == 0.1
    kinds = ["Dist libre", "Dist libre", "Dist 3 pts", "triangle", "Dist 2 pts"]
    ends = [5.0, 5.01, 20.0, 40.0, 1.0]
    path = [50.0, 50.0, 100.0, 50.0, 0.0]
    # 0.1 exactly is not open (strict), 0.1002 is, the triangle is not, a zero path is not
    assert openflights.is_open(kinds, ends, path).tolist() == [False, True, True, False, False]
