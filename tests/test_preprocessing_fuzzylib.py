# -*- coding: utf-8 -*-
"""Regression tests for ARMxtend.preprocessing.FuzzyLib."""
import pandas as pd
import pytest

from ARMxtend.preprocessing import FuzzyLib


def test_fuzzification_triangular_partition_sums_to_one():
    fuzzyLib = FuzzyLib()
    fuzzyLib.data = pd.DataFrame({"Temperature": [15, 19, 20, 22, 25, 28]})

    addedColumns = fuzzyLib.Fuzzification(["Temperature"], [[18, 22, 26]], [["cold", "comfort", "warm"]])

    assert addedColumns == ["Temperature_cold", "Temperature_comfort", "Temperature_warm"]
    assert fuzzyLib.data[addedColumns].sum(axis=1).round(9).tolist() == [1.0] * 6
    # below the first peak, full membership in the first set (shoulder)
    assert fuzzyLib.data.loc[0, "Temperature_cold"] == pytest.approx(1.0)
    # exactly at a peak, full membership in its set
    assert fuzzyLib.data.loc[3, "Temperature_comfort"] == pytest.approx(1.0)
    assert fuzzyLib.data.loc[4, "Temperature_warm"] == pytest.approx(0.75)


def test_fuzzification_rejects_mismatched_lengths():
    fuzzyLib = FuzzyLib()
    fuzzyLib.data = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        fuzzyLib.Fuzzification(["x"], [[1, 2, 3]], [["a", "b"]])
