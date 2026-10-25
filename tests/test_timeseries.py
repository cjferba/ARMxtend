# -*- coding: utf-8 -*-
"""Tests de ``preprocessing.timeseries``: simbolizacion, items, calendario, retardos y ventanas."""
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ARMxtend.preprocessing import timeseries as ts


def test_lag_naming_roundtrip():
    assert ts.lagged("temp=high", 0) == "temp=high"
    assert ts.lagged("temp=high", 2) == "temp=high@lag2"
    assert ts.split_lag("temp=high@lag2") == ("temp=high", 2)
    assert ts.split_lag("temp=high") == ("temp=high", 0)
    assert ts.split_lag("door@lagx") == ("door@lagx", 0)          # sufijo no numerico: no es un retardo
    assert ts.item_variable("temp=high@lag2") == "temp"
    assert ts.item_variable("door_open") == "door_open"


def test_symbolize_quantile_balances_the_symbols():
    series = pd.Series(np.arange(1.0, 13.0))
    symbols = series.pipe(ts.symbolize, 3, "quantile")
    assert symbols.value_counts().to_dict() == {"low": 4, "medium": 4, "high": 4}
    assert symbols.iloc[0] == "low" and symbols.iloc[-1] == "high"


def test_symbolize_uniform_uses_equal_width_intervals():
    symbols = ts.symbolize([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, "uniform")
    # cortes en 3 y 6: [0, 3) low, [3, 6) medium, [6, 9] high
    assert symbols.tolist() == ["low"] * 3 + ["medium"] * 3 + ["high"] * 4


def test_symbolize_sax_uses_gaussian_breakpoints():
    rng = np.random.RandomState(0)
    data = rng.normal(5.0, 3.0, 20000)
    symbols = ts.symbolize(data, 3, "sax")
    share = symbols.value_counts(normalize=True)
    assert all(abs(share[label] - 1 / 3) < 0.02 for label in ("low", "medium", "high"))
    # la frontera esta en el cuantil 1/3 de la normal, sobre la serie normalizada
    cut = NormalDist().inv_cdf(1 / 3)
    z = (data - data.mean()) / data.std()
    assert (symbols[z < cut - 1e-9] == "low").all() and (symbols[z > cut + 1e-9] != "low").all()


def test_symbolize_keeps_missing_values_and_index():
    series = pd.Series([1.0, np.nan, 3.0, 2.0], index=list("abcd"), name="temp")
    symbols = ts.symbolize(series, 2, "uniform")
    assert symbols.name == "temp" and list(symbols.index) == list("abcd")
    assert symbols["b"] is None
    assert symbols.dropna().tolist() == ["low", "high", "high"]        # un valor sobre el corte pasa al simbolo alto


def test_symbolize_constant_series_gets_a_single_symbol():
    for method in ("quantile", "uniform", "sax"):
        assert ts.symbolize([4.0, 4.0, 4.0], 3, method).nunique() == 1


def test_symbolize_validation():
    with pytest.raises(ValueError, match="method"):
        ts.symbolize([1.0, 2.0], 2, "magic")
    with pytest.raises(ValueError, match="n_symbols"):
        ts.symbolize([1.0, 2.0], 1)
    with pytest.raises(ValueError, match="labels"):
        ts.symbolize([1.0, 2.0], 2, labels=["a"])
    with pytest.raises(ValueError, match="'-'"):
        ts.symbolize([1.0, 2.0], 2, labels=["very-low", "high"])


def test_discretize_only_touches_numeric_columns_by_default():
    table = pd.DataFrame({"temp": [1.0, 2.0, 3.0, 4.0], "room": list("abab"), "door": [True, False, True, False]})
    result = ts.discretize(table, n_symbols=2, method="uniform")
    assert result["temp"].tolist() == ["low", "low", "high", "high"]
    assert result["room"].tolist() == list("abab")
    assert result["door"].tolist() == [True, False, True, False]
    assert table["temp"].tolist() == [1.0, 2.0, 3.0, 4.0]          # no modifica la entrada


def test_to_items_categorical_boolean_and_missing():
    table = pd.DataFrame({"room": ["a", "b", None, "a"], "door": [True, False, False, True], "x-y": ["p-q", "p-q", "r", "r"]})
    items = ts.to_items(table)
    assert set(items.columns) == {"room=a", "room=b", "door", "x_y=p_q", "x_y=r"}
    assert items["room=a"].tolist() == [True, False, False, True]
    assert items.loc[2, ["room=a", "room=b"]].tolist() == [False, False]         # valor ausente: ningun item
    assert items["door"].tolist() == [True, False, False, True]
    assert all(dtype == bool for dtype in items.dtypes)
    assert not any("-" in name for name in items.columns)


def test_calendar_items_values():
    index = pd.DatetimeIndex(["2024-01-06 05:00", "2024-01-08 13:30", "2024-07-10 23:59"])   # sab, lun, mie
    items = ts.calendar_items(index, ["hour", "weekday", "daytype", "month", "part_of_day"])
    assert items.loc[index[0], ["hour=05", "weekday=Sat", "daytype=weekend", "month=Jan", "part_of_day=night"]].all()
    assert items.loc[index[1], ["hour=13", "weekday=Mon", "daytype=workday", "part_of_day=afternoon"]].all()
    assert items.loc[index[2], ["month=Jul", "part_of_day=evening", "weekday=Wed"]].all()
    assert items.sum(axis=1).tolist() == [5, 5, 5]
    assert "hour=00" not in items.columns                           # solo items que aparecen
    with pytest.raises(ValueError, match="calendario"):
        ts.calendar_items(index, ["century"])


def test_lag_items_values_columns_and_index():
    table = pd.DataFrame({"a": [True, False, True, True, False], "b": [False, False, True, False, True]},
                         index=pd.RangeIndex(10, 15))
    lagged = ts.lag_items(table, [0, 1, 2])
    assert list(lagged.columns) == ["a", "b", "a@lag1", "b@lag1", "a@lag2", "b@lag2"]
    assert list(lagged.index) == [12, 13, 14]
    assert lagged["a"].tolist() == [True, True, False]
    assert lagged["a@lag1"].tolist() == [False, True, True]
    assert lagged["a@lag2"].tolist() == [True, False, True]
    assert lagged["b@lag2"].tolist() == [False, False, True]
    assert lagged.dtypes.eq(bool).all()


def test_lag_items_without_dropping_pads_with_absence():
    table = pd.DataFrame({"a": [True, True, True]})
    lagged = ts.lag_items(table, [0, 2], drop_incomplete=False)
    assert lagged["a"].tolist() == [True, True, True]
    assert lagged["a@lag2"].tolist() == [False, False, True]
    assert ts.lag_items(table, [0, 5], drop_incomplete=False)["a@lag5"].tolist() == [False, False, False]


def test_lag_items_keeps_membership_degrees_as_floats():
    table = pd.DataFrame({"hot": [0.2, 0.9, 0.5, 1.0]})
    lagged = ts.lag_items(table, [0, 1])
    assert lagged["hot@lag1"].tolist() == [0.2, 0.9, 0.5]
    assert lagged["hot"].tolist() == [0.9, 0.5, 1.0]


def test_lag_items_validation():
    table = pd.DataFrame({"a": [True, False, True]})
    with pytest.raises(ValueError, match=">= 0"):
        ts.lag_items(table, [-1])
    with pytest.raises(ValueError, match="mas de"):
        ts.lag_items(table, [0, 3])
    with pytest.raises(ValueError, match="categoricas"):
        ts.lag_items(pd.DataFrame({"a": ["x", "y", "z"]}), [0, 1])


def test_window_items_any_all_mean():
    table = pd.DataFrame({"a": [True, False, False, True, False, False], "b": [True, True, True, False, True, True]},
                         index=pd.date_range("2024-01-01", periods=6, freq="h"))
    anyWindow = ts.window_items(table, 3, 1, "any")
    assert len(anyWindow) == 4 and anyWindow.index[0] == table.index[2]
    assert anyWindow["a"].tolist() == [True, True, True, True]
    allWindow = ts.window_items(table, 3, 1, "all")
    assert allWindow["b"].tolist() == [True, False, False, False]
    meanWindow = ts.window_items(table, 3, 3, "mean")
    assert len(meanWindow) == 2 and meanWindow.index.tolist() == [table.index[2], table.index[5]]
    assert meanWindow["a"].tolist() == pytest.approx([1 / 3, 1 / 3])
    with pytest.raises(ValueError, match="how"):
        ts.window_items(table, 3, 1, "median")
    with pytest.raises(ValueError, match="width"):
        ts.window_items(table, 10)


def test_window_items_on_membership_degrees_uses_max_and_min():
    table = pd.DataFrame({"hot": [0.1, 0.8, 0.3, 0.6]})
    assert ts.window_items(table, 2, 1, "any")["hot"].tolist() == [0.8, 0.8, 0.6]
    assert ts.window_items(table, 2, 1, "all")["hot"].tolist() == [0.1, 0.3, 0.3]


def test_transactions_conversion():
    items = pd.DataFrame({"a": [True, False], "b": [True, True]})
    assert ts.to_transactions(items) == [{"a", "b"}, {"b"}]
    degrees = pd.DataFrame({"a": [0.0, 0.5], "b": [1.0, 0.25]})
    assert ts.to_fuzzy_transactions(degrees) == [[("b", 1.0)], [("a", 0.5), ("b", 0.25)]]
    assert ts.to_fuzzy_transactions(degrees, min_degree=0.3) == [[("b", 1.0)], [("a", 0.5)]]


def test_check_item_names():
    ts.check_item_names(["a", "b=c"])
    with pytest.raises(ValueError, match="'-'"):
        ts.check_item_names(["ok", "bad-name"])
