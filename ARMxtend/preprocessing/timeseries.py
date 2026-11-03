# -*- coding: utf-8 -*-
"""
Preparation of time series and sensor data for association rule mining: it converts numeric or categorical columns
ordered in time into ITEMS (discrete symbols, events, calendar variables), and builds the lagged items ("lag") on
which ``ARM.temporal_rules`` and ``ARM.rule_drift`` work.

Conventions:

* A set of items is a ``pandas.DataFrame`` with one row per instant and one column per item. If it is boolean, the
  item is present or absent; if it is numeric in [0, 1], they are membership degrees (fuzzy items, e.g. those of
  ``preprocessing.FuzzyLib``).
* Categorical items are named ``column=value`` (``temp=high``).
* The item ``x`` seen ``k`` steps earlier is called ``x@lagk`` (``temp=high@lag2``); the current instant (lag 0) keeps
  the name without suffix.

NOTE: item names must not contain '-', which is the item separator in the itemset keys of ``FIM._shared``;
``to_items`` replaces it by '_'.

The symbolization ``method="sax"`` follows the idea of SAX (Lin, Keogh, Wei & Lonardi, 2007, "Experiencing SAX: a
novel symbolic representation of time series", Data Mining and Knowledge Discovery 15): normalize the series (mean 0,
deviation 1) and split it with the cut points that leave the same probability to each symbol under a normal. Here it
is applied point by point, without the piecewise aggregation (PAA) of the original paper.

@author Carlos Fernandez-Basso
"""
from statistics import NormalDist
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .membership import default_labels

LAG_MARK = "@lag"
VALUE_SEP = "="

_SYMBOLIZE_METHODS = ("quantile", "uniform", "sax")
_CALENDAR_PARTS = ("hour", "weekday", "daytype", "month", "part_of_day")
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def lagged(item: str, lag: int) -> str:
    """Name of the item `item` seen `lag` steps earlier ('temp=high', 2 -> 'temp=high@lag2'; lag 0 does not change the name)."""
    return item if lag == 0 else "{}{}{}".format(item, LAG_MARK, lag)


def split_lag(item: str) -> Tuple[str, int]:
    """Inverse of `lagged`: 'temp=high@lag2' -> ('temp=high', 2); an item without suffix has lag 0."""
    base, mark, lag = item.rpartition(LAG_MARK)
    if mark and lag.isdigit():
        return base, int(lag)
    return item, 0


def item_variable(item: str) -> str:
    """Variable of an item, without lag or value: 'temp=high@lag2' -> 'temp'; 'door_open' -> 'door_open'."""
    return split_lag(item)[0].split(VALUE_SEP, 1)[0]


def _symbol_labels(n_symbols: int, labels: Optional[Sequence[str]]) -> List[str]:
    if n_symbols < 2:
        raise ValueError("n_symbols debe ser >= 2, recibido: {}".format(n_symbols))
    names = list(labels) if labels is not None else default_labels(n_symbols)
    if len(names) != n_symbols:
        raise ValueError("labels debe tener {} etiquetas, recibidas: {}".format(n_symbols, len(names)))
    for name in names:
        if "-" in name or VALUE_SEP in name:
            raise ValueError("labels cannot contain '-' or '{}': '{}'".format(VALUE_SEP, name))
    return names


def _breakpoints(values: np.ndarray, n_symbols: int, method: str) -> np.ndarray:
    if method == "uniform":
        return np.linspace(values.min(), values.max(), n_symbols + 1)[1:-1]
    if method == "quantile":
        return np.quantile(values, np.linspace(0.0, 1.0, n_symbols + 1)[1:-1])
    normal = NormalDist()
    return np.array([normal.inv_cdf(i / n_symbols) for i in range(1, n_symbols)])


def symbolize(values: Union[pd.Series, Sequence[float], np.ndarray], n_symbols: int = 3, method: str = "quantile",
              labels: Optional[Sequence[str]] = None) -> pd.Series:
    """
    Convert a numeric series into discrete symbols ('low', 'medium', 'high', ...).

    Args:
        values: numeric series; missing values (NaN) stay missing
        n_symbols (int): number of symbols, >= 2
        method (str): how the cut points are computed:
            'quantile' (default) -- symbols with the same frequency;
            'uniform' -- intervals of equal width between the minimum and the maximum;
            'sax' -- normalizes the series (mean 0, deviation 1) and uses the quantiles of the normal (see the module
            docstring)
        labels (Sequence[str], optional): names of the symbols, from lowest to highest; by default those of
            ``preprocessing.membership.default_labels``

    Returns:
        pandas.Series of labels (dtype object), with the same index and name as the input. With 'quantile' and many
        ties there can be fewer than `n_symbols` distinct symbols, and a constant series gets a single symbol.
    """
    if method not in _SYMBOLIZE_METHODS:
        raise ValueError("method debe ser uno de {}, recibido: '{}'".format(_SYMBOLIZE_METHODS, method))
    names = _symbol_labels(n_symbols, labels)
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values, dtype=float))
    data = series.to_numpy(dtype=float)
    valid = ~np.isnan(data)

    symbols = np.full(len(data), None, dtype=object)
    if valid.any():
        column = data[valid]
        if method == "sax":
            deviation = column.std()
            column = (column - column.mean()) / deviation if deviation > 0 else np.zeros_like(column)
        codes = np.searchsorted(_breakpoints(column, n_symbols, method), column, side="right")
        symbols[valid] = [names[code] for code in codes]
    return pd.Series(symbols, index=series.index, name=series.name, dtype=object)


def discretize(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, n_symbols: int = 3,
               method: str = "quantile", labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Apply `symbolize` to several numeric columns, leaving the others untouched.

    Args:
        df (pandas.DataFrame): table with one row per instant
        columns (Iterable[str], optional): columns to symbolize; by default, all the non-boolean numeric ones
        n_symbols, method, labels: see `symbolize`

    Returns:
        pandas.DataFrame, copy of `df` with the chosen columns converted into symbols
    """
    if columns is None:
        columns = [name for name in df.columns
                   if pd.api.types.is_numeric_dtype(df[name]) and not pd.api.types.is_bool_dtype(df[name])]
    result = df.copy()
    for name in columns:
        result[name] = symbolize(df[name], n_symbols=n_symbols, method=method, labels=labels)
    return result


def to_items(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, sep: str = VALUE_SEP) -> pd.DataFrame:
    """
    Convert categorical (or boolean) columns into boolean items.

    * A categorical column produces one item ``column=value`` for each value that appears (missing values produce no
      item).
    * A boolean column produces a single item with the name of the column, present when it is True (sensor events:
      'door_open', 'motion', ...).

    The '-' in names are replaced by '_' (see the module docstring).

    Args:
        df (pandas.DataFrame): table with one row per instant
        columns (Iterable[str], optional): columns to convert; by default, all
        sep (str): separator between column and value

    Returns:
        boolean pandas.DataFrame with the same index as `df`
    """
    names = list(df.columns) if columns is None else list(columns)
    items = {}
    for name in names:
        column = df[name]
        if pd.api.types.is_bool_dtype(column):
            items[str(name).replace("-", "_")] = column.to_numpy(dtype=bool)
            continue
        present = column.notna().to_numpy()
        text = column.astype(str).to_numpy()
        for value in sorted(set(text[present])):
            label = "{}{}{}".format(name, sep, value).replace("-", "_")
            items[label] = present & (text == value)
    return pd.DataFrame(items, index=df.index, dtype=bool)


def calendar_items(index: Union[pd.DatetimeIndex, Sequence[Any]],
                   parts: Sequence[str] = ("hour", "weekday")) -> pd.DataFrame:
    """
    Calendar items of each instant, to discover rules that only hold at certain hours, days or months.

    Args:
        index: instants (``pandas.DatetimeIndex`` or something convertible)
        parts (Sequence[str]): which parts of the calendar to include: 'hour' (``hour=08``), 'weekday'
            (``weekday=Mon``), 'daytype' (``daytype=weekend`` / ``daytype=workday``), 'month' (``month=Jan``) and
            'part_of_day' (``part_of_day=morning``: night 0-5, morning 6-11, afternoon 12-17, evening 18-23)

    Returns:
        boolean pandas.DataFrame indexed by `index`; only the items that appear at some instant are included
    """
    unknown = [part for part in parts if part not in _CALENDAR_PARTS]
    if unknown:
        raise ValueError("partes de calendario desconocidas {}; validas: {}".format(unknown, _CALENDAR_PARTS))
    stamps = index if isinstance(index, pd.DatetimeIndex) else pd.DatetimeIndex(list(index))

    labels = {
        "hour": ["hour={:02d}".format(hour) for hour in stamps.hour],
        "weekday": ["weekday=" + _WEEKDAYS[day] for day in stamps.day_of_week],
        "daytype": ["daytype=" + ("weekend" if day >= 5 else "workday") for day in stamps.day_of_week],
        "month": ["month=" + _MONTHS[month - 1] for month in stamps.month],
        "part_of_day": ["part_of_day=" + ("night", "morning", "afternoon", "evening")[hour // 6]
                        for hour in stamps.hour],
    }
    columns = {}
    for part in parts:
        values = np.array(labels[part], dtype=object)
        for value in sorted(set(values)):
            columns[value] = values == value
    return pd.DataFrame(columns, index=stamps, dtype=bool)


def _as_array(df: pd.DataFrame) -> np.ndarray:
    values = df.to_numpy()
    if values.dtype == object:
        try:
            values = values.astype(float)       # columnas booleanas y numericas mezcladas
        except (TypeError, ValueError):
            raise ValueError("the items must be boolean or numeric (membership degrees); "
                             "use to_items(...) to convert categorical columns") from None
    return values


def lag_items(df: pd.DataFrame, lags: Iterable[int], drop_incomplete: bool = True) -> pd.DataFrame:
    """
    Add to each instant the items of the `k` previous instants, for each lag `k` of `lags` (the item ``x`` seen ``k``
    steps earlier is called ``x@lagk``).

    Args:
        df (pandas.DataFrame): boolean items, or membership degrees in [0, 1], with one row per instant (ordered in
            time)
        lags (Iterable[int]): lags, >= 0; 0 keeps the current items
        drop_incomplete (bool): if True (default), drop the first ``max(lags)`` rows, for which there is no previous
            data, so that all the returned rows have all the lags; if False, they are kept and whatever would fall
            before the start is taken as absent

    Returns:
        pandas.DataFrame with one column per (lag, item), sorted by lag and then by original column; boolean if `df`
        is, and floats otherwise
    """
    requested = list(lags)
    steps = sorted({int(lag) for lag in requested})
    if not steps or steps[0] < 0:
        raise ValueError("lags debe contener enteros >= 0, recibido: {}".format(requested))
    values = _as_array(df)
    rows = len(values)
    if drop_incomplete and rows <= steps[-1]:
        raise ValueError("more than {} rows are needed to use lags up to {}; there are {}".format(
            steps[-1], steps[-1], rows))

    zero = np.zeros((1,) + values.shape[1:], dtype=values.dtype)
    start = steps[-1] if drop_incomplete else 0
    blocks = []
    for lag in steps:
        if drop_incomplete:
            block = values[start - lag:rows - lag]
        else:
            padding = np.repeat(zero, min(lag, rows), axis=0)
            block = np.concatenate([padding, values[:max(rows - lag, 0)]]) if lag else values
        blocks.append(block)
    columns = [lagged(str(name), lag) for lag in steps for name in df.columns]
    return pd.DataFrame(np.concatenate(blocks, axis=1), index=df.index[start:], columns=columns)


def window_items(df: pd.DataFrame, width: int, step: int = 1, how: str = "any") -> pd.DataFrame:
    """
    Summarize each window of `width` consecutive instants into a single row of items, to mine co-occurrences "within a
    window" (episodes) without fixing the order of the events.

    Args:
        df (pandas.DataFrame): boolean items or membership degrees
        width (int): number of rows of each window
        step (int): displacement between consecutive windows (1 = sliding window)
        how (str): 'any' -- the item is present if it appears at some instant of the window (the maximum of its
            degrees, if fuzzy); 'all' -- if it appears at all of them (the minimum); 'mean' -- fraction of the window
            in which it is present (returns floats)

    Returns:
        pandas.DataFrame with one row per window, indexed by the last instant of the window
    """
    if how not in ("any", "all", "mean"):
        raise ValueError("how debe ser 'any', 'all' o 'mean', recibido: '{}'".format(how))
    if width < 1 or step < 1:
        raise ValueError("width y step deben ser >= 1")
    values = _as_array(df)
    if len(values) < width:
        raise ValueError("the table has {} rows, fewer than width={}".format(len(values), width))
    windows = sliding_window_view(values, width, axis=0)[::step]
    if how == "any":
        reduced = windows.max(axis=-1)
    elif how == "all":
        reduced = windows.min(axis=-1)
    else:
        reduced = windows.mean(axis=-1)
    return pd.DataFrame(reduced, index=df.index[width - 1::step][:len(windows)], columns=df.columns)


def to_transactions(items: pd.DataFrame) -> List[Set[str]]:
    """Crisp transactions (sets of present items) of a boolean DataFrame, one per row."""
    names = np.array(items.columns, dtype=object)
    values = _as_array(items).astype(bool)
    return [set(names[row]) for row in values]


def to_fuzzy_transactions(memberships: pd.DataFrame, min_degree: float = 0.0) -> List[List[Tuple[str, float]]]:
    """
    Fuzzy transactions, one per row: list of (item, degree) pairs with degree strictly greater than `min_degree`. It
    is the input format of ``FFIM.fuzzy_fpgrowth``.
    """
    names = list(memberships.columns)
    degrees = _as_array(memberships).astype(float)
    return [[(names[column], float(row[column])) for column in np.flatnonzero(row > min_degree)] for row in degrees]


def check_item_names(names: Iterable[str]) -> None:
    """Raise ValueError if any item name contains '-' (the item separator of ``FIM._shared``)."""
    bad = [name for name in names if "-" in str(name)]
    if bad:
        raise ValueError("item names cannot contain '-' (e.g. {!r}); "
                         "use to_items(...) or rename the columns".format(bad[0]))
