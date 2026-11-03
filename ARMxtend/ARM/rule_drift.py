# -*- coding: utf-8 -*-
"""
Rule drift: how association rules change over time. The rules are mined in successive windows of a time-ordered table
of items (sensor data, logs, a chunked stream) and each rule is followed from one window to another to know which are
stable, which emerge, which vanish and which appear and disappear (intermittent), and how the value of its measure
evolves.

It is the "batch" version of the trends that FIMoTS (``SARE``/``SFIM``) maintains incrementally over a Spark stream:
here the miner is any function ``miner(window) -> rules``, so crisp, fuzzy or temporal rules (``ARM.temporal_rules``)
all work, and it runs on a single machine.

States of a rule (``rule_drift``), with ``W`` windows:

* ``stable``: it is in the first and in the last window and in at least ``stable_fraction`` of all of them.
* ``emerging``: it is not in the first window and it is in the last.
* ``vanishing``: it is in the first window and not in the last.
* ``intermittent``: the rest (it appears and disappears, or it is only in the middle, or it is at the ends but with
  many gaps).

@author Carlos Fernandez-Basso
"""
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth, fuzzy_fpgrowth
from ..FIM._shared import key_to_itemset
from ..FIM.FARE import fuzzy_association_rules
from ..preprocessing.timeseries import to_fuzzy_transactions, to_transactions
from .association_rules import association_rules
from .meta_rules import rule_key

Miner = Callable[[pd.DataFrame], pd.DataFrame]
Width = Union[int, str, pd.Timedelta]

STATUSES = ("stable", "emerging", "vanishing", "intermittent")


def crisp_rule_miner(min_supp: float, metric: str = "confidence", min_threshold: float = 0.8) -> Miner:
    """Default miners of `window_rules`: FP-Growth + `ARM.association_rules` over boolean items."""
    def miner(window: pd.DataFrame) -> pd.DataFrame:
        frequent = fpgrowth(to_transactions(window), min_supp)
        itemsets = pd.DataFrame({"support": list(frequent.values()),
                                 "itemsets": [key_to_itemset(key) for key in frequent]})
        return association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    return miner


def fuzzy_rule_miner(min_supp: float, num_alpha: int = 10, metric: str = "confidence",
                     min_threshold: float = 0.8) -> Miner:
    """Fuzzy miner for `window_rules`: `FFIM.fuzzy_fpgrowth` + `FIM.FARE.fuzzy_association_rules` over membership degrees."""
    def miner(window: pd.DataFrame) -> pd.DataFrame:
        frequent = fuzzy_fpgrowth(to_fuzzy_transactions(window), min_supp, num_alpha)
        return fuzzy_association_rules(frequent, num_alpha, metric=metric, min_threshold=min_threshold)
    return miner


def _windows(index: pd.Index, width: Width, step: Optional[Width]) -> Iterator[Tuple[int, int]]:
    """Intervals [start, end) of rows of each window, by number of rows or by duration."""
    rows = len(index)
    if isinstance(width, (int, np.integer)):
        stride = int(width) if step is None else int(cast(int, step))
        if width < 1 or stride < 1:
            raise ValueError("width y step deben ser >= 1")
        for start in range(0, rows - width + 1, stride):
            yield start, start + width
        return

    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("a window by duration ('7D', '1h', ...) requires a sorted DatetimeIndex")
    span = pd.Timedelta(width)
    stride_time = span if step is None else pd.Timedelta(step)
    if span <= pd.Timedelta(0) or stride_time <= pd.Timedelta(0):
        raise ValueError("width y step deben ser duraciones positivas")
    cursor, last = index[0], index[-1]
    while cursor <= last:
        yield int(index.searchsorted(cursor, side="left")), int(index.searchsorted(cursor + span, side="left"))
        cursor += stride_time


def window_rules(data: pd.DataFrame, width: Width, step: Optional[Width] = None, min_supp: float = 0.1,
                 metric: str = "confidence", min_threshold: float = 0.8, miner: Optional[Miner] = None,
                 min_rows: int = 2) -> pd.DataFrame:
    """
    Mine the rules of each window of a time-ordered table of items.

    Args:
        data (pandas.DataFrame): boolean items (or membership degrees, with a fuzzy `miner`), one row per instant and
            ordered in time
        width (int | str | pandas.Timedelta): size of the window: a number of rows, or a duration ('7D', '12h') if the
            index is a ``DatetimeIndex``
        step (int | str | pandas.Timedelta, optional): displacement between windows, in the same units as `width`; by
            default equal to `width` (non-overlapping windows)
        min_supp, metric, min_threshold: parameters of the default miner (`crisp_rule_miner`), ignored if `miner` is
            given
        miner (Callable, optional): function ``window -> DataFrame of rules`` with columns 'antecedents' and
            'consequents'; e.g. `fuzzy_rule_miner(...)`, or a function of your own that calls
            ``ARM.temporal_rules``
        min_rows (int): windows with fewer rows (e.g. the last one, if it was left incomplete) are omitted and not
            numbered

    Returns:
        pandas.DataFrame with the rules of all the windows, plus the columns 'window' (0, 1, ...), 'start' and 'end'
        (first and last index of the window) and 'n_rows'. The number of windows, including those with no rules, is
        in ``result.attrs["n_windows"]``.
    """
    mine = miner if miner is not None else crisp_rule_miner(min_supp, metric, min_threshold)
    frames: List[pd.DataFrame] = []
    count = 0
    for lo, hi in _windows(data.index, width, step):
        if hi - lo < min_rows:
            continue
        window = data.iloc[lo:hi]
        rules = mine(window).copy()
        rules.insert(0, "window", count)
        rules.insert(1, "start", window.index[0])
        rules.insert(2, "end", window.index[-1])
        rules.insert(3, "n_rows", len(window))
        frames.append(rules)
        count += 1
    if not frames:
        raise ValueError("ninguna ventana tiene al menos {} filas".format(min_rows))
    result = pd.concat(frames, ignore_index=True)
    result.attrs["n_windows"] = count
    return result


def _key(rules: pd.DataFrame) -> pd.Series:
    return pd.Series([rule_key(a, c) for a, c in zip(rules["antecedents"], rules["consequents"])], index=rules.index)


def _status(present: np.ndarray, stable_fraction: float) -> str:
    first, last = bool(present[0]), bool(present[-1])
    if first and last:
        return "stable" if present.mean() >= stable_fraction else "intermittent"
    if last:
        return "emerging"
    if first:
        return "vanishing"
    return "intermittent"


def rule_drift(windowed_rules: pd.DataFrame, measure: str = "confidence", stable_fraction: float = 0.8,
               n_windows: Optional[int] = None) -> pd.DataFrame:
    """
    Summarize the evolution of each rule across the windows of `window_rules`.

    Args:
        windowed_rules (pandas.DataFrame): result of `window_rules`
        measure (str): column whose evolution is followed ('confidence', 'support', 'lift', ...)
        stable_fraction (float): minimum fraction of windows in which a rule must be present, being in the first and
            the last, to be 'stable'
        n_windows (int, optional): total number of windows; by default that of ``windowed_rules.attrs`` or, if lost,
            the largest index plus one

    Returns:
        pandas.DataFrame with one row per rule: 'antecedents', 'consequents', 'windows' (in how many it is present),
        'persistence' (that fraction of the total), 'first_window', 'last_window', 'mean_<measure>', 'min_<measure>',
        'max_<measure>', 'slope' (slope of the measure per window, NaN if the rule is in fewer than 3 windows) and
        'status' ('stable', 'emerging', 'vanishing' or 'intermittent', see the module docstring). Sorted by status
        and persistence.
    """
    if measure not in windowed_rules.columns:
        raise ValueError("measure '{}' is not a column of the rules: {}".format(
            measure, sorted(windowed_rules.columns)))
    total = n_windows or windowed_rules.attrs.get("n_windows") or int(windowed_rules["window"].max()) + 1
    table = windowed_rules.assign(_key=_key(windowed_rules))

    rows: List[dict] = []
    for _, group in table.groupby("_key", sort=False):
        windows = group["window"].to_numpy()
        values = group[measure].to_numpy(dtype=float)
        present = np.zeros(total, dtype=bool)
        present[windows] = True
        slope = float(np.polyfit(windows, values, 1)[0]) if len(windows) >= 3 else float("nan")
        rows.append({
            "antecedents": group["antecedents"].iloc[0], "consequents": group["consequents"].iloc[0],
            "windows": int(present.sum()), "persistence": float(present.mean()),
            "first_window": int(windows.min()), "last_window": int(windows.max()),
            "mean_" + measure: float(values.mean()), "min_" + measure: float(values.min()),
            "max_" + measure: float(values.max()), "slope": slope,
            "status": _status(present, stable_fraction),
        })
    columns = ["antecedents", "consequents", "windows", "persistence", "first_window", "last_window",
               "mean_" + measure, "min_" + measure, "max_" + measure, "slope", "status"]
    result = pd.DataFrame(rows, columns=columns)
    result["_order"] = result["status"].map({status: position for position, status in enumerate(STATUSES)})
    result = result.sort_values(["_order", "persistence"], ascending=[True, False], kind="mergesort")
    return result.drop(columns="_order").reset_index(drop=True)


def compare_rule_sets(before: pd.DataFrame, after: pd.DataFrame, measure: str = "confidence",
                      tolerance: float = 0.05) -> pd.DataFrame:
    """
    Compare two sets of rules (e.g. from two periods or two datasets).

    Args:
        before, after (pandas.DataFrame): rules with columns 'antecedents', 'consequents' and `measure`
        measure (str): measure to compare
        tolerance (float): a rule present in both sets is 'changed' if its measure changes, in absolute value, by more
            than this, and 'stable' if not

    Returns:
        pandas.DataFrame with one row per rule of either set: 'antecedents', 'consequents', '<measure>_before',
        '<measure>_after', 'delta' (after minus before) and 'status' ('new', 'lost', 'changed' or 'stable'), sorted:
        first the 'changed' (largest change first), then 'new', 'lost' and 'stable'.
    """
    for name, rules in (("before", before), ("after", after)):
        missing = [column for column in ("antecedents", "consequents", measure) if column not in rules.columns]
        if missing:
            raise ValueError("`{}` does not have the columns {}".format(name, missing))
    old = pd.Series(before[measure].to_numpy(dtype=float), index=_key(before).to_numpy())
    new = pd.Series(after[measure].to_numpy(dtype=float), index=_key(after).to_numpy())
    sides: Dict[str, Tuple[Any, Any]] = {}
    for rules, key in ((before, old.index), (after, new.index)):
        for antecedent, consequent, name in zip(rules["antecedents"], rules["consequents"], key):
            sides.setdefault(name, (antecedent, consequent))

    rows = []
    for name, (antecedent, consequent) in sides.items():
        was, now = old.get(name, np.nan), new.get(name, np.nan)
        if np.isnan(was):
            status = "new"
        elif np.isnan(now):
            status = "lost"
        else:
            status = "changed" if abs(now - was) > tolerance else "stable"
        rows.append({"antecedents": antecedent, "consequents": consequent, measure + "_before": was,
                     measure + "_after": now, "delta": now - was, "status": status})
    result = pd.DataFrame(rows, columns=["antecedents", "consequents", measure + "_before", measure + "_after",
                                         "delta", "status"])
    result["_order"] = result["status"].map({"changed": 0, "new": 1, "lost": 2, "stable": 3})
    result["_size"] = -result["delta"].abs().fillna(0.0)
    result = result.sort_values(["_order", "_size"], kind="mergesort")
    return result.drop(columns=["_order", "_size"]).reset_index(drop=True)
