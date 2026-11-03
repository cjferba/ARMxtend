# -*- coding: utf-8 -*-
"""
Application of already mined rules to a table of boolean items, shared by anomaly detection (``ARM.anomaly``),
prediction (``ARM.prediction``) and classification (``ARM.classification``).

Items can carry a lag (``item@lagk``, see ``preprocessing.timeseries``): ``x@lag2`` in row ``t`` is the value of ``x``
in row ``t - 2``. For the first rows, with no previous data, the lagged item is taken as absent.

@author Carlos Fernandez-Basso
"""
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from ..preprocessing.timeseries import split_lag


class ItemTable:
    """A table of boolean items with access to lagged items and to conjunctions of items."""

    def __init__(self, items: pd.DataFrame) -> None:
        values = items.to_numpy()
        if values.dtype != bool:
            if not np.isin(values, (0, 1)).all():
                raise ValueError("`items` must be a table of boolean (or 0/1) items; for fuzzy or numeric data first use "
                                 "preprocessing.timeseries.to_items")
            values = values.astype(bool)
        self.rows = len(items)
        self.values = values
        self.columns: Dict[str, int] = {str(name): position for position, name in enumerate(items.columns)}
        self._cache: Dict[str, np.ndarray] = {}

    def item(self, name: str) -> np.ndarray:
        """Boolean vector (one entry per row) with the value of item `name`, lagged if it carries a lag."""
        if name not in self._cache:
            base, lag = split_lag(name)
            if base not in self.columns:
                raise ValueError("item '{}' of the rules is not a column of `items`".format(base))
            column = self.values[:, self.columns[base]]
            if lag == 0:
                shifted = column
            else:
                shifted = np.zeros(self.rows, dtype=bool)
                if lag < self.rows:
                    shifted[lag:] = column[:self.rows - lag]
            self._cache[name] = shifted
        return self._cache[name]

    def holds(self, names: Iterable[str]) -> np.ndarray:
        """Rows where all the items of `names` hold."""
        result = np.ones(self.rows, dtype=bool)
        for name in names:
            result &= self.item(name)
        return result

    def evaluable(self, names: Iterable[str]) -> np.ndarray:
        """Rows with enough previous data to read all the items of `names` (lag <= row)."""
        reach = max([split_lag(name)[1] for name in names] or [0])
        return np.arange(self.rows) >= reach


def max_lag(itemsets: Iterable[Iterable[str]]) -> int:
    """The largest lag of the items of the given sets (0 if there are none)."""
    return max([split_lag(item)[1] for itemset in itemsets for item in itemset] or [0])


def require_columns(rules: pd.DataFrame, *names: str) -> None:
    missing = [name for name in names if name not in rules.columns]
    if missing:
        raise ValueError("a `rules` le faltan las columnas {}".format(missing))


def describe_rule(antecedent: Iterable[str], consequent: Iterable[str]) -> str:
    """Readable text of a rule: 'door (t-1), motion (t-1) => light (t)'."""
    from .temporal_rules import format_lagged_item

    def side(items: Iterable[str]) -> str:
        return ", ".join(sorted(format_lagged_item(item) for item in items))

    return "{} => {}".format(side(antecedent), side(consequent))
