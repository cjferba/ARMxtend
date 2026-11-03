# -*- coding: utf-8 -*-
"""
Interval-based QUANTITATIVE association rules (Srikant and Agrawal, 1996, "Mining quantitative association rules in
large relational tables"): rules over tables with numeric and categorical attributes, such as
``age in [30, 45) => income in [40000, 90000)`` and ``sex=F, age in [20, 30) => purchase=yes``.

Each numeric attribute is split into ``n_intervals`` equal-frequency (equi-depth) BASE intervals and the RANGES of
contiguous base intervals whose support does not exceed ``max_supp`` are formed (a range that is too wide is almost
always true and says nothing). The ranges form a hierarchy by inclusion -- ``[30, 40)`` is inside ``[30, 50)`` -- and
so everything reduces to the generalized rules of ``ARM.generalized_rules``, with a taxonomy whose "ancestors" of an
interval are the ranges that contain it, and with the restriction that a rule does not mention two ranges of the same
attribute. The R-interesting pruning of Srikant and Agrawal then removes the rules whose range can be widened or
narrowed without the rule really changing.

* ``IntervalPartition``: fits (``fit``) the intervals with some data and transforms (``transform``) rows -- also new
  ones -- into items.
* ``quantitative_rules``: table -> rules.
* ``partial_completeness_intervals``: how many intervals are needed so as not to lose more than a certain "partial
  completeness" K (formula of the paper).

Categorical items are ``column=value``; ranges are ``column=[low,high)`` (``]`` in the last interval; the minus sign is
written with U+2212 because items do not admit '-').

@author Carlos Fernandez-Basso
"""
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..FIM.taxonomy import MinSupp, Taxonomy
from .generalized_rules import generalized_rules

_EPS = 1e-9


def partial_completeness_intervals(n_quantitative: int, min_supp: float, completeness: float = 1.5) -> int:
    """
    Base intervals needed per attribute, ``2 n / (min_supp (K - 1))``, so that the formed ranges do not lose more than
    a "partial completeness" K (> 1) with respect to the best possible ones (Srikant and Agrawal, 1996).

    Args:
        n_quantitative (int): maximum number of numeric attributes in a rule
        min_supp (float): minimum support, in (0, 1]
        completeness (float): level K of partial completeness, > 1 (the closer to 1, the more intervals)
    """
    if n_quantitative < 1 or not 0 < min_supp <= 1 or completeness <= 1:
        raise ValueError("n_quantitative >= 1, min_supp in (0, 1] and completeness > 1 are required")
    return int(math.ceil(2 * n_quantitative / (min_supp * (completeness - 1))))


def _number(value: float) -> str:
    return "{:g}".format(value).replace("-", "−")


class IntervalPartition(object):
    """
    Partition of the attributes of a table into items: base intervals and ranges for the numeric ones,
    ``column=value`` for the others.

    Args:
        n_intervals (int): base intervals per numeric attribute (fewer if there are repeated values)
        method (str): 'quantile' (equal frequency, default) or 'uniform' (equal width)
        max_supp (float): maximum support of a range of several base intervals, in (0, 1]
        columns (list, optional): columns to use (all by default)
        quantitative (list, optional): columns treated as numeric (by default, the numeric ones with more than
            `n_intervals` distinct values; the others are categorical)

    Attributes after ``fit``: ``taxonomy_`` (hierarchy of ranges), ``groups_`` (range -> column, for ``exclusive``),
    ``intervals_`` (``DataFrame`` with item, column, low, high, support and n_base of each range), ``edges_`` (edges of
    each numeric column) and ``categorical_`` (categorical columns).
    """

    def __init__(self, n_intervals: int = 8, method: str = "quantile", max_supp: float = 0.5,
                 columns: Optional[Sequence[str]] = None, quantitative: Optional[Sequence[str]] = None):
        if n_intervals < 1:
            raise ValueError("n_intervals must be >= 1")
        if method not in ("quantile", "uniform"):
            raise ValueError("method must be 'quantile' or 'uniform', got: '{}'".format(method))
        if not 0 < max_supp <= 1:
            raise ValueError("max_supp must be in (0, 1]")
        self.n_intervals, self.method, self.max_supp = n_intervals, method, max_supp
        self.columns, self.quantitative = columns, quantitative

    def fit(self, table: pd.DataFrame) -> "IntervalPartition":
        columns = list(self.columns) if self.columns is not None else list(table.columns)
        missing = [name for name in columns if name not in table.columns]
        if missing:
            raise ValueError("nonexistent columns: {}".format(", ".join(map(str, missing))))
        if self.quantitative is not None:
            numeric = list(self.quantitative)
            wrong = [name for name in numeric if name not in columns or not pd.api.types.is_numeric_dtype(table[name])]
            if wrong:
                raise ValueError("the columns {} are not numeric or are not in `columns`".format(wrong))
        else:
            numeric = [name for name in columns if pd.api.types.is_numeric_dtype(table[name])
                       and not pd.api.types.is_bool_dtype(table[name]) and table[name].nunique() > self.n_intervals]
        self.categorical_: List[str] = [name for name in columns if name not in numeric]
        if not numeric and not self.categorical_:
            raise ValueError("there are no columns to use")
        total = len(table)
        self.edges_: Dict[str, np.ndarray] = {}
        rows: List[Dict[str, Any]] = []
        pairs: List[Any] = []
        for name in numeric:
            values = table[name].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                raise ValueError("column '{}' has no values".format(name))
            if self.method == "quantile":
                edges = np.unique(np.quantile(values, np.linspace(0, 1, self.n_intervals + 1)))
            else:
                edges = np.unique(np.linspace(values.min(), values.max(), self.n_intervals + 1))
            if len(edges) == 1:
                edges = np.array([edges[0], edges[0]])
            self.edges_[str(name)] = edges
            count = len(edges) - 1
            base = np.bincount(self._index(values, edges), minlength=count) / total
            cumulative = np.concatenate([[0.0], np.cumsum(base)])
            kept = {}
            for first in range(count):
                for last in range(first, count):
                    support = float(cumulative[last + 1] - cumulative[first])
                    if first == last or support <= self.max_supp + _EPS:
                        kept[(first, last)] = self._label(str(name), edges, first, last)
                        rows.append({"item": kept[(first, last)], "column": str(name), "low": float(edges[first]),
                                     "high": float(edges[last + 1]), "support": support, "n_base": last - first + 1})
            for (first, last), item in kept.items():
                for parent in ((first - 1, last), (first, last + 1)):
                    if parent in kept:
                        pairs.append((item, kept[parent]))
        self.intervals_ = pd.DataFrame(rows, columns=["item", "column", "low", "high", "support", "n_base"])
        self.taxonomy_ = Taxonomy(pairs)
        self.groups_: Dict[str, str] = dict(zip(self.intervals_["item"], self.intervals_["column"]))
        self._numeric = list(numeric)
        return self

    @staticmethod
    def _index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Index of the base interval of each value; values outside the fitted range fall into the first or the last one."""
        return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)

    @staticmethod
    def _label(name: str, edges: np.ndarray, first: int, last: int) -> str:
        closing = "]" if last == len(edges) - 2 else ")"
        return "{}=[{},{}{}".format(name, _number(edges[first]), _number(edges[last + 1]), closing)

    def transform(self, table: pd.DataFrame) -> List[List[str]]:
        """The base items (interval or ``column=value``) of each row; missing values give no item."""
        if not hasattr(self, "taxonomy_"):
            raise ValueError("fit must be called before transform")
        missing = [name for name in self._numeric + self.categorical_ if name not in table.columns]
        if missing:
            raise ValueError("missing the columns {}".format(missing))
        baskets: List[List[str]] = [[] for _ in range(len(table))]
        for name in self._numeric:
            edges = self.edges_[str(name)]
            values = table[name].to_numpy(dtype=float)
            present = ~np.isnan(values)
            index = np.full(len(values), -1)
            index[present] = self._index(values[present], edges)
            labels = [self._label(str(name), edges, k, k) for k in range(len(edges) - 1)]
            for row, k in enumerate(index):
                if k >= 0:
                    baskets[row].append(labels[k])
        for name in self.categorical_:
            for row, value in enumerate(table[name]):
                if not pd.isna(value):
                    baskets[row].append("{}={}".format(name, value))
        return baskets


def quantitative_rules(table: pd.DataFrame, min_supp: MinSupp = 0.1, n_intervals: int = 8, max_supp: float = 0.5,
                       columns: Optional[Sequence[str]] = None, quantitative: Optional[Sequence[str]] = None,
                       method: str = "quantile", metric: str = "confidence", min_threshold: float = 0.8,
                       max_len: Optional[int] = None, interest: Optional[float] = None,
                       partition: Optional[IntervalPartition] = None) -> pd.DataFrame:
    """
    Quantitative rules of a table (see the module docstring).

    Args:
        table (pandas.DataFrame): one row per record
        min_supp: minimum support (or ``{level: support}``, see ``ARM.generalized_rules``)
        n_intervals, max_supp, columns, quantitative, method: see ``IntervalPartition``
        metric, min_threshold, max_len: see ``ARM.generalized_rules``
        interest (float, optional): if given (>= 1), only the R-interesting rules of that R: those that exceed by that
            factor what is expected from the rule with wider ranges
        partition (IntervalPartition, optional): an already fitted partition (for example, with other data); if given,
            `n_intervals`, `max_supp`, `columns`, `quantitative` and `method` are ignored

    Returns:
        pandas.DataFrame with the format of ``ARM.generalized_rules``; ``attrs['partition']`` holds the partition and
        ``attrs['taxonomy']`` and ``attrs['supports']`` what ``interesting_rules`` needs
    """
    if partition is None:
        partition = IntervalPartition(n_intervals, method, max_supp, columns, quantitative).fit(table)
    baskets = partition.transform(table)
    rules = generalized_rules(baskets, partition.taxonomy_, min_supp, metric=metric, min_threshold=min_threshold,
                              max_len=max_len, interest=interest, exclusive=partition.groups_)
    rules.attrs["partition"] = partition
    return rules
