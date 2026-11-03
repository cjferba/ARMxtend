# -*- coding: utf-8 -*-
"""
FUZZY sequential rules: ``X => Y`` ("if all the items of X occur, then all those of Y occur afterwards") over
sequences of events with a membership degree (see ``FIM.fuzzy_sequential``). It is the fuzzy version of
``ARM.sequential_rules``, with the same definition of valid rule: here it is valid TO A DEGREE.

Degree of ``X => Y`` in a sequence: with ``T`` the t-norm (``min`` or product),
``mu_X(k) = T_{x in X} max_{p <= k} d(p, x)`` (to what degree all the items of X have appeared up to position k),
``mu_Y(k) = T_{y in Y} max_{p > k} d(p, y)`` (to what degree all those of Y appear after k), and

    degree(X => Y) = max_k T(mu_X(k), mu_Y(k)).

Measures, with the means over the sequences:

* support(X => Y) = mean of the degree of the rule;
* support of the antecedent (consequent) = mean of ``T_{x in X} max_p d(p, x)``: to what degree all the items of X (of
  Y) appear in the sequence, in any order;
* confidence = support(X => Y) / support of the antecedent; lift and the other measures of ``ARM._measures`` with those
  three supports.

With 0/1 degrees everything coincides with ``sequential_rules``. The search starts from the frequent itemsets of the
sequences seen as baskets (the degree of an itemset is the t-norm of the best degrees of its items; it is
anti-monotone), and tries their partitions.

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..FIM.fuzzy_sequential import _check_tnorm, degree_tensor, normalize
from ._measures import MEASURES

_EPS = 1e-9


def _reduce(combine: Any, values: np.ndarray) -> np.ndarray:
    """The t-norm of the columns (last axis) of `values`."""
    result = values[..., 0]
    for column in range(1, values.shape[-1]):
        result = combine(result, values[..., column])
    return result


def _frequent_itemsets(basket: np.ndarray, combine: Any, min_supp: float, max_size: int) -> Dict[Tuple[int, ...], float]:
    """Itemsets (tuples of columns) whose fuzzy support as baskets is >= min_supp, level by level."""
    frequent: Dict[Tuple[int, ...], float] = {}
    level: List[Tuple[int, ...]] = []
    for column in range(basket.shape[1]):
        support = float(basket[:, column].mean())
        if support >= min_supp - _EPS:
            frequent[(column,)] = support
            level.append((column,))
    size = 1
    while level and size < max_size:
        known = set(level)
        candidates = sorted({a + (b[-1],) for a in level for b in level if a[:-1] == b[:-1] and a[-1] < b[-1]})
        level = []
        for candidate in candidates:
            if any(subset not in known for subset in combinations(candidate, size)):
                continue
            support = float(_reduce(combine, basket[:, list(candidate)]).mean())
            if support >= min_supp - _EPS:
                frequent[candidate] = support
                level.append(candidate)
        size += 1
    return frequent


def fuzzy_sequential_rules(sequences: Iterable[Iterable[Any]], min_supp: float, min_conf: float = 0.8,
                           max_antecedent: int = 2, max_consequent: int = 2, metric: str = "confidence",
                           min_threshold: float = 0.0, tnorm: str = "min") -> pd.DataFrame:
    """
    Mine fuzzy sequential rules ``X => Y`` (see the module docstring).

    Args:
        sequences: one sequence per element; each, a list of elements with their membership degrees (see
            ``FIM.fuzzy_sequential``)
        min_supp (float): minimum fuzzy support of the rule, in (0, 1]
        min_conf (float): minimum confidence, in [0, 1]
        max_antecedent, max_consequent (int): maximum number of items of X and of Y
        metric, min_threshold: additional filter, as in ``sequential_rules``
        tnorm (str): 'min' (default) or 'product'

    Returns:
        pandas.DataFrame with the format of ``sequential_rules`` (columns 'antecedents' and 'consequents', 'antecedent
        support', 'consequent support', 'support', 'confidence', 'lift' and the other measures), sorted by decreasing
        confidence and support. Memory: proportional to sequences x maximum length x items
    """
    combine = _check_tnorm(tnorm)
    database = normalize(sequences)
    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support",
               "confidence", "lift"] + [name for name in MEASURES if name not in ("confidence", "lift")]
    if not database:
        return pd.DataFrame(columns=columns)
    if metric not in MEASURES and metric != "support":
        raise ValueError("metric must be 'support' or one of {}, received: '{}'".format(sorted(MEASURES), metric))
    if not 0 < min_supp <= 1:
        raise ValueError("min_supp must be in (0, 1]")
    if max_antecedent < 1 or max_consequent < 1:
        raise ValueError("max_antecedent and max_consequent must be >= 1")

    tensor, vocabulary = degree_tensor(database)
    total = len(database)
    basket = tensor.max(axis=1)                                    # best degree of each item in each sequence
    upto = np.maximum.accumulate(tensor, axis=1)                   # mejor grado hasta la posicion k (incluida)
    later = np.maximum.accumulate(tensor[:, ::-1, :], axis=1)[:, ::-1, :]
    after = np.concatenate([later[:, 1:, :], np.zeros_like(later[:, :1, :])], axis=1)      # ... estrictamente despues de k

    cache: Dict[Tuple[int, ...], float] = {}

    def basket_support(items: Tuple[int, ...]) -> float:
        if items not in cache:
            cache[items] = float(_reduce(combine, basket[:, list(items)]).mean())
        return cache[items]

    rows: List[dict] = []
    for itemset in _frequent_itemsets(basket, combine, min_supp, max_antecedent + max_consequent):
        if len(itemset) < 2:
            continue
        for size in range(1, min(max_antecedent, len(itemset) - 1) + 1):
            if len(itemset) - size > max_consequent:
                continue
            for antecedent in combinations(itemset, size):
                consequent = tuple(column for column in itemset if column not in antecedent)
                degree = combine(_reduce(combine, upto[:, :, list(antecedent)]),
                                 _reduce(combine, after[:, :, list(consequent)])).max(axis=1)
                support = float(degree.mean())
                if support < min_supp - _EPS:
                    continue
                supportX, supportY = basket_support(antecedent), basket_support(consequent)
                confidence = support / supportX
                if confidence < min_conf - _EPS:
                    continue
                entry: Dict[str, Any] = {"antecedents": frozenset(vocabulary[c] for c in antecedent),
                                         "consequents": frozenset(vocabulary[c] for c in consequent),
                                         "antecedent support": supportX, "consequent support": supportY,
                                         "support": support}
                for name, function in MEASURES.items():
                    entry[name] = float(function(support, supportX, supportY))
                if metric != "support" and min_threshold and entry[metric] < min_threshold:
                    continue
                rows.append(entry)

    result = pd.DataFrame(rows, columns=columns)
    result = result.sort_values(["confidence", "support"], ascending=False, kind="mergesort").reset_index(drop=True)
    result.attrs["n_sequences"] = total
    return result


def fuzzy_rule_degree(sequence: Iterable[Any], antecedent: Iterable[str], consequent: Iterable[str],
                      tnorm: str = "min") -> float:
    """Degree to which the rule ``antecedent => consequent`` is valid in a sequence (see the module docstring)."""
    combine = _check_tnorm(tnorm)
    tensor, vocabulary = degree_tensor(normalize([sequence]))
    column = {item: position for position, item in enumerate(vocabulary)}
    x, y = list(antecedent), list(consequent)
    if any(item not in column for item in x + y):
        return 0.0
    upto = np.maximum.accumulate(tensor[0], axis=0)
    later = np.maximum.accumulate(tensor[0][::-1], axis=0)[::-1]
    after = np.concatenate([later[1:], np.zeros_like(later[:1])], axis=0)
    before = _reduce(combine, upto[:, [column[item] for item in x]])
    following = _reduce(combine, after[:, [column[item] for item in y]])
    return float(combine(before, following).max())
