# -*- coding: utf-8 -*-
"""
Utilities shared by the frequent-itemset mining algorithms (crisp and fuzzy, sequential and Big Data) of ARMxtend.

These functions replace the several slightly different, buggy copies of ``GenItems``/``AlphaCortes`` that used to be
duplicated in ``FIM/apriori.py``, ``FIM/BD_ARE``, ``FIM/BD_FARE`` and ``FFIM``.

@author Carlos Fernandez-Basso
"""
from typing import FrozenSet, Iterable, List, Mapping
import itertools

import numpy as np
from numpy.typing import ArrayLike

ITEM_SEP = "-"


def itemset_to_key(itemset: Iterable[str]) -> str:
    """Convert an itemset (iterable of items) into its canonical sorted key."""
    return ITEM_SEP.join(sorted(itemset))


def key_to_itemset(key: str) -> FrozenSet[str]:
    """Convert a canonical key ('A-B-C') into a frozenset of items."""
    return frozenset(key.split(ITEM_SEP))


def generate_candidates(freq_itemsets: Iterable[str]) -> List[str]:
    """
    Generate candidates of length k+1 from a set of frequent itemsets of length k, following the apriori-gen
    procedure (Agrawal & Srikant, 1994): two frequent itemsets are joined if they share their first k-1 items
    (once sorted lexicographically), and the resulting candidate is kept only if ALL its subsets of length k are
    also frequent (downward closure property).

    Args:
        freq_itemsets (Iterable[str]): keys ('A-B-C') of the frequent itemsets of length k

    Returns:
        Sorted list of keys of the candidate itemsets of length k+1
    """
    itemsets = sorted(key_to_itemset(k) for k in freq_itemsets)
    freq_set = set(itemsets)
    length = (len(itemsets[0]) + 1) if itemsets else 1

    sorted_items = [tuple(sorted(s)) for s in itemsets]
    candidates = set()
    n = len(sorted_items)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted_items[i], sorted_items[j]
            # Only itemsets that share the same prefix are joined
            # (longitud k-1), difiriendo unicamente en el ultimo item
            if a[:-1] != b[:-1]:
                break
            candidate = frozenset(a) | frozenset(b)
            if len(candidate) != length:
                continue
            if all(frozenset(sub) in freq_set
                   for sub in itertools.combinations(candidate, length - 1)):
                candidates.add(candidate)

    return sorted(itemset_to_key(c) for c in candidates)


def alpha_levels(num_alpha: int) -> np.ndarray:
    """
    Set Lambda = {alpha_1, ..., alpha_p} of equidistant alpha levels used to decompose fuzzy measures in the
    Representation by Levels theory (RL-theory) of Delgado, Ruiz, Sanchez & Serrano (2011, Information Sciences 181)
    and in Fernandez-Basso, Ruiz & Martin-Bautista (2021, Int. J. of Approximate Reasoning 137):
    alpha_i = 1 - (i-1)/p, for i = 1..p, with 1 = alpha_1 > alpha_2 > ... > alpha_p > alpha_(p+1) = 0.

    For example, for num_alpha=4: [1, 0.75, 0.5, 0.25] (the same numerical example used in both papers).

    Args:
        num_alpha (int): number of alpha levels (p), >= 1

    Returns:
        numpy.ndarray of length `num_alpha`, decreasing, with the first value equal to 1.0
    """
    if num_alpha < 1:
        raise ValueError("El numero de alpha-cortes debe ser >= 1, recibido: {}".format(num_alpha))
    return np.arange(num_alpha, 0, -1) / float(num_alpha)


def alpha_weights(levels: np.ndarray) -> np.ndarray:
    """
    Weights (alpha_i - alpha_(i+1)) of each alpha level, with alpha_(p+1) = 0, used to aggregate a measure computed
    at each level (see `alpha_levels`, Eqs. (2)-(6) of Fernandez-Basso et al. 2021 / Ruiz et al. 2016). For
    equidistant levels (see `alpha_levels`) all the weights are equal to 1/num_alpha, so the resulting aggregation is
    an arithmetic mean.

    Args:
        levels (numpy.ndarray): alpha levels, in decreasing order

    Returns:
        numpy.ndarray of the same length as `levels`
    """
    return levels - np.append(levels[1:], 0.0)


def weighted_alpha_aggregate(values_per_level: ArrayLike, weights: ArrayLike) -> float:
    """
    Weighted aggregation Σ_i weight_i * value_i of a measure computed at each alpha level (see `alpha_weights`). It
    is the central mechanism of the RL-theory to generalize any crisp measure (support, confidence, certainty
    factor, ...) to the fuzzy case, replacing the notion of "choosing an alpha-cut" by an integration over all the
    levels.

    Args:
        values_per_level (array-like): value of the crisp measure at each alpha level (same length as `weights`)
        weights (array-like): weight of each level (see `alpha_weights`)

    Returns:
        float: the aggregated fuzzy measure
    """
    return float(np.dot(np.asarray(values_per_level, dtype=float), weights))


def alpha_cuts(value: float, num_alpha: int) -> np.ndarray:
    """
    Alpha-cut representation (bit-list) of a fuzzy membership degree, following the FuzzyToArray procedure
    (Algorithm 2) of Fernandez-Basso, Ruiz & Martin-Bautista (2021): for each level alpha_i in
    `alpha_levels(num_alpha)`, position i is 1 if value >= alpha_i, and 0 otherwise.

    For example, alpha_cuts(0.25, 4) = [0, 0, 0, 1] with levels [1, 0.75, 0.5, 0.25] (the literal example of both
    papers).

    Args:
        value (float): fuzzy membership degree, in [0, 1]
        num_alpha (int): number of levels/alpha-cuts to consider (p)

    Returns:
        numpy.ndarray of integers (0/1) of length `num_alpha`, non-increasing
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError("El grado de pertenencia difuso debe estar en [0, 1], recibido: {}".format(value))

    return (value >= alpha_levels(num_alpha)).astype(int)


def fuzzy_support(itemset: Iterable[str], transaction_memberships: Mapping[str, np.ndarray],
                  num_alpha: int) -> np.ndarray:
    """
    Bit-list per alpha-cut of an itemset in a transaction: since t(A) = min_{i in A} t(i) (Definition 2, fuzzy
    transaction), the bit-list of A at each alpha level is the minimum (logical AND, as they are 0/1) of the
    bit-lists of its items.

    Args:
        itemset (Iterable[str]): items of the itemset to evaluate
        transaction_memberships (dict[str, numpy.ndarray]): for each item of the transaction, its bit-list of
            alpha-cuts (see `alpha_cuts`)
        num_alpha (int): number of alpha-cuts used

    Returns:
        numpy.ndarray: the bit-list of alpha-cuts of the itemset, or a vector of zeros if any of its items is
        missing from the transaction.
    """
    result = np.ones(num_alpha, dtype=int)
    for item in itemset:
        membership = transaction_memberships.get(item)
        if membership is None:
            return np.zeros(num_alpha, dtype=int)
        result = np.minimum(result, membership)
    return result
