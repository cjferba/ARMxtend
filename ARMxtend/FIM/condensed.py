# -*- coding: utf-8 -*-
"""
Condensed representations of the frequent itemsets: CLOSED and MAXIMAL itemsets.

* An itemset is CLOSED if none of its supersets has exactly its same support (Pasquier, Bastide, Taouil & Lakhal,
  1999). The closed itemsets keep the support of ALL the frequent ones (see `support_from_closed`) with fewer
  itemsets.
* An itemset is MAXIMAL if none of its supersets is frequent (Bayardo, 1998). It is an even smaller
  representation, but loses the supports of the subsets.

Both functions work on any dictionary {key 'A-B-C': support} produced by the ARMxtend algorithms, both crisp
(scalar support: ``FFIM.fpgrowth``, ``FIM.apriori.DApriori``, ``FIM.Eclat.DECLAT``, ...) and fuzzy (support =
bit-list per alpha-cut: ``FFIM.fuzzy_fpgrowth``, ``FIM.Eclat.FuzzyDECLAT``, ...). In the fuzzy case, two itemsets
"have the same support" if they coincide in ALL the alpha-cuts, that is, the closure is defined level by level,
consistently with the RL-theory.

It is enough to compare each itemset with its IMMEDIATE supersets (one more item): support is anti-monotone, so if
some superset had the same support the intermediate itemset would too, and it is frequent and therefore in the
dictionary.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, Mapping, Union
import numpy as np

from ._shared import key_to_itemset, itemset_to_key


def _same_support(first, second, tol):
    return bool(np.all(np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) <= tol))


def _immediate_subsets(freq_itemsets):
    """Generate (key of itemset Z, key of each subset X = Z - {item} that is also frequent)."""
    for key in freq_itemsets:
        items = key_to_itemset(key)
        if len(items) < 2:
            continue
        for item in items:
            subsetKey = itemset_to_key(items - {item})
            if subsetKey in freq_itemsets:
                yield key, subsetKey


def closed_itemsets(freq_itemsets: Mapping[str, Any], tol: float = 1e-12) -> Dict[str, Any]:
    """
    Closed frequent itemsets.

    Args:
        freq_itemsets (dict[str, float | numpy.ndarray]): frequent itemsets with their support (scalar or
            bit-list per alpha-cut)
        tol (float): absolute tolerance to consider two supports equal

    Returns:
        dict with the entries of `freq_itemsets` that are closed
    """
    notClosed = set()
    for superset, subset in _immediate_subsets(freq_itemsets):
        if _same_support(freq_itemsets[superset], freq_itemsets[subset], tol):
            notClosed.add(subset)
    return {key: support for key, support in freq_itemsets.items() if key not in notClosed}


def maximal_itemsets(freq_itemsets: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Maximal frequent itemsets (with no frequent superset).

    Args:
        freq_itemsets (dict[str, float | numpy.ndarray]): frequent itemsets

    Returns:
        dict with the entries of `freq_itemsets` that are maximal
    """
    notMaximal = {subset for _, subset in _immediate_subsets(freq_itemsets)}
    return {key: support for key, support in freq_itemsets.items() if key not in notMaximal}


def support_from_closed(itemset: Union[str, Iterable[str]], closed: Mapping[str, Any]) -> Any:
    """
    Recover the support of a frequent itemset from the closed itemsets: it is the MAXIMUM (element by element, if it
    is a bit-list) among the supports of the closed itemsets that contain it.

    Args:
        itemset (str | Iterable[str]): key 'A-B' or collection of items
        closed (dict): output of `closed_itemsets`

    Returns:
        The support of the itemset, or None if no closed itemset contains it (that is, if it is not frequent).
    """
    items = key_to_itemset(itemset) if isinstance(itemset, str) else frozenset(itemset)
    supports = [support for key, support in closed.items() if items <= key_to_itemset(key)]
    if not supports:
        return None
    if np.ndim(supports[0]) == 0:
        return max(supports)
    return np.max(np.vstack(supports), axis=0)
