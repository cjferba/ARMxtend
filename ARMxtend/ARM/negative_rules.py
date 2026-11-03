# -*- coding: utf-8 -*-
"""
Association rules with ABSENT items (negative rules), crisp and fuzzy: rules such as  A -> ~B  ("if A, then NOT B"),
~A -> B  or  ~A -> ~B, where ``~i`` means that item ``i`` is not present in the transaction.

It follows the approach of section 4 of:

    Delgado, M., Ruiz, M.D., Sanchez, D., Serrano, J.M. (2011). "A formal
    model for mining fuzzy rules using the RL representation theory".
    Information Sciences, 181, 5194-5213.

In the RL-theory the absence of an item is its COMPLEMENT at each restriction level (Eqs. (27)-(28)): the bit-list of
``~i`` at level alpha_k is the complement of the bit-list of ``i`` at that level, and the 4ft contingency table of
(~A, B) is <c, d, a, b> from that of (A, B) = <a, b, c, d>. Therefore no new frequencies need to be computed, and --
unlike the negation of fuzzy set theory (1 - mu) -- an itemset never contains ``i`` and ``~i`` at the same time (its
support is 0 at every level), so trivial rules such as ``i -> ~i`` cannot appear.

The biggest problem of mining with absent items is that, in sparse data, the absence of an item is almost always
frequent and the number of itemsets explodes (the support of absence does not satisfy the upward-closure property
used for pruning). The paper adopts the restriction of Han & Beheshti: consider only the negative items ``~i`` whose
positive item ``i`` is frequent (``only_frequent_positive=True``, default). With ``negate_items`` it is also possible
to explicitly bound which items are negated.

NOTE: item names must not contain '-', which is the item separator in the itemset keys of ``FIM._shared``.

@author Carlos Fernandez-Basso
"""
from typing import Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from ..FFIM import fpgrowth, fuzzy_fpgrowth_from_vectors
from ..FIM._shared import key_to_itemset, alpha_cuts, alpha_levels, alpha_weights, weighted_alpha_aggregate
from ..FIM.FARE import fuzzy_association_rules
from .association_rules import association_rules

NEGATION_PREFIX = "~"


def negate(item: str) -> str:
    """Name of the item that represents the ABSENCE of `item` ('A' -> '~A')."""
    return NEGATION_PREFIX + item


def is_negated(item: str) -> bool:
    """True if `item` is an absent item ('~A')."""
    return item.startswith(NEGATION_PREFIX)


def _has_negated_item(row):
    return any(is_negated(item) for item in set(row["antecedents"]) | set(row["consequents"]))


def _negatable_items(vocabulary, positive_supports, negate_items, only_frequent_positive, min_supp):
    candidates = set(vocabulary) if negate_items is None else set(negate_items) & set(vocabulary)
    if only_frequent_positive:
        candidates = {item for item in candidates if positive_supports[item] >= min_supp}
    return sorted(candidates)


def add_absent_items(transactions: Sequence[Iterable[str]],
                     negate_items: Optional[Iterable[str]] = None) -> List[Set[str]]:
    """
    Extend each crisp transaction with an item ``~i`` for each item ``i`` of `negate_items` that is NOT present in it.

    Args:
        transactions (Sequence[Iterable[str]]): crisp transactions
        negate_items (Iterable[str], optional): items whose absence is represented; by default, all those that appear
            in some transaction

    Returns:
        list[set[str]]: the extended transactions
    """
    sets = [set(t) for t in transactions]
    negatable = set(negate_items) if negate_items is not None else set().union(*sets) if sets else set()
    return [t | {negate(item) for item in negatable - t} for t in sets]


def negative_association_rules(transactions: Sequence[Iterable[str]], min_supp: float, metric: str = "confidence",
                               min_threshold: float = 0.8, negate_items: Optional[Iterable[str]] = None,
                               only_frequent_positive: bool = True, only_negative: bool = True) -> pd.DataFrame:
    """
    Mine crisp association rules with present and/or absent items.

    Args:
        transactions (Sequence[Iterable[str]]): crisp transactions
        min_supp (float): minimum relative support of the itemsets (which can include absent items), in (0, 1]
        metric (str): measure of `ARM.association_rules` to filter by
        min_threshold (float): minimum threshold of `metric`
        negate_items (Iterable[str], optional): items whose absence is considered (all by default)
        only_frequent_positive (bool): only the items whose positive support reaches `min_supp` are negated
            (restriction of Han & Beheshti adopted in Delgado et al., 2011, Section 6)
        only_negative (bool): if True (default), return only the rules that contain at least one absent item; if
            False, also the purely positive ones

    Returns:
        pandas.DataFrame of rules (same format as `ARM.association_rules`); absent items appear as '~item'.
    """
    transactions = [set(t) for t in transactions]
    total = len(transactions)
    if total == 0:
        return association_rules(pd.DataFrame({"support": [], "itemsets": []}), min_threshold=min_threshold)

    vocabulary = set().union(*transactions)
    positiveSupports = {item: sum(item in t for t in transactions) / total for item in vocabulary}
    negatable = _negatable_items(vocabulary, positiveSupports, negate_items, only_frequent_positive, min_supp)

    extended = add_absent_items(transactions, negate_items=negatable)
    frequent = fpgrowth(extended, min_supp)
    itemsets = pd.DataFrame({
        "support": list(frequent.values()),
        "itemsets": [key_to_itemset(key) for key in frequent],
    })
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)

    if only_negative and len(rules):
        rules = rules[rules.apply(_has_negated_item, axis=1)].reset_index(drop=True)
    return rules


def fuzzy_negative_association_rules(transactions: Sequence[Iterable[Tuple[str, float]]], min_supp: float,
                                     num_alpha: int, metric: str = "confidence", min_threshold: float = 0.8,
                                     negate_items: Optional[Iterable[str]] = None,
                                     only_frequent_positive: bool = True, only_negative: bool = True,
                                     measures: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Mine FUZZY association rules with present and/or absent items, according to the RL-theory (see the module): the
    bit-list of ``~i`` at each alpha-cut is the complement of that of ``i`` (not ``1 - degree``), and the support
    (FSupp), confidence (FConf), certainty factor (FCF) and the rest of the measures are aggregated level by level
    with the weights (alpha_i - alpha_i+1) as in `FIM.FARE.fuzzy_association_rules`.

    Args:
        transactions (Sequence[Iterable[Tuple[str, float]]]): fuzzy transactions, each a list of (item, degree in
            [0, 1]) pairs. An item that does not appear in a transaction has degree 0 in it
        min_supp (float): minimum FSupp of the itemsets, in (0, 1]
        num_alpha (int): number of alpha-cuts (the paper recommends 10)
        metric (str): measure of `FIM.FARE.fuzzy_association_rules`
        min_threshold (float): minimum threshold of `metric`
        negate_items (Iterable[str], optional): items whose absence is considered (all by default)
        only_frequent_positive (bool): only the items whose positive FSupp reaches `min_supp` are negated
            (Han & Beheshti)
        only_negative (bool): only rules with at least one absent item
        measures (Iterable[str], optional): additional measures to return (see
            `FIM.FARE.fuzzy_association_rules`)

    Returns:
        pandas.DataFrame of fuzzy rules; absent items appear as '~item'.
    """
    columns = ["antecedents", "consequents", "support", "confidence", "certainty_factor"]
    total = len(transactions)
    if total == 0:
        return pd.DataFrame(columns=columns + list(measures or []))

    weights = alpha_weights(alpha_levels(num_alpha))
    vectors = [{item: alpha_cuts(degree, num_alpha) for item, degree in tx} for tx in transactions]

    vocabulary = set().union(*(v.keys() for v in vectors))
    positiveSupports = {}
    for item in vocabulary:
        perLevel = sum((v[item] for v in vectors if item in v), np.zeros(num_alpha)) / total
        positiveSupports[item] = weighted_alpha_aggregate(perLevel, weights)
    negatable = _negatable_items(vocabulary, positiveSupports, negate_items, only_frequent_positive, min_supp)

    zeros = np.zeros(num_alpha, dtype=int)
    for v in vectors:
        for item in negatable:
            v[negate(item)] = 1 - v.get(item, zeros)

    frequent = fuzzy_fpgrowth_from_vectors(vectors, min_supp, num_alpha)
    rules = fuzzy_association_rules(frequent, num_alpha, metric=metric, min_threshold=min_threshold,
                                    measures=measures)

    if only_negative and len(rules):
        rules = rules[rules.apply(_has_negated_item, axis=1)].reset_index(drop=True)
    return rules
