# -*- coding: utf-8 -*-
"""
TEMPORAL association rules with lag (inter-transaction rules), crisp and fuzzy: rules such as
``temp=high@lag2, door_open@lag1 -> heater_on`` ("if two steps ago the temperature was high and one step ago the door
was open, then now the heating is on").

The formulation of the inter-transaction association rules of the following is followed:

    Lu, H., Han, J., Feng, L. (1998). "Stock movement prediction and
    N-dimensional inter-transaction association rules". ACM SIGMOD Workshop
    on Research Issues in Data Mining and Knowledge Discovery.

    Tung, A.K.H., Lu, H., Han, J., Feng, L. (2003). "Efficient mining of
    intertransaction association rules". IEEE Transactions on Knowledge and
    Data Engineering, 15(1), 43-56.

Each instant ``t`` (from ``max_lag``) becomes a transaction that contains the items of ``t`` (without suffix) and
those of ``t-1, ..., t-max_lag`` (with suffix ``@lagk``, see ``preprocessing.timeseries``). The frequent itemsets of
those transactions are mined and the rules whose CONSEQUENT is only at the current instant and whose antecedent looks
at the past are kept. The support is computed over the ``N - max_lag`` rows that have all the lags available, so that
all the measures share the denominator. The fuzzy version uses the membership degrees with the alpha-cut
decomposition of the RL-theory (``FIM.FARE``).

The cost grows with the number of items times the number of lags: tune ``min_supp``, ``lags`` and, above all,
symbolize with few values (``preprocessing.timeseries.symbolize``).

NOTE: item names must not contain '-', which is the item separator in the itemset keys of ``FIM._shared``.

@author Carlos Fernandez-Basso
"""
from typing import Any, FrozenSet, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth, fuzzy_fpgrowth
from ..FIM._shared import key_to_itemset
from ..FIM.FARE import fuzzy_association_rules
from ..preprocessing.timeseries import (
    check_item_names, item_variable, lag_items, split_lag, to_fuzzy_transactions, to_transactions,
)
from .association_rules import association_rules
from .significance import count_candidates

def _lag_set(max_lag: int, lags: Optional[Iterable[int]]) -> List[int]:
    if max_lag < 1:
        raise ValueError("max_lag debe ser >= 1, recibido: {}".format(max_lag))
    chosen = range(1, max_lag + 1) if lags is None else sorted({int(lag) for lag in lags})
    if any(lag < 1 or lag > max_lag for lag in chosen):
        raise ValueError("lags debe estar contenido en 1..{}, recibido: {}".format(max_lag, list(chosen)))
    return [0] + list(chosen)


def _is_target(item: str, targets: Optional[FrozenSet[str]]) -> bool:
    return targets is None or item in targets or item_variable(item) in targets


def _keep_rules(rules: pd.DataFrame, targets: Optional[Iterable[str]], require_lagged_antecedent: bool,
                same_time_antecedent: bool, exclude_persistence: bool) -> pd.DataFrame:
    """Filter the rules by temporal structure and add the columns 'lags' and 'max_lag'."""
    wanted = None if targets is None else frozenset(targets)
    if len(rules):
        mask = rules.apply(lambda rule: _passes(rule, wanted, require_lagged_antecedent, same_time_antecedent,
                                                exclude_persistence), axis=1).astype(bool)
        result = rules[mask].reset_index(drop=True)
    else:
        result = rules.copy()
    lags = [tuple(sorted({lag for lag in (split_lag(item)[1] for item in antecedent) if lag > 0}))
            for antecedent in result["antecedents"]]
    result["lags"] = pd.Series(lags, index=result.index, dtype=object)
    result["max_lag"] = pd.Series([max(item, default=0) for item in lags], index=result.index, dtype=int)
    return result


def _passes(rule: Any, wanted: Optional[FrozenSet[str]], require_lagged_antecedent: bool,
            same_time_antecedent: bool, exclude_persistence: bool) -> bool:
    return _valid_split(rule["antecedents"], rule["consequents"], wanted, require_lagged_antecedent,
                        same_time_antecedent, exclude_persistence)


def _valid_split(antecedents: Iterable[str], consequents: Iterable[str], wanted: Optional[FrozenSet[str]],
                 require_lagged_antecedent: bool, same_time_antecedent: bool, exclude_persistence: bool) -> bool:
    consequent = [split_lag(item) for item in consequents]
    antecedent = [split_lag(item) for item in antecedents]
    if any(lag != 0 for _, lag in consequent):
        return False
    if wanted is not None and not all(_is_target(item, wanted) for item, _ in consequent):
        return False
    if require_lagged_antecedent and not any(lag > 0 for _, lag in antecedent):
        return False
    if not same_time_antecedent and any(lag == 0 for _, lag in antecedent):
        return False
    if exclude_persistence:
        consequentVariables = {item_variable(item) for item, _ in consequent}
        if any(lag > 0 and item_variable(item) in consequentVariables for item, lag in antecedent):
            return False
    return True


def _count_candidates(itemsets: pd.DataFrame, targets: Optional[Iterable[str]], require_lagged_antecedent: bool,
                      same_time_antecedent: bool, exclude_persistence: bool) -> int:
    """Candidate rules with valid temporal structure that come out of the frequent itemsets (the family of hypotheses)."""
    wanted = None if targets is None else frozenset(targets)
    return count_candidates(
        itemsets["itemsets"],
        lambda a, c: _valid_split(a, c, wanted, require_lagged_antecedent, same_time_antecedent, exclude_persistence),
        lambda item: split_lag(item)[1] == 0 and _is_target(item, wanted))


def temporal_association_rules(items: pd.DataFrame, max_lag: int, min_supp: float, metric: str = "confidence",
                               min_threshold: float = 0.8, lags: Optional[Iterable[int]] = None,
                               targets: Optional[Iterable[str]] = None, require_lagged_antecedent: bool = True,
                               same_time_antecedent: bool = True, exclude_persistence: bool = False) -> pd.DataFrame:
    """
    Mine crisp association rules with lag: antecedent at earlier instants (``item@lagk``) and, optionally, at the
    current one, and consequent at the current instant.

    Args:
        items (pandas.DataFrame): boolean items, one row per instant and ordered in time (see
            ``preprocessing.timeseries.to_items``)
        max_lag (int): largest lag of the antecedent, in steps, >= 1
        min_supp (float): minimum relative support of the itemsets, in (0, 1]
        metric (str): measure of `ARM.association_rules` to filter by
        min_threshold (float): minimum threshold of `metric`
        lags (Iterable[int], optional): allowed lags, within 1..max_lag (all by default); e.g. [1, 24] to look only at
            "one step ago" and "24 steps ago"
        targets (Iterable[str], optional): allowed consequents, by item name ('heater_on', 'temp=high') or by variable
            ('temp'); by default, any
        require_lagged_antecedent (bool): if True (default), the antecedent must include some item from an earlier
            instant; if False, co-occurrence rules at the same instant are also returned
        same_time_antecedent (bool): if False, the antecedent can only contain items from earlier instants
        exclude_persistence (bool): if True, discard the rules whose antecedent includes, with lag, the same variable
            as the consequent (e.g. 'temp=high@lag1 -> temp=high')

    Returns:
        pandas.DataFrame with the format of `ARM.association_rules` and two more columns: 'lags' (tuple of distinct
        lags of the antecedent) and 'max_lag' (the largest of them). It records in ``rules.attrs`` the number of
        transactions ('n_transactions') and that of candidate rules with valid temporal structure ('n_candidates'),
        which ``ARM.significance`` uses
    """
    check_item_names(items.columns)
    lagged = lag_items(items, _lag_set(max_lag, lags))
    frequent = fpgrowth(to_transactions(lagged), min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()),
                             "itemsets": [key_to_itemset(key) for key in frequent]})
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    kept = _keep_rules(rules, targets, require_lagged_antecedent, same_time_antecedent, exclude_persistence)
    kept.attrs["n_transactions"] = len(lagged)
    kept.attrs["n_candidates"] = _count_candidates(itemsets, targets, require_lagged_antecedent,
                                                   same_time_antecedent, exclude_persistence)
    return kept


def fuzzy_temporal_association_rules(memberships: pd.DataFrame, max_lag: int, min_supp: float, num_alpha: int = 10,
                                     metric: str = "confidence", min_threshold: float = 0.8,
                                     lags: Optional[Iterable[int]] = None, targets: Optional[Iterable[str]] = None,
                                     require_lagged_antecedent: bool = True, same_time_antecedent: bool = True,
                                     exclude_persistence: bool = False,
                                     measures: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Fuzzy version of `temporal_association_rules`: the items are membership degrees in [0, 1] and the support,
    confidence and certainty factor are computed with the alpha-cut decomposition of the RL-theory (`FIM.FARE`).

    Args:
        memberships (pandas.DataFrame): membership degrees, one column per item (e.g. the columns added by
            ``FuzzyLib.AutoFuzzification``) and one row per instant, ordered in time
        max_lag, lags, targets, require_lagged_antecedent,
        same_time_antecedent, exclude_persistence: see `temporal_association_rules`
        min_supp (float): minimum fuzzy support FSupp, in (0, 1]
        num_alpha (int): number of alpha-cuts
        metric, min_threshold, measures: see `FIM.FARE.fuzzy_association_rules`

    Returns:
        pandas.DataFrame with the format of `FIM.FARE.fuzzy_association_rules` and the columns 'lags' and 'max_lag'
    """
    check_item_names(memberships.columns)
    lagged = lag_items(memberships, _lag_set(max_lag, lags))
    frequent = fuzzy_fpgrowth(to_fuzzy_transactions(lagged), min_supp, num_alpha)
    rules = fuzzy_association_rules(frequent, num_alpha, metric=metric, min_threshold=min_threshold,
                                    measures=measures)
    kept = _keep_rules(rules, targets, require_lagged_antecedent, same_time_antecedent, exclude_persistence)
    kept.attrs["n_transactions"] = len(lagged)
    return kept


def format_lagged_item(item: str) -> str:
    """Readable text of a lagged item: 'temp=high@lag2' -> 'temp=high (t-2)'; without lag, 'temp=high (t)'."""
    base, lag = split_lag(item)
    return "{} (t{})".format(base, "-{}".format(lag) if lag else "")


def rule_profile(items: pd.DataFrame, antecedent: Iterable[str], consequent: Iterable[str],
                 by: Union[Sequence[Any], np.ndarray, pd.Series, Any]) -> pd.DataFrame:
    """
    How a rule behaves in each group of instants (by hour of the day, day of the week, month, season...): it serves to
    detect CYCLIC rules, which are weak on average but very reliable at certain hours or days.

    Args:
        items (pandas.DataFrame): boolean items, one row per instant. The rule can use lagged items (``item@lagk``):
            they are computed from the columns of `items`
        antecedent (Iterable[str]): items of the antecedent (with or without lag)
        consequent (Iterable[str]): items of the consequent (with or without lag)
        by: group of each row of `items`: a sequence or ``Series`` of the same length (e.g. ``items.index.hour``) or a
            function that receives the index of `items` and returns that sequence

    Returns:
        pandas.DataFrame indexed by group with the columns 'n' (instants of the group), 'antecedent_count',
        'joint_count', 'support', 'confidence' (NaN if the antecedent does not appear) and 'lift' (NaN if not
        defined). The first rows, without data for the lags, are not counted. For the total, use a constant `by`.
    """
    antecedent = list(antecedent)
    consequent = list(consequent)
    parsed = [split_lag(item) for item in antecedent + consequent]
    missing = [base for base, _ in parsed if base not in items.columns]
    if missing:
        raise ValueError("items that are not columns of `items`: {}".format(sorted(set(missing))))

    groups = np.asarray(by(items.index) if callable(by) else by)
    if len(groups) != len(items):
        raise ValueError("`by` must have one entry per row ({}), it has {}".format(len(items), len(groups)))
    start = max(lag for _, lag in parsed)
    rows = len(items)

    def holds(names):
        present = np.ones(rows - start, dtype=bool)
        for item in names:
            base, lag = split_lag(item)
            present &= items[base].to_numpy(dtype=bool)[start - lag:rows - lag]
        return present

    antecedentHolds, consequentHolds = holds(antecedent), holds(consequent)
    table = pd.DataFrame({"group": groups[start:], "a": antecedentHolds, "c": consequentHolds})
    table["ac"] = table["a"] & table["c"]
    grouped = table.groupby("group")
    result = pd.DataFrame({"n": grouped.size(), "antecedent_count": grouped["a"].sum(),
                           "joint_count": grouped["ac"].sum(), "consequent_count": grouped["c"].sum()})
    result["support"] = result["joint_count"] / result["n"]
    result["confidence"] = result["joint_count"] / result["antecedent_count"].where(result["antecedent_count"] > 0)
    consequentRate = result["consequent_count"] / result["n"]
    result["lift"] = result["confidence"] / consequentRate.where(consequentRate > 0)
    return result.drop(columns="consequent_count")
