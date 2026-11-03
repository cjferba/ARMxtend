# -*- coding: utf-8 -*-
"""
Explanation of rules: why a rule fires on a row, why it does not, what evidence supports it and how it compares with
the alternatives. Everything is computed from the boolean table of items the rules apply to (items with a lag
``item@lagk`` work as in ``ARM.anomaly``), so it explains rules from any miner.

* ``rule_evidence``: the contingency table of ``A => C`` on the data (rows where both hold, only A, only C, neither)
  and the rows that support and violate it.
* ``explain_rule``: evidence, the contribution of each item of the antecedent (confidence of the rule WITHOUT that item)
  and the alternatives in a set of rules: more general (a proper subset of the antecedent, same consequent), more
  specific (a proper superset), and other consequents of the same antecedent.
* ``explain_row``: for one row, which rules fire, which ones are confirmed or violated by it and, for those that do not
  fire, which items of the antecedent are missing.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, FrozenSet, Iterable, List, cast

import numpy as np
import pandas as pd

from ._apply import ItemTable, describe_rule, require_columns


def _confidence(table: ItemTable, antecedent: Iterable[str], consequent: Iterable[str]) -> float:
    usable = table.evaluable(list(antecedent) + list(consequent))
    holds = table.holds(antecedent) & usable
    return float((holds & table.holds(consequent)).sum() / holds.sum()) if holds.sum() else float("nan")


def rule_evidence(items: pd.DataFrame, antecedent: Iterable[str], consequent: Iterable[str]) -> Dict[str, Any]:
    """
    Evidence of the rule ``antecedent => consequent`` in `items`.

    Returns:
        dictionary with 'n_rows' (rows with enough history for the lags of the rule), the counts 'both' (A and C), 'only_antecedent'
        (A without C: the violations), 'only_consequent' and 'neither', 'support', 'confidence' and 'lift', and the
        index labels 'supporting' (A and C) and 'violating' (A without C)
    """
    antecedent, consequent = list(antecedent), list(consequent)
    table = ItemTable(items)
    usable = table.evaluable(antecedent + consequent)
    a, c = table.holds(antecedent) & usable, table.holds(consequent) & usable
    n = int(usable.sum())
    both, onlyA = int((a & c).sum()), int((a & ~c).sum())
    onlyC, neither = int((~a & c & usable).sum()), int((~a & ~c & usable).sum())
    confidence = both / (both + onlyA) if both + onlyA else float("nan")
    base = (both + onlyC) / n if n else float("nan")
    return {"n_rows": n, "both": both, "only_antecedent": onlyA, "only_consequent": onlyC, "neither": neither,
            "support": both / n if n else float("nan"), "confidence": confidence,
            "lift": confidence / base if base else float("nan"),
            "supporting": list(items.index[a & c]), "violating": list(items.index[a & ~c])}


def explain_rule(items: pd.DataFrame, rules: pd.DataFrame, position: int, top: int = 5) -> Dict[str, Any]:
    """
    Explain the rule at `position` of `rules` (a row label of the frame) against the data and its alternatives.

    Args:
        items (pandas.DataFrame): boolean (0/1) items
        rules (pandas.DataFrame): rules with 'antecedents' and 'consequents' (and optionally 'confidence'); it also
            supplies the alternatives
        position: label (index value) of the rule to explain
        top (int): maximum number of alternatives of each kind

    Returns:
        dictionary with 'rule' (text), 'evidence' (``rule_evidence``), 'contributions' (``DataFrame`` with, for each item of the antecedent,
        the confidence of the rule without it and the 'gain' that the item adds; NaN if the antecedent has one item), and
        'alternatives' (``DataFrame`` with 'relation', 'rule', 'confidence', 'support' and 'delta_confidence' with respect
        to the explained rule, measured on `items`)
    """
    require_columns(rules, "antecedents", "consequents")
    if position not in rules.index:
        raise ValueError("the rule {!r} is not in `rules`".format(position))
    antecedent = cast(FrozenSet[str], rules.loc[position, "antecedents"])
    consequent = cast(FrozenSet[str], rules.loc[position, "consequents"])
    table = ItemTable(items)
    evidence = rule_evidence(items, antecedent, consequent)
    rows: List[Dict[str, Any]] = []
    for item in sorted(antecedent):
        rest = antecedent - {item}
        without = _confidence(table, rest, consequent) if rest else _confidence(table, [], consequent)
        rows.append({"item": item, "confidence_without": without, "gain": evidence["confidence"] - without})
    contributions = pd.DataFrame(rows, columns=["item", "confidence_without", "gain"])

    alternatives: List[Dict[str, Any]] = []
    for other in rules.index:
        if other == position:
            continue
        a = cast(FrozenSet[str], rules.loc[other, "antecedents"])
        c = cast(FrozenSet[str], rules.loc[other, "consequents"])
        if c == consequent and a < antecedent:
            relation = "more general"
        elif c == consequent and a > antecedent:
            relation = "more specific"
        elif a == antecedent and c != consequent:
            relation = "same antecedent"
        else:
            continue
        other_evidence = rule_evidence(items, a, c)
        alternatives.append({"relation": relation, "rule": describe_rule(a, c), "label": other,
                             "confidence": other_evidence["confidence"], "support": other_evidence["support"],
                             "delta_confidence": other_evidence["confidence"] - evidence["confidence"]})
    table_alternatives = pd.DataFrame(alternatives, columns=["relation", "rule", "label", "confidence", "support",
                                                             "delta_confidence"])
    if len(table_alternatives):
        table_alternatives = table_alternatives.sort_values(["relation", "confidence"], ascending=[True, False],
                                                            kind="mergesort")
        table_alternatives = table_alternatives.groupby("relation", sort=False).head(top).reset_index(drop=True)
    return {"rule": describe_rule(antecedent, consequent), "evidence": evidence, "contributions": contributions,
            "alternatives": table_alternatives}


def explain_row(items: pd.DataFrame, rules: pd.DataFrame, row: Any) -> pd.DataFrame:
    """
    What the rules say about one row of `items`.

    Args:
        items (pandas.DataFrame): boolean (0/1) items (the previous rows are used for the lags)
        rules (pandas.DataFrame): rules with 'antecedents' and 'consequents'
        row: index label of the row

    Returns:
        ``DataFrame`` with one row per rule (same index as `rules`) and the columns 'rule' (text), 'status' ('confirmed':
        A and C hold; 'violated': A holds and C does not; 'not fired': A does not hold; 'no history': the lags reach
        before the first row), 'missing' (items of the antecedent that do not hold: why it did not fire) and 'confidence'
        (of the rule, if `rules` has that column)
    """
    require_columns(rules, "antecedents", "consequents")
    if row not in items.index:
        raise ValueError("the row {!r} is not in `items`".format(row))
    table = ItemTable(items)
    where = int(np.flatnonzero(items.index == row)[0])
    records: List[Dict[str, Any]] = []
    for label, antecedent, consequent in zip(rules.index, rules["antecedents"], rules["consequents"]):
        missing = sorted(item for item in antecedent if not table.item(item)[where])
        if not table.evaluable(list(antecedent) + list(consequent))[where]:
            status = "no history"
        elif not missing:
            status = "confirmed" if table.holds(consequent)[where] else "violated"
        else:
            status = "not fired"
        records.append({"rule": describe_rule(antecedent, consequent), "status": status,
                        "missing": [] if status == "no history" else missing,
                        "confidence": rules.loc[label, "confidence"] if "confidence" in rules.columns else np.nan})
    return pd.DataFrame(records, index=rules.index)


__all__ = ["rule_evidence", "explain_rule", "explain_row"]
