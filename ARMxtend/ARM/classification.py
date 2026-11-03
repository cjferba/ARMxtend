# -*- coding: utf-8 -*-
"""
Classification with CLASS association rules (CARs): rules ``A => class=c`` whose consequent is a value of the
variable to predict, and a classifier that orders and selects them, following the CBA algorithm (Classification Based
on Associations):

    Liu, B., Hsu, W., Ma, Y. (1998). "Integrating classification and association
    rule mining". Proceedings of the 4th International Conference on Knowledge
    Discovery and Data Mining (KDD-98), 80-86.

* ``class_association_rules`` extracts the frequent and reliable ``A => class=c`` rules. With ``support="class"``
  (default) the minimum support is applied within each class, so that infrequent classes do not end up without rules.
  This is the multiple-minimum-supports scheme of Liu, Ma and Wong (2000): the minimum support of the rules of class c
  is ``t * freq(c)`` over all the cases, which is the same as ``t`` over the cases of the class. It can also be given
  PER CLASS (``min_supp={"rare": 0.2, "common": 0.05}``, ``min_conf`` too) and floored with ``min_count`` (an absolute
  number of cases, so that a tiny class does not yield rules that hold for one single case). With ``support="global"``
  it is applied over all the transactions.
* ``select_rules`` is the CBA builder (M1): it orders the rules by precedence (higher confidence, then higher support,
  then shorter antecedent), goes through the training data keeping each rule that classifies correctly some
  still-uncovered case, and cuts the list at the point where the total error (rules plus default class) is minimal.
* ``RuleClassifier`` encapsulates the two phases with a ``fit`` / ``predict`` / ``score`` interface; a case is
  classified with the first rule of the list that holds, and if none does, with the default class.

The rules have the same format as ``ARM.association_rules`` (and the antecedent items, a name without '-'), so
``significant_rules`` and ``prune_redundant_rules`` work on them. The consequent is the item ``<name>=<value>`` (the
'class' column stores the value as is).

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..FFIM import fpgrowth
from ..FIM._shared import key_to_itemset
from ..base import ParamsMixin
from ..preprocessing.timeseries import check_item_names
from ._apply import ItemTable, require_columns
from ._measures import MEASURES
from .redundancy import prune_unproductive_rules

_EPS = 1e-9


def _classes(labels: Union[Sequence[Any], np.ndarray, pd.Series]) -> Tuple[np.ndarray, List[Any]]:
    """Labels as a vector and sorted list of distinct classes."""
    values = np.asarray(labels, dtype=object)
    if pd.isna(values).any():
        raise ValueError("`labels` cannot have null values")
    classes = list(pd.unique(values))
    try:
        classes.sort()
    except TypeError:
        classes.sort(key=str)
    return values, classes


def _per_class(value: Union[float, Mapping[Any, float]], classes: List[Any], name: str, low: float, high: float,
               open_low: bool = False) -> Dict[Any, float]:
    """A threshold for each class: the same number for all or a dictionary {class: number} that covers all."""
    if isinstance(value, Mapping):
        missing = [label for label in classes if label not in value]
        if missing:
            raise ValueError("{} has no value for the classes {}".format(name, missing))
        result = {label: float(value[label]) for label in classes}
    else:
        result = {label: float(value) for label in classes}
    for label, number in result.items():
        if not (low < number if open_low else low <= number) or number > high:
            raise ValueError("{} must be in {}{}, {}], got: {}".format(name, "(" if open_low else "[", low, high, number))
    return result


def _check_inputs(items: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series]) -> None:
    if len(items) != len(labels):
        raise ValueError("`items` has {} rows and `labels` {}".format(len(items), len(labels)))
    check_item_names(items.columns)


def class_association_rules(items: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series],
                            min_supp: Union[float, Mapping[Any, float]] = 0.05,
                            min_conf: Union[float, Mapping[Any, float]] = 0.6, max_len: int = 4,
                            support: str = "class", class_name: Optional[str] = None,
                            min_count: int = 1) -> pd.DataFrame:
    """
    Mine the class association rules ``A => class=c``.

    Args:
        items (pandas.DataFrame): boolean (0/1) items, one row per case (``to_items`` converts numeric and
            categorical columns)
        labels: the class of each case, with as many values as rows
        min_supp (float or dict): minimum support of the rule, in (0, 1]: relative to the cases of the class
            (`support="class"`) or to all (`"global"`). A dictionary {class: value} sets one per class (all the
            classes must be in it)
        min_conf (float or dict): minimum confidence, in [0, 1], or one per class
        max_len (int): maximum number of items of the antecedent, >= 1
        support (str): 'class' or 'global' (see above)
        class_name (str, optional): name of the class variable in the consequent (by default, the name of `labels` or
            'class')
        min_count (int): minimum number of cases a rule must cover (hold together with its class), >= 1

    Returns:
        pandas.DataFrame with the format of ``association_rules`` (columns 'antecedents', 'consequents', 'antecedent
        support', 'consequent support', 'support', 'confidence', 'lift' and the other measures; the supports are
        always relative to all the cases) and the column 'class' with the value of the class, ordered by precedence.
        It records in ``rules.attrs`` the number of cases ('n_transactions') and that of candidate rules
        ('n_candidates': frequent antecedents of each class), which ``significant_rules`` uses
    """
    if support not in ("class", "global"):
        raise ValueError("support must be 'class' or 'global', got: '{}'".format(support))
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    _check_inputs(items, labels)
    table = ItemTable(items)
    values, classes = _classes(labels)
    suppByClass = _per_class(min_supp, classes, "min_supp", 0.0, 1.0, open_low=True)
    confByClass = _per_class(min_conf, classes, "min_conf", 0.0, 1.0)
    name = class_name or getattr(labels, "name", None) or "class"
    total = table.rows
    names = [str(column) for column in items.columns]
    rowSets = [frozenset(names[column] for column in np.flatnonzero(row)) for row in table.values]

    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support", "confidence",
               "lift"] + [measure for measure in MEASURES if measure not in ("confidence", "lift")] + ["class"]
    antecedentCount: Dict[frozenset, int] = {}
    rows: List[dict] = []
    candidates = 0
    for label in classes:
        inClass = np.flatnonzero(values == label)
        threshold = suppByClass[label] if support == "class" else suppByClass[label] * total / len(inClass)
        threshold = max(threshold, min_count / len(inClass))
        if threshold > 1 + _EPS:
            continue
        frequent = fpgrowth([rowSets[row] for row in inClass], min(threshold, 1.0))
        for key, within in frequent.items():
            antecedent = key_to_itemset(key)
            if not antecedent or len(antecedent) > max_len:
                continue
            candidates += 1
            if antecedent not in antecedentCount:
                antecedentCount[antecedent] = int(np.count_nonzero(table.holds(antecedent)))
            joint = int(round(within * len(inClass)))
            confidence = joint / antecedentCount[antecedent]
            if confidence < confByClass[label] - _EPS:
                continue
            supportJoint, supportA, supportC = joint / total, antecedentCount[antecedent] / total, len(inClass) / total
            entry: Dict[str, Any] = {"antecedents": antecedent, "consequents": frozenset(["{}={}".format(name, label)]),
                                     "antecedent support": supportA, "consequent support": supportC,
                                     "support": supportJoint, "class": label}
            for measure, function in MEASURES.items():
                entry[measure] = float(function(supportJoint, supportA, supportC))
            rows.append(entry)

    rules = pd.DataFrame(rows, columns=columns)
    if len(rules):
        rules["_length"] = [len(antecedent) for antecedent in rules["antecedents"]]
        rules = rules.sort_values(["confidence", "support", "_length"], ascending=[False, False, True],
                                  kind="mergesort").drop(columns="_length").reset_index(drop=True)
    rules.attrs["n_transactions"] = total
    rules.attrs["n_candidates"] = candidates
    return rules


def _precedence(rules: pd.DataFrame) -> pd.DataFrame:
    """Rules ordered by CBA precedence: confidence, support (decreasing) and length of the antecedent (increasing)."""
    length = np.array([len(antecedent) for antecedent in rules["antecedents"]], dtype=int)
    order = np.lexsort((length, -rules["support"].to_numpy(dtype=float), -rules["confidence"].to_numpy(dtype=float)))
    return rules.iloc[order]


def select_rules(rules: pd.DataFrame, items: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series]
                 ) -> Tuple[pd.DataFrame, Any]:
    """
    Construction of the CBA classifier (CBA-CB, M1) from the candidate rules and the training data.

    The rules are traversed by precedence; a rule is kept if it correctly classifies at least one of the cases not yet
    covered, and then it covers (removes) all the cases that satisfy its antecedent. After each kept rule the default
    class (the majority one among the uncovered cases) and the total error are recorded: the covered cases that the
    kept rules misclassify plus the uncovered ones that are not of that class. The final list goes up to the rule with
    the smallest total error (the shortest one if there is a tie).

    Args:
        rules (pandas.DataFrame): rules from ``class_association_rules``
        items, labels: training data (the same used for mining)

    Returns:
        (selected rules in order of application, default class)
    """
    require_columns(rules, "antecedents", "class", "confidence", "support")
    _check_inputs(items, labels)
    table = ItemTable(items)
    values, classes = _classes(labels)
    code = {label: position for position, label in enumerate(classes)}
    codes = np.array([code[value] for value in values], dtype=int)
    size = len(classes)

    def majority(subset: np.ndarray, fallback: int) -> int:
        return int(np.bincount(subset, minlength=size).argmax()) if len(subset) else fallback

    overall = majority(codes, 0) if len(codes) else 0
    ordered = _precedence(rules)
    remaining = np.arange(table.rows)
    trace: List[Tuple[int, int, int]] = []            # (posicion en `ordered`, clase por defecto, error total)
    ruleErrors = 0
    for position, (antecedent, label) in enumerate(zip(ordered["antecedents"], ordered["class"])):
        if len(remaining) == 0:
            break
        if label not in code:
            continue
        columns = np.array([_column(table, item) for item in antecedent], dtype=int)
        covers = table.values[remaining[:, None], columns].all(axis=1)
        covered = remaining[covers]
        correct = int(np.count_nonzero(codes[covered] == code[label]))
        if correct == 0:
            continue
        ruleErrors += len(covered) - correct
        remaining = remaining[~covers]
        default = majority(codes[remaining], overall)
        defaultErrors = len(remaining) - int(np.count_nonzero(codes[remaining] == default))
        trace.append((position, default, ruleErrors + defaultErrors))

    if not trace:
        return ordered.iloc[:0].reset_index(drop=True), classes[overall] if classes else None
    best = min(range(len(trace)), key=lambda index: (trace[index][2], index))
    kept = ordered.iloc[[step[0] for step in trace[:best + 1]]].reset_index(drop=True)
    return kept, classes[trace[best][1]]


def _column(table: ItemTable, item: str) -> int:
    if item not in table.columns:
        raise ValueError("item '{}' of the rules is not a column of `items`".format(item))
    return table.columns[item]


class RuleClassifier(ParamsMixin):
    """
    Classifier by class association rules (CBA): ``fit`` extracts the rules (``class_association_rules``) and keeps
    the useful ones (``select_rules``); ``predict`` applies the first rule that holds, and if none does, the default
    class.

    Constructor arguments: `min_supp`, `min_conf` (numbers or one per class), `max_len`, `support` and `min_count`, as in
    `class_association_rules`, and
    `min_improvement`: if not None, before selecting, the rules whose additional conditions do not improve by more than
    that value the confidence of a more general rule are discarded (``prune_unproductive_rules``), which gives
    shorter and less overfitted lists; by default it is not pruned, as in the original CBA.

    Attributes after `fit`: ``rules_`` (selected rules, in order of application), ``default_class_``, ``classes_`` and
    ``candidate_rules_`` (all the class rules before selection).
    """

    def __init__(self, min_supp: Union[float, Mapping[Any, float]] = 0.05,
                 min_conf: Union[float, Mapping[Any, float]] = 0.6, max_len: int = 4, support: str = "class",
                 min_improvement: Optional[float] = None, min_count: int = 1) -> None:
        self.min_improvement = min_improvement
        self.min_count = min_count
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.max_len = max_len
        self.support = support
        self.rules_ = pd.DataFrame()
        self.candidate_rules_ = pd.DataFrame()
        self._fitted = False
        self.default_class_: Any = None
        self.classes_: List[Any] = []

    def fit(self, items: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series]) -> "RuleClassifier":
        """Train with `items` (boolean, one case per row) and their class `labels`."""
        if len(items) == 0:
            raise ValueError("there are no cases to train with")
        self.candidate_rules_ = class_association_rules(items, labels, self.min_supp, self.min_conf, self.max_len,
                                                        self.support, min_count=self.min_count)
        candidates = self.candidate_rules_
        if self.min_improvement is not None:
            candidates = prune_unproductive_rules(candidates, self.min_improvement)
        self.rules_, self.default_class_ = select_rules(candidates, items, labels)
        self.classes_ = _classes(labels)[1]
        self._fitted = True
        return self

    def explain(self, items: pd.DataFrame) -> pd.DataFrame:
        """
        Like `predict`, with the reason: a ``DataFrame`` with the index of `items` and the columns 'prediction',
        'rule' (position in ``rules_`` of the rule that decides; -1 if it is the default class) and 'confidence' (that
        of the rule; NaN with the default class).
        """
        if not self._fitted:
            raise ValueError("the classifier is not trained: call fit first")
        table = ItemTable(items)
        prediction = np.empty(table.rows, dtype=object)
        rule = np.full(table.rows, -1, dtype=int)
        confidence = np.full(table.rows, np.nan)
        free = np.ones(table.rows, dtype=bool)
        for position, (antecedent, label, certainty) in enumerate(
                zip(self.rules_["antecedents"], self.rules_["class"], self.rules_["confidence"])):
            fires = free & table.holds(antecedent)
            prediction[fires], rule[fires], confidence[fires] = label, position, certainty
            free &= ~fires
        prediction[free] = self.default_class_
        return pd.DataFrame({"prediction": pd.Series(prediction, index=items.index).infer_objects(), "rule": rule,
                             "confidence": confidence}, index=items.index)

    def predict(self, items: pd.DataFrame) -> pd.Series:
        """Class of each row of `items`, with its index."""
        return self.explain(items)["prediction"].rename("class")

    def score(self, items: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series]) -> float:
        """Accuracy (share of correctly classified cases) over `items` and their class `labels`."""
        _check_inputs(items, labels)
        if len(items) == 0:
            return float("nan")
        return float(np.mean(self.predict(items).to_numpy(dtype=object) == np.asarray(labels, dtype=object)))
