# -*- coding: utf-8 -*-
"""
Classification with FUZZY class association rules: ``A => class=c`` where the antecedent ``A`` is a conjunction of
fuzzy items ("temperature is HIGH and humidity is LOW") that holds in each case to a degree in [0, 1]. It follows the
fuzzy rule-based classification systems of Ishibuchi and Yamamoto (2005) in their simplest form:

    Ishibuchi, H., Yamamoto, T. (2005). "Rule weight specification in fuzzy rule-based classification systems".
    IEEE Transactions on Fuzzy Systems, 13(4), 428-435.

* The degree to which ``A`` holds in a case is the t-norm (``min`` or product) of the degrees of its items. Supports
  and confidences use the scalar cardinality (sum of degrees): for the rule ``A => c``, with ``mu_A`` the degrees of
  the antecedent, ``support = sum(mu_A over the cases of class c) / N`` and ``confidence = sum(mu_A over the cases of
  class c) / sum(mu_A over all the cases)``. They are anti-monotone with the two t-norms, so the search is exact.
* As in ``class_association_rules`` the minimum support is applied within each class, so rare classes get rules.
* The **rule weight** is the certainty grade of Ishibuchi and Yamamoto: the confidence of the rule minus the mean
  confidence of the other classes for the same antecedent. Rules with a weight <= 0 are not kept.
* A case is classified by the class of the rule with the highest ``degree x weight`` (single winner, the default) or
  by the class with the highest SUM of ``degree x weight`` over its rules (``inference="sum"``). If no rule fires, the
  majority class is predicted.

The rules have the format of ``ARM.association_rules`` plus 'class' and 'weight'. Item names must not contain '-'.

@author Carlos Fernandez-Basso
"""
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..base import ParamsMixin
from ..preprocessing.timeseries import check_item_names
from ._rulerows import RULE_COLUMNS, rule_row
from .classification import _classes

_EPS = 1e-9
TNORMS = ("min", "product")
INFERENCES = ("winner", "sum")


def _check_degrees(degrees: pd.DataFrame) -> np.ndarray:
    values = degrees.to_numpy(dtype=float)
    if np.isnan(values).any() or (values < 0).any() or (values > 1).any():
        raise ValueError("membership degrees must be in [0, 1]")
    check_item_names(degrees.columns)
    return values


def _activation(values: np.ndarray, columns: Sequence[int], tnorm: str) -> np.ndarray:
    selected = values[:, list(columns)]
    return selected.min(axis=1) if tnorm == "min" else selected.prod(axis=1)


def fuzzy_class_rules(degrees: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series],
                      min_supp: float = 0.05, min_conf: float = 0.6, max_len: int = 3, tnorm: str = "min",
                      class_name: Optional[str] = None) -> pd.DataFrame:
    """
    Mine the fuzzy class association rules ``A => class=c`` with their weight (see the module docstring).

    Args:
        degrees (pandas.DataFrame): membership degrees in [0, 1], one row per case and one column per fuzzy item
            (``preprocessing.FuzzyLib`` or ``timeseries.to_fuzzy_transactions`` produce them)
        labels: the class of each case
        min_supp (float): minimum support within the class (scalar cardinality), in (0, 1]
        min_conf (float): minimum confidence, in [0, 1]
        max_len (int): maximum number of items of the antecedent, >= 1
        tnorm (str): 'min' or 'product'
        class_name (str, optional): name of the class variable in the consequent

    Returns:
        pandas.DataFrame with the columns of ``association_rules`` (supports relative to all the cases, in scalar
        cardinality), 'class' and 'weight', by decreasing weight x confidence. ``attrs['n_transactions']`` and
        ``attrs['n_candidates']`` are recorded as in ``class_association_rules``
    """
    if tnorm not in TNORMS:
        raise ValueError("tnorm must be one of {}, got: '{}'".format(TNORMS, tnorm))
    if not 0 < min_supp <= 1 or not 0 <= min_conf <= 1:
        raise ValueError("min_supp must be in (0, 1] and min_conf in [0, 1]")
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    if len(labels) != len(degrees):
        raise ValueError("`degrees` has {} rows and `labels` {}".format(len(degrees), len(labels)))
    values = _check_degrees(degrees)
    labelValues, classes = _classes(labels)
    name = class_name or getattr(labels, "name", None) or "class"
    total = len(values)
    names = [str(column) for column in degrees.columns]
    membership = np.stack([labelValues == label for label in classes], axis=1).astype(float)     # (N, C)
    sizes = membership.sum(axis=0)

    def stats(columns: Tuple[int, ...]) -> Tuple[np.ndarray, float]:
        mu = _activation(values, columns, tnorm)
        return mu @ membership, float(mu.sum())

    kept: Dict[Tuple[int, ...], Tuple[np.ndarray, float]] = {}
    level: List[Tuple[int, ...]] = []
    for column in range(values.shape[1]):
        perClass, whole = stats((column,))
        if np.any(perClass / sizes >= min_supp - _EPS):
            kept[(column,)] = (perClass, whole)
            level.append((column,))
    candidates = len(level)
    for size in range(2, max_len + 1):
        known = set(level)
        candidatesOfLevel = sorted({tuple(sorted(set(a) | set(b))) for a, b in combinations(level, 2)
                                    if len(set(a) | set(b)) == size})
        level = []
        for itemset in candidatesOfLevel:
            if not all(subset in known for subset in combinations(itemset, size - 1)):
                continue
            perClass, whole = stats(itemset)
            if np.any(perClass / sizes >= min_supp - _EPS):
                kept[itemset] = (perClass, whole)
                level.append(itemset)
                candidates += 1
    rows: List[Dict[str, Any]] = []
    for itemset, (perClass, whole) in kept.items():
        if whole <= 0:
            continue
        confidences = perClass / whole
        for index, label in enumerate(classes):
            if perClass[index] / sizes[index] < min_supp - _EPS or confidences[index] < min_conf - _EPS:
                continue
            others = np.delete(confidences, index)
            weight = float(confidences[index] - (others.mean() if len(others) else 0.0))
            if weight <= _EPS:
                continue
            entry = rule_row(frozenset(names[c] for c in itemset), frozenset(["{}={}".format(name, label)]),
                             float(perClass[index] / total), whole / total, float(sizes[index] / total))
            entry["class"], entry["weight"] = label, weight
            rows.append(entry)
    rules = pd.DataFrame(rows, columns=RULE_COLUMNS + ["class", "weight"])
    if len(rules):
        rules = rules.assign(_key=rules["weight"] * rules["confidence"]).sort_values(
            "_key", ascending=False, kind="mergesort").drop(columns="_key").reset_index(drop=True)
    rules.attrs["n_transactions"] = total
    rules.attrs["n_candidates"] = candidates
    return rules


class FuzzyRuleClassifier(ParamsMixin):
    """
    Classifier by fuzzy class association rules: ``fit`` mines them (``fuzzy_class_rules``) and keeps the best
    `max_rules_per_class` of each class; ``predict`` applies them with the chosen inference (see the module docstring).

    Constructor arguments: `min_supp`, `min_conf`, `max_len`, `tnorm` (as in `fuzzy_class_rules`), `inference`
    ('winner' or 'sum') and `max_rules_per_class` (None keeps all the rules).

    Attributes after `fit`: ``rules_``, ``classes_`` and ``default_class_`` (the majority class).
    """

    def __init__(self, min_supp: float = 0.05, min_conf: float = 0.6, max_len: int = 3, tnorm: str = "min",
                 inference: str = "winner", max_rules_per_class: Optional[int] = None) -> None:
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.max_len = max_len
        self.tnorm = tnorm
        self.inference = inference
        self.max_rules_per_class = max_rules_per_class
        self.rules_ = pd.DataFrame()
        self.classes_: List[Any] = []
        self.default_class_: Any = None
        self._fitted = False

    def fit(self, degrees: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series]) -> "FuzzyRuleClassifier":
        """Train with the membership `degrees` (one row per case) and their class `labels`."""
        if self.inference not in INFERENCES:
            raise ValueError("inference must be one of {}, got: '{}'".format(INFERENCES, self.inference))
        if self.max_rules_per_class is not None and self.max_rules_per_class < 1:
            raise ValueError("max_rules_per_class must be >= 1")
        if len(degrees) == 0:
            raise ValueError("there are no cases to train with")
        rules = fuzzy_class_rules(degrees, labels, self.min_supp, self.min_conf, self.max_len, self.tnorm)
        if self.max_rules_per_class is not None and len(rules):
            rules = rules.groupby("class", sort=False).head(self.max_rules_per_class).reset_index(drop=True)
        values, self.classes_ = _classes(labels)
        counts = [int(np.count_nonzero(values == label)) for label in self.classes_]
        self.default_class_ = self.classes_[int(np.argmax(counts))]
        self.rules_ = rules
        self._columns = [str(column) for column in degrees.columns]
        self._fitted = True
        return self

    def decision_scores(self, degrees: pd.DataFrame) -> pd.DataFrame:
        """Score of each class for each case (one column per class): the best or the sum of ``degree x weight``."""
        if not self._fitted:
            raise ValueError("the classifier is not trained: call fit first")
        names = [str(column) for column in degrees.columns]
        missing = [name for name in self._columns if name not in names]
        if missing:
            raise ValueError("`degrees` is missing the columns {}".format(missing))
        values = _check_degrees(degrees)
        scores = np.zeros((len(values), len(self.classes_)))
        position = {label: index for index, label in enumerate(self.classes_)}
        for antecedent, label, weight in zip(self.rules_["antecedents"], self.rules_["class"], self.rules_["weight"]):
            columns = [names.index(item) for item in antecedent]
            fired = _activation(values, columns, self.tnorm) * weight
            column = position[label]
            scores[:, column] = np.maximum(scores[:, column], fired) if self.inference == "winner" \
                else scores[:, column] + fired
        return pd.DataFrame(scores, index=degrees.index, columns=self.classes_)

    def predict(self, degrees: pd.DataFrame) -> pd.Series:
        """Class of each case; the majority class if no rule fires."""
        scores = self.decision_scores(degrees)
        best = scores.to_numpy().argmax(axis=1)
        silent = scores.to_numpy().max(axis=1) <= 0
        prediction = np.empty(len(scores), dtype=object)
        for index, (choice, quiet) in enumerate(zip(best, silent)):
            prediction[index] = self.default_class_ if quiet else self.classes_[choice]
        return pd.Series(prediction, index=degrees.index, name="class").infer_objects()

    def score(self, degrees: pd.DataFrame, labels: Union[Sequence[Any], np.ndarray, pd.Series]) -> float:
        """Accuracy over `degrees` and their class `labels`."""
        if len(degrees) != len(labels):
            raise ValueError("`degrees` has {} rows and `labels` {}".format(len(degrees), len(labels)))
        if len(degrees) == 0:
            return float("nan")
        return float(np.mean(self.predict(degrees).to_numpy(dtype=object) == np.asarray(labels, dtype=object)))
