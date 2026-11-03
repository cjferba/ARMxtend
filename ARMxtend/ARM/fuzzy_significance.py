# -*- coding: utf-8 -*-
"""
Statistical significance of FUZZY rules, rule by rule, with a permutation test.

The tests of ``ARM.significance`` (Fisher, chi-squared) need integer frequencies and do not work with membership
degrees. Here, for each rule ``A => B``:

* the membership vector of the antecedent ``a`` and of the consequent ``b`` is computed in each transaction (the
  minimum, the t-norm, of the degrees of its items; an absent item is worth 0);
* the statistic is, by default, the **scalar-cardinality confidence** ``sum(min(a, b)) / sum(a)``. With
  ``statistic="fconf"`` it is the confidence aggregated by alpha-cuts (``FConf``) of ``FIM.FARE`` (RL-theory): the
  crisp confidence of the cut ``min(a, b) >= alpha_i`` over ``a >= alpha_i``, weighted by the width of each level, with
  the convention 0/0 := 1;
* the null hypothesis is that ``b`` does not depend on ``a``: ``b`` is shuffled among the transactions (it keeps the
  distribution of each side and breaks their relation) and the p-value is the fraction of shuffles whose statistic
  reaches the observed one, ``(1 + #shuffled >= observed) / (1 + n_permutations)`` (Phipson and Smyth, 2010),
  one-sided test.

The p-value is corrected for multiple testing with ``adjust_pvalues``. With ``n_permutations`` shuffles the minimum
p-value is ``1 / (n_permutations + 1)``: for a Bonferroni correction with ``m`` tests to be able to reject at level
``alpha`` at least ``m / alpha`` permutations are needed.

@author Carlos Fernandez-Basso
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..FIM._shared import alpha_levels, alpha_weights
from .significance import adjust_pvalues

STATISTICS = ("cardinality", "fconf")

FuzzyTransactions = Union[pd.DataFrame, Sequence[Iterable[Tuple[str, float]]]]

_TOL = 1e-12


def _degrees(transactions: FuzzyTransactions, items: Iterable[str]) -> Dict[str, np.ndarray]:
    """Vector of degrees (one per transaction, 0 if missing) of each item of `items`; the highest degree if repeated."""
    wanted = set(items)
    if isinstance(transactions, pd.DataFrame):
        values = transactions.to_numpy(dtype=float)
        position = {str(name): k for k, name in enumerate(transactions.columns)}
        missing = wanted - set(position)
        columns = {item: values[:, position[item]].copy() for item in wanted if item in position}
    else:
        rows = list(transactions)
        columns = {item: np.zeros(len(rows)) for item in wanted}
        missing = set()
        for row, transaction in enumerate(rows):
            for item, degree in transaction:
                if item in columns:
                    columns[item][row] = max(columns[item][row], float(degree))
    if missing:
        raise ValueError("the items {} of the rules are not in the transactions".format(sorted(missing)))
    for item, column in columns.items():
        if np.any((column < 0) | (column > 1)) or np.any(np.isnan(column)):
            raise ValueError("membership degrees must be in [0, 1] (item '{}')".format(item))
    return columns


def _side(columns: Dict[str, np.ndarray], items: Iterable[str], rows: int) -> np.ndarray:
    result = np.ones(rows)
    for item in items:
        result = np.minimum(result, columns[item])
    return result


def _fconf(a: np.ndarray, joint: np.ndarray, levels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    FConf of each row of `joint` (one shuffle per row): sum over levels of weight * |joint >= alpha| / |a >= alpha|,
    with 0/0 := 1 as in ``FIM.FARE``.
    """
    result = np.zeros(joint.shape[0])
    for level, weight in zip(levels, weights):
        denominator = np.count_nonzero(a >= level - _TOL)
        if denominator == 0:
            result += weight
        else:
            result += weight * (joint >= level - _TOL).sum(axis=1) / denominator
    return result


def fuzzy_rule_significance(rules: pd.DataFrame, transactions: FuzzyTransactions, n_permutations: int = 999,
                            random_state: Union[None, int, np.random.Generator] = None,
                            statistic: str = "cardinality", num_alpha: int = 10) -> pd.DataFrame:
    """
    Add to the fuzzy rules the ``fuzzy_confidence`` statistic and the ``p_value`` of the permutation test.

    Args:
        rules (pandas.DataFrame): rules with 'antecedents' and 'consequents' (like those of
            ``FIM.FARE.fuzzy_association_rules``)
        transactions: the transactions they were mined with, as a list of lists of (item, degree) pairs or as a table
            of degrees (one row per transaction, one column per item; see
            ``preprocessing.timeseries.to_fuzzy_transactions``)
        n_permutations (int): number of shuffles (999 by default; more gives finer but slower p-values)
        random_state: seed or ``numpy.random.Generator``
        statistic (str): 'cardinality' (scalar-cardinality confidence) or 'fconf' (confidence aggregated by
            alpha-cuts, see the module docstring)
        num_alpha (int): number of alpha-cuts of the 'fconf' statistic (use the one the rules were mined with)

    Returns:
        Copy of `rules` with the columns 'fuzzy_confidence' and 'p_value'; ``attrs`` keeps those of `rules` and adds
        'n_permutations' and 'n_transactions'.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if statistic not in STATISTICS:
        raise ValueError("statistic must be one of {}, got: '{}'".format(STATISTICS, statistic))
    levels = alpha_levels(num_alpha)
    weights = alpha_weights(levels)
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    result = rules.copy()
    result.attrs.update(rules.attrs)
    rows = len(transactions)
    result.attrs["n_permutations"] = n_permutations
    result.attrs["n_transactions"] = rows
    if not len(rules):
        result["fuzzy_confidence"] = pd.Series(dtype=float)
        result["p_value"] = pd.Series(dtype=float)
        return result
    if rows == 0:
        raise ValueError("there are no transactions")
    columns = _degrees(transactions, [item for sides in zip(rules["antecedents"], rules["consequents"])
                                      for side in sides for item in side])
    permutations = np.argsort(rng.random((n_permutations, rows)), axis=1)
    confidences, p_values = [], []
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        a, b = _side(columns, antecedent, rows), _side(columns, consequent, rows)
        total = a.sum()
        if total <= 0:
            confidences.append(0.0 if statistic == "cardinality" else 1.0)
            p_values.append(1.0)
            continue
        if statistic == "cardinality":
            observed = np.minimum(a, b).sum()
            shuffled = np.minimum(a[None, :], b[permutations]).sum(axis=1)
            confidences.append(observed / total)
        else:
            observed = _fconf(a, np.minimum(a, b)[None, :], levels, weights)[0]
            shuffled = _fconf(a, np.minimum(a[None, :], b[permutations]), levels, weights)
            confidences.append(float(observed))
        p_values.append((1 + int((shuffled >= observed - _TOL).sum())) / (n_permutations + 1))
    result["fuzzy_confidence"] = confidences
    result["p_value"] = p_values
    return result


def fuzzy_significant_rules(rules: pd.DataFrame, transactions: FuzzyTransactions, alpha: float = 0.05,
                            correction: str = "fdr_bh", n_permutations: int = 999, n_tests: Optional[int] = None,
                            random_state: Union[None, int, np.random.Generator] = None,
                            statistic: str = "cardinality", num_alpha: int = 10) -> pd.DataFrame:
    """
    The significant fuzzy rules after correcting for multiple testing (see `fuzzy_rule_significance`).

    `n_tests` is the number of hypotheses of the family; by default ``rules.attrs['n_candidates']`` (recorded by
    ``FIM.FARE.fuzzy_association_rules``: the antecedent/consequent partitions examined) or, if it is not there, the
    number of rules given, which UNDERESTIMATES the family. It returns the rules with 'p_adjusted' <= alpha, sorted by
    adjusted p-value.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1), got: {}".format(alpha))
    scored = fuzzy_rule_significance(rules, transactions, n_permutations, random_state, statistic, num_alpha)
    family = n_tests if n_tests is not None else rules.attrs.get("n_candidates")
    family = max(int(family), len(scored)) if family is not None else len(scored)
    scored["p_adjusted"] = adjust_pvalues(scored["p_value"].to_numpy(), correction, n_tests=family)
    kept = scored[scored["p_adjusted"] <= alpha].sort_values(["p_adjusted", "p_value"], kind="mergesort")
    result = kept.reset_index(drop=True)
    result.attrs.update(scored.attrs)
    result.attrs["n_tests"] = family
    return result
