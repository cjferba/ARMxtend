# -*- coding: utf-8 -*-
"""
Statistical significance of SEQUENTIAL rules ``X => Y``.

A sequential rule can look strong only because X and Y are frequent in the same sequences, with no order between
them. The null hypothesis tested here is exactly that: *the order of the elements inside each sequence carries no
information*. Each permutation shuffles the order of the elements of every sequence (the sets of items in each
sequence are kept, so the supports of X and Y as baskets are unchanged) and counts again in how many sequences
the rule is valid. The p-value of a rule is

    p = (1 + #{permutations with at least as many valid sequences as observed}) / (1 + n_permutations),

and the p-values can then be corrected for multiple testing (``ARM.significance.adjust_pvalues``).

@author Carlos Fernandez-Basso
"""
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..FIM.sequential import _normalize
from .significance import adjust_pvalues

_ABSENT_FIRST = 10 ** 9


def _first_last(database: List[List[frozenset]], column: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    first = np.full((len(database), len(column)), _ABSENT_FIRST, dtype=np.int64)
    last = np.full((len(database), len(column)), -1, dtype=np.int64)
    for row, sequence in enumerate(database):
        for position, element in enumerate(sequence):
            for item in element:
                cell = column[item]
                if first[row, cell] == _ABSENT_FIRST:
                    first[row, cell] = position
                last[row, cell] = position
    return first, last


def sequential_rule_significance(sequences: Iterable[Iterable[Union[str, Iterable[str]]]], rules: pd.DataFrame,
                                 n_permutations: int = 200, correction: str = "fdr_bh",
                                 random_state: Union[None, int, np.random.Generator] = None) -> pd.DataFrame:
    """
    Permutation p-value of each sequential rule against the null of "the order of the elements is irrelevant".

    Args:
        sequences: the sequences the `rules` were mined from (see ``FIM.sequential``)
        rules (pandas.DataFrame): output of ``ARM.sequential_rules`` (columns 'antecedents' and 'consequents')
        n_permutations (int): number of shuffles of the element order, >= 1
        correction (str): multiple-testing correction applied to the p-values (see ``ARM.significance.adjust_pvalues``)
        random_state: seed or ``numpy.random.Generator``

    Returns:
        A copy of `rules` with the extra columns 'valid_sequences' (observed), 'null_mean' (average over the
        permutations), 'p_permutation' and 'p_adjusted'.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    database = _normalize(sequences)
    result = rules.copy()
    if len(rules) == 0 or not database:
        for name in ("valid_sequences", "null_mean", "p_permutation", "p_adjusted"):
            result[name] = pd.Series(dtype=float)
        return result
    vocabulary = sorted({item for sequence in database for element in sequence for item in element})
    column = {item: index for index, item in enumerate(vocabulary)}
    unknown = sorted({item for side in ("antecedents", "consequents") for items in rules[side] for item in items
                      if item not in column})
    if unknown:
        raise ValueError("the items {} of the rules are not in the sequences".format(unknown))
    xs = [[column[item] for item in items] for items in rules["antecedents"]]
    ys = [[column[item] for item in items] for items in rules["consequents"]]
    rng = np.random.default_rng(random_state)

    def valid_counts(first: np.ndarray, last: np.ndarray) -> np.ndarray:
        return np.array([np.count_nonzero(first[:, x].max(axis=1) < last[:, y].min(axis=1))
                         for x, y in zip(xs, ys)])

    observed = valid_counts(*_first_last(database, column))
    exceed = np.zeros(len(rules), dtype=np.int64)
    total = np.zeros(len(rules), dtype=np.float64)
    for _ in range(n_permutations):
        shuffled = [[sequence[i] for i in rng.permutation(len(sequence))] for sequence in database]
        counts = valid_counts(*_first_last(shuffled, column))
        exceed += counts >= observed
        total += counts
    p_values = (1.0 + exceed) / (1.0 + n_permutations)
    result["valid_sequences"] = observed
    result["null_mean"] = total / n_permutations
    result["p_permutation"] = p_values
    result["p_adjusted"] = adjust_pvalues(p_values, correction)
    return result


def significant_sequential_rules(sequences: Iterable[Iterable[Union[str, Iterable[str]]]], rules: pd.DataFrame,
                                 alpha: float = 0.05, n_permutations: int = 200, correction: str = "fdr_bh",
                                 random_state: Union[None, int, np.random.Generator] = None) -> pd.DataFrame:
    """The rules of `rules` whose corrected permutation p-value is <= `alpha` (see ``sequential_rule_significance``)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1), got: {}".format(alpha))
    table = sequential_rule_significance(sequences, rules, n_permutations, correction, random_state)
    return table[table["p_adjusted"] <= alpha].reset_index(drop=True)
