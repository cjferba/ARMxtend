# -*- coding: utf-8 -*-
"""
Statistical significance of association rules: from "this rule passes the thresholds" to "this rule is stronger than
chance would give".

Mining many rules with support and confidence thresholds produces spurious rules -- the more are tested, the more
appear by chance -- and on autocorrelated temporal data even more. This module offers three tools:

* Per-rule TESTS on the 2x2 table of their frequencies (``rule_significance``): Fisher's exact test (tail of the
  hypergeometric) or Pearson's chi-squared. They assume independent transactions.
* MULTIPLE-TESTING CORRECTION (``adjust_pvalues``, ``significant_rules``): Bonferroni, Holm (Holm, 1979),
  Benjamini-Hochberg (1995) and Benjamini-Yekutieli (2001). The number of hypotheses is that of the candidate rules
  that were tested, not only of those shown (``candidate_rule_count``; ``association_rules`` and
  ``temporal_association_rules`` record it in ``rules.attrs["n_candidates"]``).
* PERMUTATION TESTS (``permutation_null``, ``permutation_p_values``): the data are re-mined after shuffling them in a
  way that keeps the frequency of each item but breaks the relations between them, and each rule is compared with the
  distribution of the MAXIMUM of the measure among all the rules of the shuffled data (Westfall & Young, 1993), which
  controls the family-wise error rate taking into account that the rules were selected by their own values
  (Webb, 2007). With time series the shuffling uses circular shifts (``null="shift"``), which keep the
  autocorrelation of each series.

Besides, ``confidence_intervals`` gives Wilson or bootstrap intervals for support, confidence and lift.

The per-rule tests require integer frequencies (crisp rules); for fuzzy rules use ``ARM.fuzzy_significant_rules``
(rule-by-rule permutation test) or the permutation tests of this module, which work with any miner.

@author Carlos Fernandez-Basso
"""
import math
from itertools import combinations
from statistics import NormalDist
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


Miner = Callable[[pd.DataFrame], pd.DataFrame]

CORRECTIONS = ("bonferroni", "holm", "fdr_bh", "fdr_by")
TESTS = ("fisher", "chi2")
ALTERNATIVES = ("greater", "less", "two-sided")
NULLS = ("shift", "permute")

_CHUNK = 4096


# ---------------------------------------------------------------- tests 2x2

def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_upper(k: int, draws: int, successes: int, population: int) -> float:
    """
    P(X >= k) for X ~ Hypergeometric(population, successes, draws): the upper tail of Fisher's exact test. It is
    summed from `k` upwards with the recurrence between consecutive terms, in blocks, and stops as soon as the terms,
    already decreasing, no longer count: the cost depends on the standard deviation of X and not on the size of the
    population.
    """
    low = max(0, draws + successes - population)
    high = min(draws, successes)
    if k <= low:
        return 1.0
    if k > high:
        return 0.0

    def log_ratio(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """log de pmf(x + 1) / pmf(x)."""
        return (np.log((successes - x) * (draws - x))
                - np.log((x + 1.0) * (population - successes - draws + x + 1.0)))

    logPmf = float(_log_choose(successes, k) + _log_choose(population - successes, draws - k)
                   - _log_choose(population, draws))
    total = 0.0
    start = k
    while True:
        steps = min(_CHUNK, high - start + 1)
        logs = logPmf + np.concatenate([[0.0], np.cumsum(log_ratio(start + np.arange(steps - 1, dtype=float)))])
        terms = np.exp(logs)
        total += float(terms.sum())
        start += steps
        if start > high:
            break
        logPmf = float(logs[-1] + log_ratio(float(start - 1)))
        if logPmf < logs[-1] and terms[-1] < 1e-18 * total:
            break
    return min(total, 1.0)


def _fisher_pvalue(n_ac: int, n_a: int, n_c: int, n: int, alternative: str) -> float:
    greater = _hypergeom_upper(n_ac, n_a, n_c, n)
    if alternative == "greater":
        return greater
    # P(X <= k) = P(Y >= n_a - k) with Y = draws - X ~ Hypergeometric(n, n - n_c, n_a)
    less = _hypergeom_upper(n_a - n_ac, n_a, n - n_c, n)
    if alternative == "less":
        return less
    return min(1.0, 2.0 * min(greater, less))


def _chi2_pvalue(n_ac: int, n_a: int, n_c: int, n: int, alternative: str) -> float:
    rowA, rowNotA = n_a, n - n_a
    colC, colNotC = n_c, n - n_c
    if min(rowA, rowNotA, colC, colNotC) == 0:
        return 1.0
    expected = n_a * n_c / n
    deviation = n_ac - expected
    statistic = float(n) ** 3 * deviation ** 2 / (float(rowA) * rowNotA * colC * colNotC)   # n (ad - bc)^2 / (R1 R2 C1 C2)
    two = math.erfc(math.sqrt(statistic / 2.0))
    if alternative == "two-sided":
        return two
    if deviation == 0:
        return 1.0
    towards = deviation > 0
    return two / 2.0 if (towards == (alternative == "greater")) else 1.0 - two / 2.0


def _rule_counts(rules: pd.DataFrame, n_transactions: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = n_transactions if n_transactions is not None else rules.attrs.get("n_transactions")
    if n is None:
        raise ValueError("the number of transactions is missing: pass `n_transactions` (the rules of "
                         "temporal_association_rules already carry it in rules.attrs)")
    n = int(n)
    if n < 1:
        raise ValueError("n_transactions must be >= 1, got: {}".format(n))
    missing = [column for column in ("support", "antecedent support", "consequent support") if column not in rules.columns]
    if missing:
        raise ValueError("the rules need the columns {}: this test is for crisp rules "
                         "(for fuzzy rules use permutation_null)".format(missing))
    scaled = [rules[column].to_numpy(dtype=float) * n for column in ("support", "antecedent support", "consequent support")]
    counts = [np.rint(values) for values in scaled]
    if any(np.any(np.abs(values - rounded) > 1e-3) for values, rounded in zip(scaled, counts)):
        raise ValueError("the supports are not multiples of 1/{}: n_transactions does not match the "
                         "transactions the rules come from (or they are fuzzy rules)".format(n))
    nAC, nA, nC = (rounded.astype(np.int64) for rounded in counts)
    if np.any(nAC > np.minimum(nA, nC)) or np.any(nA > n) or np.any(nC > n) or np.any(nAC < np.maximum(0, nA + nC - n)):
        raise ValueError("the frequencies of the rules are inconsistent with n_transactions={}".format(n))
    return nAC, nA, nC, n


def rule_significance(rules: pd.DataFrame, n_transactions: Optional[int] = None, method: str = "fisher",
                      alternative: str = "greater") -> pd.DataFrame:
    """
    Add to each rule the p-value of its association.

    For each rule ``A => C`` the null hypothesis that A and C are independent is tested with the 2x2 table
    (A / not A) x (C / not C) rebuilt from the supports and the number of transactions.

    Args:
        rules (pandas.DataFrame): crisp rules, with the columns 'support', 'antecedent support' and
            'consequent support' (the output of ``ARM.association_rules`` or of ``ARM.temporal_association_rules``)
        n_transactions (int, optional): number of transactions the supports come from; by default
            ``rules.attrs["n_transactions"]`` (recorded by the temporal rules; for the others it must be given). If
            the supports are not multiples of 1/n, ``ValueError`` is raised
        method (str): 'fisher' (exact test, default) or 'chi2' (Pearson, without continuity correction; it
            approximates well with expected frequencies >= 5)
        alternative (str): 'greater' (positive association: A favors C, which is what matters in a rule; default),
            'less' or 'two-sided' (twice the smaller one-sided p-value)

    Returns:
        Copy of `rules` with the column 'p_value'. It assumes independent transactions; with time series use
        ``permutation_null``.
    """
    if method not in TESTS:
        raise ValueError("method must be one of {}, got: '{}'".format(TESTS, method))
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative must be one of {}, got: '{}'".format(ALTERNATIVES, alternative))
    nAC, nA, nC, n = _rule_counts(rules, n_transactions)
    test = _fisher_pvalue if method == "fisher" else _chi2_pvalue
    result = rules.copy()
    result["p_value"] = [test(int(ac), int(a), int(c), n, alternative) for ac, a, c in zip(nAC, nA, nC)]
    result.attrs["n_transactions"] = n
    return result


# ------------------------------------------------------- pruebas multiples

def adjust_pvalues(p_values: Union[Iterable[float], np.ndarray], method: str = "fdr_bh",
                   n_tests: Optional[int] = None) -> np.ndarray:
    """
    Correct p-values for multiple testing.

    Args:
        p_values: the p-values of the hypotheses tested
        method (str): 'bonferroni' and 'holm' control the probability of any false positive (FWER); 'fdr_bh'
            (Benjamini-Hochberg, for independent tests or positive dependence) and 'fdr_by' (Benjamini-Yekutieli,
            valid under any dependence) control the expected proportion of false discoveries (FDR)
        n_tests (int, optional): total number of hypotheses of the family, if larger than ``len(p_values)``: the
            ones not listed are treated as not rejected (p-value 1); by default, ``len(p_values)``

    Returns:
        numpy.ndarray with the adjusted p-values, in the input order and in [0, 1].
    """
    if method not in CORRECTIONS:
        raise ValueError("method must be one of {}, got: '{}'".format(CORRECTIONS, method))
    p = np.asarray(list(p_values) if not isinstance(p_values, np.ndarray) else p_values, dtype=float)
    count = len(p)
    total = count if n_tests is None else int(n_tests)
    if total < count:
        raise ValueError("n_tests ({}) cannot be smaller than the number of p-values ({})".format(total, count))
    if count == 0:
        return p.copy()
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("the p-values must be in [0, 1]")
    order = np.argsort(p, kind="mergesort")
    sortedP = p[order]
    ranks = np.arange(1, count + 1, dtype=float)
    if method == "bonferroni":
        adjusted = sortedP * total
    elif method == "holm":
        adjusted = np.maximum.accumulate((total - ranks + 1.0) * sortedP)
    else:
        factor = 1.0 if method == "fdr_bh" else float(np.sum(1.0 / np.arange(1, total + 1)))
        adjusted = np.minimum.accumulate((sortedP * total * factor / ranks)[::-1])[::-1]
    result = np.empty(count)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def candidate_rule_count(itemsets: pd.DataFrame) -> int:
    """
    Number of candidate rules that come out of some frequent itemsets: each itemset of size k >= 2 is split in
    ``2^k - 2`` ways into antecedent and consequent. It is the natural size of the family of hypotheses for
    `adjust_pvalues` (``n_tests``).

    Args:
        itemsets (pandas.DataFrame): with the column 'itemsets' (like the input of ``ARM.association_rules``)
    """
    return int(sum(2 ** len(items) - 2 for items in itemsets["itemsets"] if len(items) >= 2))


def significant_rules(rules: pd.DataFrame, alpha: float = 0.05, correction: str = "fdr_bh",
                      n_transactions: Optional[int] = None, method: str = "fisher", alternative: str = "greater",
                      n_tests: Optional[int] = None) -> pd.DataFrame:
    """
    Keep the significant rules after correcting for multiple testing.

    Args:
        rules (pandas.DataFrame): crisp rules (see `rule_significance`)
        alpha (float): significance level, in (0, 1): FWER or FDR depending on `correction`
        correction (str): see `adjust_pvalues`
        n_transactions, method, alternative: see `rule_significance`
        n_tests (int, optional): number of hypotheses of the family. By default ``rules.attrs["n_candidates"]`` (the
            number of candidate rules recorded by ``association_rules`` and ``temporal_association_rules``) or, if it
            is not there, the number of rules given -- which UNDERESTIMATES the family, because the rules were chosen
            for passing thresholds; use `candidate_rule_count` or a permutation test

    Returns:
        pandas.DataFrame with the rules whose 'p_adjusted' <= alpha, with the columns 'p_value' and 'p_adjusted',
        sorted by adjusted p-value.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1), got: {}".format(alpha))
    family = n_tests if n_tests is not None else rules.attrs.get("n_candidates")
    scored = rule_significance(rules, n_transactions, method, alternative)
    family = max(int(family), len(scored)) if family is not None else len(scored)
    scored["p_adjusted"] = adjust_pvalues(scored["p_value"].to_numpy(), correction, n_tests=family)
    kept = scored[scored["p_adjusted"] <= alpha].sort_values(["p_adjusted", "p_value"], kind="mergesort")
    result = kept.reset_index(drop=True)
    result.attrs.update(scored.attrs)
    result.attrs["n_tests"] = family
    return result


def significant_miner(miner: Miner, alpha: float = 0.05, correction: str = "fdr_bh", method: str = "fisher",
                      alternative: str = "greater") -> Miner:
    """
    Wrap a ``table -> rules`` miner so that it returns only the significant rules (see `significant_rules`). It serves
    as ``miner=`` of ``ARM.window_rules`` and of `permutation_null`. The number of transactions is
    ``rules.attrs["n_transactions"]`` if the miner records it (the temporal rules) and, if not, the number of rows of
    the table.
    """
    def significant(table: pd.DataFrame) -> pd.DataFrame:
        rules = miner(table)
        if not len(rules):
            return rules
        n = rules.attrs.get("n_transactions", len(table))
        return significant_rules(rules, alpha, correction, n_transactions=n, method=method, alternative=alternative)
    return significant


# ---------------------------------------------------------- intervalos

def confidence_intervals(rules: pd.DataFrame, n_transactions: Optional[int] = None, level: float = 0.95,
                         method: str = "wilson", n_resamples: int = 1000, random_state: Optional[int] = 0) -> pd.DataFrame:
    """
    Confidence intervals for the support, confidence and lift of each rule.

    Args:
        rules (pandas.DataFrame): crisp rules (see `rule_significance`)
        n_transactions (int, optional): number of transactions (see `rule_significance`)
        level (float): level of the interval, in (0, 1)
        method (str): 'wilson' -- Wilson interval of a proportion: the support over n transactions and the confidence
            over the transactions of the antecedent; it gives no interval for the lift (NaN) --; or 'bootstrap' --
            percentiles of `n_resamples` resamplings of the transactions with replacement, which also give the lift.
            For one rule it is enough to resample its four 2x2 cells (a multinomial), so the table of transactions is
            not needed
        n_resamples (int), random_state (int): only for 'bootstrap'

    Returns:
        Copy of `rules` with the columns 'support_low', 'support_high', 'confidence_low', 'confidence_high',
        'lift_low' and 'lift_high'. Both methods assume independent transactions: with autocorrelated series the
        intervals are too narrow.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1), got: {}".format(level))
    if method not in ("wilson", "bootstrap"):
        raise ValueError("method must be 'wilson' or 'bootstrap', got: '{}'".format(method))
    nAC, nA, nC, n = _rule_counts(rules, n_transactions)
    result = rules.copy()
    columns = ["support_low", "support_high", "confidence_low", "confidence_high", "lift_low", "lift_high"]
    values = np.full((len(rules), 6), np.nan)
    if method == "wilson":
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        for row in range(len(rules)):
            values[row, 0:2] = _wilson(int(nAC[row]), n, z)
            if nA[row] > 0:
                values[row, 2:4] = _wilson(int(nAC[row]), int(nA[row]), z)
    else:
        rng = np.random.default_rng(random_state)
        low, high = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        for row in range(len(rules)):
            cells = np.array([nAC[row], nA[row] - nAC[row], nC[row] - nAC[row], n - nA[row] - nC[row] + nAC[row]], dtype=float)
            draws = rng.multinomial(n, cells / n, size=n_resamples).astype(float)
            ac, a, c = draws[:, 0], draws[:, 0] + draws[:, 1], draws[:, 0] + draws[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                confidence = np.where(a > 0, ac / a, np.nan)
                lift = np.where((a > 0) & (c > 0), confidence / (c / n), np.nan)
            for offset, sample in ((0, ac / n), (2, confidence), (4, lift)):
                if not np.all(np.isnan(sample)):
                    values[row, offset:offset + 2] = np.nanquantile(sample, [low, high])
    for index, column in enumerate(columns):
        result[column] = values[:, index]
    return result


def _wilson(successes: int, trials: int, z: float) -> Tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------- permutaciones

def shuffled_items(items: pd.DataFrame, null: str = "shift", random_state: Union[None, int, np.random.Generator] = None) -> pd.DataFrame:
    """
    A version of `items` without the relations between columns, to use as null data.

    Args:
        items (pandas.DataFrame): table of items (boolean or of degrees), one row per instant
        null (str): 'permute' -- each column is permuted at random separately: valid with independent
            transactions --; or 'shift' (default) -- each column is circularly shifted by a random number of rows:
            it keeps the autocorrelation and the cycles of each series and only breaks its alignment with the
            others, which is what must be done with time series
        random_state: seed or ``numpy.random.Generator``
    """
    if null not in NULLS:
        raise ValueError("null must be one of {}, got: '{}'".format(NULLS, null))
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    values = items.to_numpy()
    rows = len(values)
    shuffled = np.empty_like(values)
    for column in range(values.shape[1]):
        if null == "permute":
            shuffled[:, column] = values[rng.permutation(rows), column]
        else:
            shuffled[:, column] = np.roll(values[:, column], int(rng.integers(1, rows)) if rows > 1 else 0)
    return pd.DataFrame(shuffled, index=items.index, columns=items.columns)


def permutation_null(items: pd.DataFrame, miner: Miner, measure: str = "p_value", n_permutations: int = 100,
                     null: str = "shift", random_state: Optional[int] = 0, n_transactions: Optional[int] = None,
                     test: str = "fisher") -> pd.DataFrame:
    """
    Null distribution of the best value of a measure: it re-mines `n_permutations` times over shuffled data
    (`shuffled_items`) and stores, for each one, the number of rules found and the BEST measure among them (the
    maximum, or the minimum if it is a p-value).

    Args:
        items (pandas.DataFrame): the table of items the rules came from
        miner (Callable): ``table -> DataFrame of rules`` with the same parameters used to mine the real rules (e.g.
            ``crisp_rule_miner(0.05)``, a function that calls ``temporal_association_rules`` or a fuzzy miner)
        measure (str): statistic to follow. 'p_value' (default): the SMALLEST p-value (`rule_significance`) among the
            rules of the shuffled data -- the minP method of Westfall and Young, which takes into account that rules
            of low support reach extreme measures by chance; only for crisp rules --. Any other column of the rules
            ('lift', 'confidence', 'zhangs_metric', ...): its MAXIMUM; it is the only option with fuzzy rules
        n_permutations (int): number of shuffled datasets (the minimum possible p-value is ``1/(n+1)``)
        null (str): 'shift' or 'permute' (see `shuffled_items`)
        random_state (int): seed
        n_transactions (int, optional): with 'p_value', number of transactions if the miner does not record it in
            ``rules.attrs`` (by default, the number of rows)
        test (str): with 'p_value', 'fisher' or 'chi2'

    Returns:
        pandas.DataFrame with one row per permutation: 'permutation', 'n_rules' and 'min_p_value' or
        'max_<measure>' (NaN if no rule came out).
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    pvalues = measure == "p_value"
    column = "min_p_value" if pvalues else "max_" + measure
    rng = np.random.default_rng(random_state)
    rows = []
    for permutation in range(n_permutations):
        table = shuffled_items(items, null, rng)
        rules = miner(table)
        best = float("nan")
        if len(rules):
            if pvalues:
                n = rules.attrs.get("n_transactions", n_transactions if n_transactions is not None else len(table))
                best = float(rule_significance(rules, n, test)["p_value"].min())
            elif measure in rules.columns:
                best = float(rules[measure].max())
            else:
                raise ValueError("measure '{}' is not a column of the rules: {}".format(measure, sorted(rules.columns)))
        rows.append({"permutation": permutation, "n_rules": len(rules), column: best})
    result = pd.DataFrame(rows, columns=["permutation", "n_rules", column])
    result.attrs.update({"measure": measure, "test": test})
    return result


def permutation_p_values(rules: pd.DataFrame, null_distribution: pd.DataFrame,
                         n_transactions: Optional[int] = None) -> pd.DataFrame:
    """
    Permutation p-value of each rule against the null distribution of the best measure (`permutation_null`): the
    fraction of shuffled datasets whose best rule is at least as good as its own,
    ``(1 + #{null best better or equal}) / (1 + n_permutations)``. Since it compares with the best of ALL the rules of
    the shuffled data, it is already adjusted for multiple testing (FWER) and for having selected the rules from the
    data itself.

    Args:
        rules (pandas.DataFrame): the real rules, mined with the same miner
        null_distribution (pandas.DataFrame): result of `permutation_null`
        n_transactions (int, optional): with the 'p_value' measure, number of transactions if the rules do not have
            the 'p_value' column yet nor record it in ``rules.attrs``

    Returns:
        Copy of `rules` with the column 'p_permutation' (and 'p_value' if that measure was used).
    """
    measure = null_distribution.attrs.get("measure")
    if measure is None:
        raise ValueError("`null_distribution` does not come from permutation_null (attrs['measure'] is missing)")
    result = rules.copy()
    if measure == "p_value":
        if "p_value" not in result.columns:
            result = rule_significance(result, n_transactions, null_distribution.attrs.get("test", "fisher"))
        null = null_distribution["min_p_value"].to_numpy(dtype=float)
        exceed = (null[None, :] <= result["p_value"].to_numpy(dtype=float)[:, None] * (1 + 1e-9) + 1e-300).sum(axis=1)
    else:
        if measure not in result.columns:
            raise ValueError("measure '{}' is not a column of the rules".format(measure))
        null = null_distribution["max_" + measure].to_numpy(dtype=float)
        exceed = (null[None, :] >= result[measure].to_numpy(dtype=float)[:, None] - 1e-12).sum(axis=1)
    result["p_permutation"] = (1.0 + exceed) / (1.0 + len(null))
    return result


def count_candidates(itemsets: Iterable[Iterable[str]], is_valid: Callable[[frozenset, frozenset], bool],
                     eligible_consequent: Callable[[str], bool]) -> int:
    """
    Count the candidate rules ``A => C`` of some itemsets that meet a structural condition: the consequent uses only
    items with ``eligible_consequent`` and the (antecedent, consequent) pair passes ``is_valid``. It is used by
    ``temporal_association_rules`` to give the size of its family of hypotheses.
    """
    total = 0
    for itemset in itemsets:
        items = frozenset(itemset)
        usable = [item for item in sorted(items) if eligible_consequent(item)]
        for size in range(1, min(len(usable), len(items) - 1) + 1):
            for chosen in combinations(usable, size):
                consequent = frozenset(chosen)
                if is_valid(items - consequent, consequent):
                    total += 1
    return total


__all__ = [
    "rule_significance", "adjust_pvalues", "candidate_rule_count", "significant_rules", "significant_miner",
    "confidence_intervals", "shuffled_items", "permutation_null", "permutation_p_values",
]
