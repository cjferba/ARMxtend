# -*- coding: utf-8 -*-
"""
FARE (Fuzzy Association Rule Extraction): generation of fuzzy association rules from fuzzy frequent itemsets, using
the alpha-cut decomposition (Representation by Levels, RL-theory) of:

    Delgado, M., Ruiz, M.D., Sanchez, D., Serrano, J.M. (2011). "A formal
    model for mining fuzzy rules using the RL representation theory".
    Information Sciences, 181, 5194-5213.

    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223 (Section 4.3, Eqs. (4)-(7)).

    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2021). "Spark
    solutions for discovering fuzzy association rules in Big Data". Int. J.
    of Approximate Reasoning, 137, 94-112 (Section 2.1, Eqs. (2)-(4)).

Unlike the first version of this module (which approximated a fuzzy rule by evaluating it at a single alpha-cut,
delegating to `ARM.association_rules`), this version computes the EXACT fuzzy support (FSupp), confidence (FConf) and
certainty factor (FCF) according to the formula of Eqs. (2)-(7): an integral (sum weighted by (alpha_i - alpha_i+1))
of the corresponding crisp measure at each alpha-cut, instead of evaluating the crisp measure at a single level.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, Mapping, Optional
import itertools

import numpy as np
import pandas as pd

from .._shared import alpha_levels, alpha_weights, weighted_alpha_aggregate
from ...ARM._measures import MEASURES

# 'support' (FSupp) is computed from the bit-list of the itemset A u B; the rest are
# functions of (supp(AB), supp(A), supp(B)) evaluated at each alpha-cut and
# aggregated with the weights (alpha_i - alpha_i+1), according to the RL-theory. The
# confianza difusa usa ademas la convencion 0/0 := 1 (ver
# `_confidence_per_level`).
_DEFAULT_COLUMNS = ("support", "confidence", "certainty_factor")
_METRICS = ("support",) + tuple(MEASURES)


def _confidence_per_level(support_AB_vector, support_A_vector):
    """
    Ratio a_i / (a_i + b_i) of Eq. (4), level by level. When the antecedent does not appear in any alpha-cut at that
    level (a_i + b_i = 0), the convention of the paper is followed, taking 1 as the value of the "0/0"
    indeterminacy (see Section 4.4 of Ruiz et al., 2016).
    """
    return np.where(support_A_vector > 0,
                    np.divide(support_AB_vector, support_A_vector,
                             out=np.ones_like(support_AB_vector, dtype=float),
                             where=support_A_vector > 0),
                    1.0)


def fuzzy_association_rules(freq_itemsets: Mapping[str, np.ndarray], num_alpha: int, metric: str = "confidence",
                            min_threshold: float = 0.8, measures: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Generate the fuzzy association rules A -> B with `metric` >= `min_threshold`, from a set of fuzzy frequent
    itemsets.

    Args:
        freq_itemsets (dict[str, numpy.ndarray]): fuzzy frequent itemsets, with key 'A-B-C' and as value their
            bit-list of relative support in each of the `num_alpha` alpha-cuts (see `FIM._shared.alpha_cuts`), as
            returned by `FIM.Eclat.FuzzyDECLAT`, `FIM.BD_FARE.FuzzyDAprioriTID` or `FFIM.fuzzy_fpgrowth`. It must
            include the bit-list of every non-empty subset of each itemset (guaranteed by the downward closure
            property of those algorithms)
        num_alpha (int): number of alpha-cuts used to build `freq_itemsets` (it must match the one used to mine them)
        metric (str): 'support' (FSupp), 'confidence' (FConf), 'certainty_factor' (FCF) or any of the measures of
            `ARM._measures.MEASURES` ('lift', 'leverage', 'jaccard', 'cosine', 'kulczynski', 'all_confidence',
            'max_confidence', 'added_value', 'zhangs_metric', 'yules_q'), which are computed at each alpha-cut and
            aggregated with the weights of the RL-theory
        min_threshold (float): minimum threshold for `metric`
        measures (Iterable[str], optional): additional measures to include as columns of the result, besides
            'support', 'confidence' and 'certainty_factor' (which are always returned)

    Returns:
        pandas.DataFrame with columns 'antecedents', 'consequents', 'support' (FSupp), 'confidence' (FConf) and
        'certainty_factor' (FCF), plus a column per measure of `measures`, for each rule that reaches
        `min_threshold` in `metric`.
    """
    if metric not in _METRICS:
        raise ValueError("metric debe ser uno de {}, recibido: '{}'".format(_METRICS, metric))
    extraMeasures = [m for m in (measures or []) if m not in _DEFAULT_COLUMNS]
    for measure in extraMeasures:
        if measure not in _METRICS:
            raise ValueError("measure must be one of {}, received: '{}'".format(_METRICS, measure))
    columns = list(_DEFAULT_COLUMNS) + extraMeasures
    # `metric` must always be computed to be able to filter by it, even if it is not returned
    computed = columns + ([metric] if metric not in columns else [])

    weights = alpha_weights(alpha_levels(num_alpha))
    rows = []
    nCandidates = 0

    for itemsetKey, supportVectorAB in freq_itemsets.items():
        items = itemsetKey.split("-")
        if len(items) < 2:
            continue

        for antecedentSize in range(1, len(items)):
            for antecedent in itertools.combinations(items, antecedentSize):
                consequent = sorted(item for item in items if item not in antecedent)
                antecedentKey = "-".join(sorted(antecedent))
                consequentKey = "-".join(consequent)

                supportVectorA = freq_itemsets.get(antecedentKey)
                supportVectorB = freq_itemsets.get(consequentKey)
                if supportVectorA is None or supportVectorB is None:
                    continue

                nCandidates += 1
                # FSupp(A -> B) = FSupp(A U B), Ec. (3): t(AUB)>=alpha <=> t(A)>=alpha and t(B)>=alpha
                scores = {"support": weighted_alpha_aggregate(supportVectorAB, weights)}
                for measure in computed:
                    if measure == "support":
                        continue
                    if measure == "confidence":
                        perLevel = _confidence_per_level(supportVectorAB, supportVectorA)
                    else:
                        perLevel = MEASURES[measure](supportVectorAB, supportVectorA, supportVectorB)
                    scores[measure] = weighted_alpha_aggregate(perLevel, weights)

                if scores[metric] >= min_threshold:
                    row: Dict[str, Any] = {"antecedents": frozenset(antecedent), "consequents": frozenset(consequent)}
                    row.update({column: scores[column] for column in columns})
                    rows.append(row)

    result = pd.DataFrame(rows, columns=["antecedents", "consequents"] + columns)
    result.attrs["n_candidates"] = nCandidates      # partitions examined: the family of hypotheses of ARM.fuzzy_significant_rules
    return result
