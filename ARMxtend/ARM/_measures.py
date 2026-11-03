# -*- coding: utf-8 -*-
"""
Interest measures for association rules shared by the crisp case (``ARM.association_rules``) and the fuzzy one
(``FIM.FARE``, ``ARM.meta_rules``).

@author Carlos Fernandez-Basso
"""
from typing import Union
import numpy as np
from numpy.typing import ArrayLike


def _as_arrays(support_AB, support_A, support_B):
    return (np.asarray(support_AB, dtype=float),
            np.asarray(support_A, dtype=float),
            np.asarray(support_B, dtype=float))


def _finish(result):
    """Return a float if the result is scalar and an ndarray otherwise."""
    return float(result) if np.ndim(result) == 0 else result


def _safe_div(numerator, denominator, default=0.0):
    """numerator / denominator element by element, with `default` where denominator == 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    positive = denominator != 0
    return np.where(positive, numerator / np.where(positive, denominator, 1.0), default)


def certainty_factor(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """
    Certainty factor of a rule A -> B (Shortliffe & Buchanan), as defined in:
        Delgado, M., Ruiz, M.D., Sanchez, D. (2011). "A formal model for
        mining fuzzy rules using the RL representation theory". Information
        Sciences, 181, 5194-5213 (Definition 1).

    Unlike confidence, the certainty factor takes into account the "background" support of the consequent: a positive
    value indicates that the presence of the antecedent INCREASES the belief in the consequent, a negative value that
    it DECREASES it, and 0 that it does not change it (independence). Its domain is [-1, 1], versus the unbounded
    [0, +inf) of conviction, which makes it easier to set a threshold and compare rules with each other.

    CF(A -> B) = (Conf(A->B) - supp(B)) / (1 - supp(B))  if Conf(A->B) > supp(B)
               = (Conf(A->B) - supp(B)) / supp(B)         if Conf(A->B) < supp(B)
               = 0                                         if Conf(A->B) = supp(B)

    When the antecedent has no support (supp(A) = 0) the certainty factor is 0, as in Eq. (7) of Ruiz et al. (2016) on
    the 4ft table.

    It is valid with both scalars and numpy arrays (applied element by element), which allows reusing it to aggregate
    the fuzzy certainty factor level by level (see `FIM._shared.weighted_alpha_aggregate`).

    Args:
        support_AB (float | numpy.ndarray): support of A union B (support of the rule), Supp(A -> B)
        support_A (float | numpy.ndarray): support of the antecedent, Supp(A)
        support_B (float | numpy.ndarray): support of the consequent, Supp(B)

    Returns:
        float | numpy.ndarray in [-1, 1]
    """
    supportAB = np.asarray(support_AB, dtype=float)
    supportA = np.asarray(support_A, dtype=float)
    supportB = np.asarray(support_B, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.where(supportA > 0, supportAB / np.where(supportA > 0, supportA, 1), 0.0)

    increaseBelief = (confidence - supportB) / np.where(supportB < 1, 1 - supportB, 1)
    decreaseBelief = (confidence - supportB) / np.where(supportB > 0, supportB, 1)

    result = np.where(confidence > supportB, increaseBelief,
                      np.where(confidence < supportB, decreaseBelief, 0.0))
    # Without an antecedent (supp(A) = 0, e.g. at a high alpha-cut where A does not
    # appear) the 4ft table satisfies ad = bc and Eq. (7) of Ruiz et al. (2016)
    # gives CF = 0; without this line it would give -1 (confidence 0 < supp(B)).
    result = np.where(supportA > 0, result, 0.0)

    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Additional interest measures. All are functions of the three supports
# (supp(A u B), supp(A), supp(B)) -- equivalently, of the
# contingencia 4ft <a, b, c, d> = <sAB, sA - sAB, sB - sAB, 1 - sA - sB + sAB>
# -- so, according to the RL-theory (Delgado et al., 2011), each one is
# generalized to the fuzzy case by evaluating it at each alpha-cut and aggregating with
# `FIM._shared.weighted_alpha_aggregate`. Where the measure is a 0/0
# indeterminacy (e.g. antecedent without support at that level) 0.0 is returned, the
# neutral value ("no evidence"); the only exception is fuzzy confidence, which
# follows the convention of the paper (0/0 := 1), implemented in `FIM.FARE`.
# ---------------------------------------------------------------------------

def confidence(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """Conf(A -> B) = supp(AB) / supp(A). Rango [0, 1]."""
    sAB, sA, _ = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, sA))


def lift(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """Lift(A -> B) = Conf(A -> B) / supp(B). Rango [0, inf); 1 = independencia."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(_safe_div(sAB, sA), sB))


def leverage(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """Leverage(A -> B) = supp(AB) - supp(A) supp(B). Rango [-1/4, 1/4]; 0 = independencia."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(sAB - sA * sB)


def jaccard(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """Jaccard(A, B) = supp(AB) / (supp(A) + supp(B) - supp(AB)). Rango [0, 1]; simetrica."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, sA + sB - sAB))


def cosine(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """Cosine(A, B) = supp(AB) / sqrt(supp(A) supp(B)). Rango [0, 1]; simetrica."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, np.sqrt(sA * sB)))


def kulczynski(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """
    Kulczynski(A, B) = (Conf(A -> B) + Conf(B -> A)) / 2. Range [0, 1]; symmetric. Unlike lift or cosine, it is
    invariant to "null" transactions (that contain neither A nor B) (Wu, Chen & Han, 2010).
    """
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(0.5 * (_safe_div(sAB, sA) + _safe_div(sAB, sB)))


def all_confidence(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """AllConf(A, B) = supp(AB) / max(supp(A), supp(B)) = min(Conf(A->B), Conf(B->A)). Rango [0, 1]."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, np.maximum(sA, sB)))


def max_confidence(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """MaxConf(A, B) = max(Conf(A -> B), Conf(B -> A)). Rango [0, 1]."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(np.maximum(_safe_div(sAB, sA), _safe_div(sAB, sB)))


def added_value(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """AddedValue(A -> B) = Conf(A -> B) - supp(B) (Tan, Kumar & Srivastava, 2004). Rango [-1, 1]."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(np.where(sA > 0, _safe_div(sAB, sA) - sB, 0.0))


def zhangs_metric(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """
    Zhang's metric (Zhang, 2000) = (supp(AB) - supp(A) supp(B)) /
    max(supp(AB)(1 - supp(A)), supp(B)(supp(A) - supp(AB))). Rango [-1, 1];
    +1 = asociacion positiva perfecta, -1 = negativa perfecta, 0 = independencia.
    """
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    denominator = np.maximum(sAB * (1 - sA), sB * (sA - sAB))
    return _finish(_safe_div(sAB - sA * sB, denominator))


def yules_q(support_AB: ArrayLike, support_A: ArrayLike, support_B: ArrayLike) -> Union[float, np.ndarray]:
    """
    Yule's Q = (ad - bc) / (ad + bc) over the 4ft table <a, b, c, d> of the rule (Tan et al., 2004). Range [-1, 1];
    0 = independence.
    """
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    a, b, c, d = sAB, sA - sAB, sB - sAB, 1.0 - sA - sB + sAB
    return _finish(_safe_div(a * d - b * c, a * d + b * c))


# Pure measures of (sAB, sA, sB), by name. `ARM.association_rules` and
# `FIM.FARE` build their columns from this registry.
MEASURES = {
    "confidence": confidence,
    "lift": lift,
    "leverage": leverage,
    "certainty_factor": certainty_factor,
    "jaccard": jaccard,
    "cosine": cosine,
    "kulczynski": kulczynski,
    "all_confidence": all_confidence,
    "max_confidence": max_confidence,
    "added_value": added_value,
    "zhangs_metric": zhangs_metric,
    "yules_q": yules_q,
}

# Measures added in this version (those before it were confidence,
# lift, leverage, conviction y certainty_factor).
EXTRA_MEASURES = ("jaccard", "cosine", "kulczynski", "all_confidence", "max_confidence",
                  "added_value", "zhangs_metric", "yules_q")
