# -*- coding: utf-8 -*-
"""
Medidas de interes para reglas de asociacion compartidas por el caso crisp
(``ARM.association_rules``) y el difuso (``FIM.FARE``, ``ARM.meta_rules``).

@author Carlos Fernandez-Basso
"""
import numpy as np


def _as_arrays(support_AB, support_A, support_B):
    return (np.asarray(support_AB, dtype=float),
            np.asarray(support_A, dtype=float),
            np.asarray(support_B, dtype=float))


def _finish(result):
    """Devuelve un float si el resultado es escalar y un ndarray en otro caso."""
    return float(result) if np.ndim(result) == 0 else result


def _safe_div(numerator, denominator, default=0.0):
    """numerator / denominator elemento a elemento, con `default` donde denominator == 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    positive = denominator != 0
    return np.where(positive, numerator / np.where(positive, denominator, 1.0), default)


def certainty_factor(support_AB, support_A, support_B):
    """
    Factor de certeza (certainty factor) de una regla A -> B (Shortliffe &
    Buchanan), tal y como se define en:
        Delgado, M., Ruiz, M.D., Sanchez, D. (2011). "A formal model for
        mining fuzzy rules using the RL representation theory". Information
        Sciences, 181, 5194-5213 (Definicion 1).

    A diferencia de la confianza, el factor de certeza tiene en cuenta el
    soporte "de fondo" del consecuente: un valor positivo indica que la
    presencia del antecedente AUMENTA la creencia en el consecuente, un
    valor negativo que la DISMINUYE, y 0 que no la cambia (independencia).
    Su dominio es [-1, 1], frente al [0, +inf) no acotado de la conviccion,
    lo que facilita fijar un umbral y comparar reglas entre si.

    CF(A -> B) = (Conf(A->B) - supp(B)) / (1 - supp(B))  si Conf(A->B) > supp(B)
               = (Conf(A->B) - supp(B)) / supp(B)         si Conf(A->B) < supp(B)
               = 0                                         si Conf(A->B) = supp(B)

    Es valido tanto con escalares como con arrays de numpy (aplicado
    elemento a elemento), lo que permite reutilizarlo para agregar el factor
    de certeza difuso nivel a nivel (ver `FIM._shared.weighted_alpha_aggregate`).

    Argumentos:
        support_AB (float | numpy.ndarray): soporte de A uniob B (soporte de
            la regla), Supp(A -> B)
        support_A (float | numpy.ndarray): soporte del antecedente, Supp(A)
        support_B (float | numpy.ndarray): soporte del consecuente, Supp(B)

    Retorna:
        float | numpy.ndarray en [-1, 1]
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

    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Medidas de interes adicionales. Todas son funciones de los tres soportes
# (supp(A u B), supp(A), supp(B)) -- equivalentemente, de la tabla de
# contingencia 4ft <a, b, c, d> = <sAB, sA - sAB, sB - sAB, 1 - sA - sB + sAB>
# -- por lo que, segun la RL-theory (Delgado et al., 2011), cada una se
# generaliza al caso difuso evaluandola en cada alpha-corte y agregando con
# `FIM._shared.weighted_alpha_aggregate`. Donde la medida es una indeterminacion
# 0/0 (p.ej. antecedente sin soporte en ese nivel) se devuelve 0.0, el valor
# neutro ("sin evidencia"); la unica excepcion es la confianza difusa, que
# sigue la convencion del articulo (0/0 := 1), implementada en `FIM.FARE`.
# ---------------------------------------------------------------------------

def confidence(support_AB, support_A, support_B):
    """Conf(A -> B) = supp(AB) / supp(A). Rango [0, 1]."""
    sAB, sA, _ = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, sA))


def lift(support_AB, support_A, support_B):
    """Lift(A -> B) = Conf(A -> B) / supp(B). Rango [0, inf); 1 = independencia."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(_safe_div(sAB, sA), sB))


def leverage(support_AB, support_A, support_B):
    """Leverage(A -> B) = supp(AB) - supp(A) supp(B). Rango [-1/4, 1/4]; 0 = independencia."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(sAB - sA * sB)


def jaccard(support_AB, support_A, support_B):
    """Jaccard(A, B) = supp(AB) / (supp(A) + supp(B) - supp(AB)). Rango [0, 1]; simetrica."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, sA + sB - sAB))


def cosine(support_AB, support_A, support_B):
    """Cosine(A, B) = supp(AB) / sqrt(supp(A) supp(B)). Rango [0, 1]; simetrica."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, np.sqrt(sA * sB)))


def kulczynski(support_AB, support_A, support_B):
    """
    Kulczynski(A, B) = (Conf(A -> B) + Conf(B -> A)) / 2. Rango [0, 1]; simetrica.
    A diferencia del lift o el cosine, es invariante frente a transacciones
    "nulas" (que no contienen A ni B) (Wu, Chen & Han, 2010).
    """
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(0.5 * (_safe_div(sAB, sA) + _safe_div(sAB, sB)))


def all_confidence(support_AB, support_A, support_B):
    """AllConf(A, B) = supp(AB) / max(supp(A), supp(B)) = min(Conf(A->B), Conf(B->A)). Rango [0, 1]."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(_safe_div(sAB, np.maximum(sA, sB)))


def max_confidence(support_AB, support_A, support_B):
    """MaxConf(A, B) = max(Conf(A -> B), Conf(B -> A)). Rango [0, 1]."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(np.maximum(_safe_div(sAB, sA), _safe_div(sAB, sB)))


def added_value(support_AB, support_A, support_B):
    """AddedValue(A -> B) = Conf(A -> B) - supp(B) (Tan, Kumar & Srivastava, 2004). Rango [-1, 1]."""
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    return _finish(np.where(sA > 0, _safe_div(sAB, sA) - sB, 0.0))


def zhangs_metric(support_AB, support_A, support_B):
    """
    Zhang's metric (Zhang, 2000) = (supp(AB) - supp(A) supp(B)) /
    max(supp(AB)(1 - supp(A)), supp(B)(supp(A) - supp(AB))). Rango [-1, 1];
    +1 = asociacion positiva perfecta, -1 = negativa perfecta, 0 = independencia.
    """
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    denominator = np.maximum(sAB * (1 - sA), sB * (sA - sAB))
    return _finish(_safe_div(sAB - sA * sB, denominator))


def yules_q(support_AB, support_A, support_B):
    """
    Yule's Q = (ad - bc) / (ad + bc) sobre la tabla 4ft <a, b, c, d> de la
    regla (Tan et al., 2004). Rango [-1, 1]; 0 = independencia.
    """
    sAB, sA, sB = _as_arrays(support_AB, support_A, support_B)
    a, b, c, d = sAB, sA - sAB, sB - sAB, 1.0 - sA - sB + sAB
    return _finish(_safe_div(a * d - b * c, a * d + b * c))


# Medidas puras de (sAB, sA, sB), por nombre. `ARM.association_rules` y
# `FIM.FARE` construyen sus columnas a partir de este registro.
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

# Medidas anadidas en esta version (las anteriores a ella eran confidence,
# lift, leverage, conviction y certainty_factor).
EXTRA_MEASURES = ("jaccard", "cosine", "kulczynski", "all_confidence", "max_confidence",
                  "added_value", "zhangs_metric", "yules_q")
