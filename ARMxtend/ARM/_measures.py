# -*- coding: utf-8 -*-
"""
Medidas de interes para reglas de asociacion compartidas por el caso crisp
(``ARM.association_rules``) y el difuso (``FIM.FARE``, ``ARM.meta_rules``).

@author Carlos Fernandez-Basso
"""
import numpy as np


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
