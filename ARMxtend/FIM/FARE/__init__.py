# -*- coding: utf-8 -*-
"""
FARE (Fuzzy Association Rule Extraction): generacion de reglas de asociacion
difusas a partir de itemsets frecuentes difusos, usando la decomposicion por
alpha-cortes (Representation by Levels, RL-theory) de:

    Delgado, M., Ruiz, M.D., Sanchez, D., Serrano, J.M. (2011). "A formal
    model for mining fuzzy rules using the RL representation theory".
    Information Sciences, 181, 5194-5213.

    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223 (Seccion 4.3, Ecs. (4)-(7)).

    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2021). "Spark
    solutions for discovering fuzzy association rules in Big Data". Int. J.
    of Approximate Reasoning, 137, 94-112 (Seccion 2.1, Ecs. (2)-(4)).

A diferencia de la primera version de este modulo (que aproximaba una regla
difusa evaluandola en un unico alpha-corte, delegando en
`ARM.association_rules`), esta version calcula el soporte (FSupp), la
confianza (FConf) y el factor de certeza (FCF) difusos EXACTOS segun la
formula de las Ecs. (2)-(7): una integral (suma ponderada por
(alpha_i - alpha_i+1)) de la medida crisp correspondiente en cada
alpha-corte, en lugar de evaluar la medida crisp en un unico nivel.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, Mapping, Optional
import itertools

import numpy as np
import pandas as pd

from .._shared import alpha_levels, alpha_weights, weighted_alpha_aggregate
from ...ARM._measures import MEASURES

# 'support' (FSupp) se calcula del bit-list del itemset A u B; el resto son
# funciones de (supp(AB), supp(A), supp(B)) evaluadas en cada alpha-corte y
# agregadas con los pesos (alpha_i - alpha_i+1), segun la RL-theory. La
# confianza difusa usa ademas la convencion 0/0 := 1 (ver
# `_confidence_per_level`).
_DEFAULT_COLUMNS = ("support", "confidence", "certainty_factor")
_METRICS = ("support",) + tuple(MEASURES)


def _confidence_per_level(support_AB_vector, support_A_vector):
    """
    Ratio a_i / (a_i + b_i) de la Ec. (4), nivel a nivel. Cuando el
    antecedente no aparece en ningun alpha-corte a ese nivel (a_i + b_i = 0),
    se sigue la convencion del articulo de tomar 1 como valor de la
    indeterminacion "0/0" (ver Seccion 4.4 de Ruiz et al., 2016).
    """
    return np.where(support_A_vector > 0,
                    np.divide(support_AB_vector, support_A_vector,
                             out=np.ones_like(support_AB_vector, dtype=float),
                             where=support_A_vector > 0),
                    1.0)


def fuzzy_association_rules(freq_itemsets: Mapping[str, np.ndarray], num_alpha: int, metric: str = "confidence",
                            min_threshold: float = 0.8, measures: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Genera las reglas de asociacion difusas A -> B con `metric` >=
    `min_threshold`, a partir de un conjunto de itemsets frecuentes difusos.

    Argumentos:
        freq_itemsets (dict[str, numpy.ndarray]): itemsets frecuentes
            difusos, con clave 'A-B-C' y valor su bit-list de soporte
            relativo en cada uno de los `num_alpha` alpha-cortes (ver
            `FIM._shared.alpha_cuts`), tal y como lo devuelven
            `FIM.Eclat.FuzzyDECLAT`, `FIM.BD_FARE.FuzzyDAprioriTID` o
            `FFIM.fuzzy_fpgrowth`. Debe incluir el bit-list de todo
            subconjunto no vacio de cada itemset (garantizado por la
            propiedad de cierre descendente de dichos algoritmos)
        num_alpha (int): numero de alpha-cortes usados para construir
            `freq_itemsets` (debe coincidir con el empleado al minarlos)
        metric (str): 'support' (FSupp), 'confidence' (FConf),
            'certainty_factor' (FCF) o cualquiera de las medidas de
            `ARM._measures.MEASURES` ('lift', 'leverage', 'jaccard',
            'cosine', 'kulczynski', 'all_confidence', 'max_confidence',
            'added_value', 'zhangs_metric', 'yules_q'), que se calculan en
            cada alpha-corte y se agregan con los pesos de la RL-theory
        min_threshold (float): umbral minimo para `metric`
        measures (Iterable[str], opcional): medidas adicionales a incluir
            como columnas del resultado, ademas de 'support', 'confidence'
            y 'certainty_factor' (que se devuelven siempre)

    Retorna:
        pandas.DataFrame con columnas 'antecedents', 'consequents',
        'support' (FSupp), 'confidence' (FConf) y 'certainty_factor' (FCF),
        mas una columna por cada medida de `measures`, de cada regla que
        alcanza `min_threshold` en `metric`.
    """
    if metric not in _METRICS:
        raise ValueError("metric debe ser uno de {}, recibido: '{}'".format(_METRICS, metric))
    extraMeasures = [m for m in (measures or []) if m not in _DEFAULT_COLUMNS]
    for measure in extraMeasures:
        if measure not in _METRICS:
            raise ValueError("measure debe ser una de {}, recibida: '{}'".format(_METRICS, measure))
    columns = list(_DEFAULT_COLUMNS) + extraMeasures
    # `metric` ha de calcularse siempre para poder filtrar por el, aunque no se devuelva
    computed = columns + ([metric] if metric not in columns else [])

    weights = alpha_weights(alpha_levels(num_alpha))
    rows = []

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

    return pd.DataFrame(rows, columns=["antecedents", "consequents"] + columns)
