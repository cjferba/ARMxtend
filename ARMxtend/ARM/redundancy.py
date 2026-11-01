# -*- coding: utf-8 -*-
"""
Poda de reglas de asociacion redundantes.

Una regla  X2 -> Y2  es REDUNDANTE si existe otra regla distinta  X1 -> Y1
mas general -- con antecedente menor o igual (X1 incluido en X2) y
consecuente mayor o igual (Y1 incluye a Y2) -- que tiene EXACTAMENTE el
mismo soporte y la misma confianza (Bastide, Taouil, Pasquier, Stumme &
Lakhal, 2000, "Mining minimal non-redundant association rules using frequent
closed itemsets"; ver tambien Zaki, 2004). La regla general transmite toda la informacion de la
redundante: exige menos condiciones y concluye al menos lo mismo.

Ademas, ``prune_unproductive_rules`` elimina las reglas "improductivas": las que
anaden condiciones al antecedente de otra regla sin mejorar apreciablemente su
confianza (Bayardo, Agrawal y Gunopulos, 2000, "Constraint-based rule mining in
large, dense databases", Data Mining and Knowledge Discovery 4, 217-240). A
diferencia de la redundancia, que exige igualdad exacta, la tolerancia
(``min_improvement``) es un parametro.

Funciona con cualquier DataFrame de reglas con columnas 'antecedents' y
'consequents' (frozensets) mas 'support' y 'confidence': la salida de
``ARM.association_rules``, ``FIM.FARE.fuzzy_association_rules``, las reglas
con items ausentes de ``ARM.negative_rules`` o las graduales de
``ARM.gradual_rules``.

@author Carlos Fernandez-Basso
"""
import math
from itertools import combinations
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd


def prune_redundant_rules(rules: pd.DataFrame, tol: float = 1e-9, support_column: str = "support",
                         confidence_column: str = "confidence") -> pd.DataFrame:
    """
    Elimina las reglas redundantes de `rules`.

    Argumentos:
        rules (pandas.DataFrame): reglas con columnas 'antecedents',
            'consequents', `support_column` y `confidence_column`
        tol (float): tolerancia absoluta para considerar iguales dos valores
            de soporte/confianza
        support_column, confidence_column (str): nombres de las columnas

    Retorna:
        pandas.DataFrame con las reglas no redundantes (misma estructura y
        orden relativo que `rules`; las reglas duplicadas se reducen a una).
    """
    if len(rules) == 0:
        return rules.copy()

    support = rules[support_column].to_numpy(dtype=float)
    confidence = rules[confidence_column].to_numpy(dtype=float)
    antecedents = [frozenset(a) for a in rules["antecedents"]]
    consequents = [frozenset(c) for c in rules["consequents"]]

    # Solo pueden ser equivalentes las reglas con (soporte, confianza) iguales:
    # se agrupan por esos valores redondeados a `tol` y se comparan dentro del grupo.
    digits = max(0, int(round(-math.log10(tol)))) if tol > 0 else 12
    groups: Dict[Tuple[float, float], List[int]] = {}
    for position in range(len(rules)):
        groups.setdefault((round(support[position], digits), round(confidence[position], digits)), []).append(position)

    redundant = set()
    for positions in groups.values():
        for candidate in positions:
            for general in positions:
                if general == candidate or general in redundant:
                    continue
                if abs(support[general] - support[candidate]) > tol or abs(confidence[general] - confidence[candidate]) > tol:
                    continue
                isMoreGeneral = antecedents[general] <= antecedents[candidate] and consequents[general] >= consequents[candidate]
                isIdentical = antecedents[general] == antecedents[candidate] and consequents[general] == consequents[candidate]
                # Una regla identica solo elimina a las de indice mayor (se conserva la primera)
                if isMoreGeneral and (not isIdentical or general < candidate):
                    redundant.add(candidate)
                    break

    return rules.iloc[[p for p in range(len(rules)) if p not in redundant]].reset_index(drop=True)


_EXHAUSTIVE_LIMIT = 12      # antecedentes de hasta este tamano: se prueban todos sus subconjuntos


def prune_unproductive_rules(rules: pd.DataFrame, min_improvement: float = 0.0,
                             confidence_column: str = "confidence") -> pd.DataFrame:
    """
    Elimina las reglas improductivas: una regla ``X => Y`` se descarta si existe
    otra ``X' => Y`` en `rules`, con X' subconjunto propio de X, tal que
    ``conf(X => Y) - conf(X' => Y) <= min_improvement``; es decir, si las
    condiciones de X que no estan en X' no mejoran la confianza en mas de
    `min_improvement`. Se compara con CUALQUIER regla mas general presente
    (no solo con las que se conservan), asi que el resultado no depende del orden.

    Argumentos:
        rules (pandas.DataFrame): reglas con 'antecedents', 'consequents' y `confidence_column`
        min_improvement (float): mejora minima de confianza que ha de aportar
            cada condicion adicional; 0 (por defecto) descarta las que no mejoran
            la confianza de una regla mas general; con 0.05 hay que ganar al menos
            5 puntos
        confidence_column (str): nombre de la columna de confianza

    Retorna:
        pandas.DataFrame con las reglas productivas, en el orden de `rules`
        (la regla mas general de cada familia siempre se conserva)
    """
    if len(rules) == 0:
        return rules.copy()
    antecedents = [frozenset(a) for a in rules["antecedents"]]
    consequents = [frozenset(c) for c in rules["consequents"]]
    confidence = rules[confidence_column].to_numpy(dtype=float)

    best: Dict[Tuple[FrozenSet[str], FrozenSet[str]], float] = {}
    byConsequent: Dict[FrozenSet[str], List[int]] = {}
    for position, key in enumerate(zip(antecedents, consequents)):
        best[key] = max(best.get(key, -math.inf), confidence[position])
        byConsequent.setdefault(consequents[position], []).append(position)

    keep = []
    for position in range(len(rules)):
        antecedent, consequent = antecedents[position], consequents[position]
        if len(antecedent) <= _EXHAUSTIVE_LIMIT:
            general = (best.get((frozenset(subset), consequent), -math.inf)
                       for size in range(1, len(antecedent))
                       for subset in combinations(antecedent, size))
        else:
            general = (confidence[other] for other in byConsequent[consequent] if antecedents[other] < antecedent)
        strongest = max(general, default=-math.inf)
        keep.append(not (confidence[position] - strongest <= min_improvement + 1e-12))
    return rules.iloc[[position for position, kept in enumerate(keep) if kept]].reset_index(drop=True)
