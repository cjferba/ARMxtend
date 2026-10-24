# -*- coding: utf-8 -*-
"""
Reglas de asociacion con items AUSENTES (reglas negativas), crisp y difusas:
reglas como  A -> ~B  ("si A, entonces NO B"), ~A -> B  o  ~A -> ~B, donde
``~i`` significa que el item ``i`` no esta presente en la transaccion.

Sigue el enfoque de la seccion 4 de:

    Delgado, M., Ruiz, M.D., Sanchez, D., Serrano, J.M. (2011). "A formal
    model for mining fuzzy rules using the RL representation theory".
    Information Sciences, 181, 5194-5213.

En la RL-theory la ausencia de un item es su COMPLEMENTO en cada nivel de
restriccion (Ecs. (27)-(28)): el bit-list de ``~i`` en el nivel alpha_k es
el complemento del bit-list de ``i`` en ese nivel, y la tabla de
contingencia 4ft de (~A, B) es <c, d, a, b> a partir de la de (A, B)
= <a, b, c, d>. Por tanto no hace falta calcular frecuencias nuevas, y -- a
diferencia de la negacion de la teoria de conjuntos difusos (1 - mu) --
un itemset nunca contiene a la vez ``i`` y ``~i`` (su soporte es 0 en todo
nivel), de modo que reglas triviales como ``i -> ~i`` no pueden aparecer.

El mayor problema de minar con items ausentes es que, en datos dispersos, la
ausencia de un item es casi siempre frecuente y el numero de itemsets
explota (el soporte de la ausencia no cumple la propiedad de cierre
ascendente utilizada para podar). El articulo adopta la restriccion de Han &
Beheshti: considerar solo los items negativos ``~i`` cuyo item positivo
``i`` es frecuente (``only_frequent_positive=True``, por defecto). Con
``negate_items`` se puede ademas acotar explicitamente que items se niegan.

NOTA: los nombres de item no deben contener '-', que es el separador de
items en las claves de itemset de ``FIM._shared``.

@author Carlos Fernandez-Basso
"""
from typing import Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from ..FFIM import fpgrowth, fuzzy_fpgrowth_from_vectors
from ..FIM._shared import key_to_itemset, alpha_cuts, alpha_levels, alpha_weights, weighted_alpha_aggregate
from ..FIM.FARE import fuzzy_association_rules
from .association_rules import association_rules

NEGATION_PREFIX = "~"


def negate(item: str) -> str:
    """Nombre del item que representa la AUSENCIA de `item` ('A' -> '~A')."""
    return NEGATION_PREFIX + item


def is_negated(item: str) -> bool:
    """True si `item` es un item ausente ('~A')."""
    return item.startswith(NEGATION_PREFIX)


def _has_negated_item(row):
    return any(is_negated(item) for item in set(row["antecedents"]) | set(row["consequents"]))


def _negatable_items(vocabulary, positive_supports, negate_items, only_frequent_positive, min_supp):
    candidates = set(vocabulary) if negate_items is None else set(negate_items) & set(vocabulary)
    if only_frequent_positive:
        candidates = {item for item in candidates if positive_supports[item] >= min_supp}
    return sorted(candidates)


def add_absent_items(transactions: Sequence[Iterable[str]],
                     negate_items: Optional[Iterable[str]] = None) -> List[Set[str]]:
    """
    Amplia cada transaccion crisp con un item ``~i`` por cada item ``i`` de
    `negate_items` que NO este presente en ella.

    Argumentos:
        transactions (Sequence[Iterable[str]]): transacciones crisp
        negate_items (Iterable[str], opcional): items cuya ausencia se
            representa; por defecto, todos los que aparecen en alguna
            transaccion

    Retorna:
        list[set[str]]: las transacciones ampliadas
    """
    sets = [set(t) for t in transactions]
    negatable = set(negate_items) if negate_items is not None else set().union(*sets) if sets else set()
    return [t | {negate(item) for item in negatable - t} for t in sets]


def negative_association_rules(transactions: Sequence[Iterable[str]], min_supp: float, metric: str = "confidence",
                               min_threshold: float = 0.8, negate_items: Optional[Iterable[str]] = None,
                               only_frequent_positive: bool = True, only_negative: bool = True) -> pd.DataFrame:
    """
    Minera reglas de asociacion crisp con items presentes y/o ausentes.

    Argumentos:
        transactions (Sequence[Iterable[str]]): transacciones crisp
        min_supp (float): soporte minimo relativo de los itemsets (que
            pueden incluir items ausentes), en (0, 1]
        metric (str): medida de `ARM.association_rules` para filtrar
        min_threshold (float): umbral minimo de `metric`
        negate_items (Iterable[str], opcional): items cuya ausencia se
            considera (por defecto, todos)
        only_frequent_positive (bool): solo se niegan los items cuyo
            soporte positivo alcanza `min_supp` (restriccion de Han &
            Beheshti adoptada en Delgado et al., 2011, Seccion 6)
        only_negative (bool): si True (por defecto), devuelve solo las
            reglas que contienen al menos un item ausente; si False,
            tambien las puramente positivas

    Retorna:
        pandas.DataFrame de reglas (mismo formato que
        `ARM.association_rules`); los items ausentes aparecen como '~item'.
    """
    transactions = [set(t) for t in transactions]
    total = len(transactions)
    if total == 0:
        return association_rules(pd.DataFrame({"support": [], "itemsets": []}), min_threshold=min_threshold)

    vocabulary = set().union(*transactions)
    positiveSupports = {item: sum(item in t for t in transactions) / total for item in vocabulary}
    negatable = _negatable_items(vocabulary, positiveSupports, negate_items, only_frequent_positive, min_supp)

    extended = add_absent_items(transactions, negate_items=negatable)
    frequent = fpgrowth(extended, min_supp)
    itemsets = pd.DataFrame({
        "support": list(frequent.values()),
        "itemsets": [key_to_itemset(key) for key in frequent],
    })
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)

    if only_negative and len(rules):
        rules = rules[rules.apply(_has_negated_item, axis=1)].reset_index(drop=True)
    return rules


def fuzzy_negative_association_rules(transactions: Sequence[Iterable[Tuple[str, float]]], min_supp: float,
                                     num_alpha: int, metric: str = "confidence", min_threshold: float = 0.8,
                                     negate_items: Optional[Iterable[str]] = None,
                                     only_frequent_positive: bool = True, only_negative: bool = True,
                                     measures: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Minera reglas de asociacion DIFUSAS con items presentes y/o ausentes,
    segun la RL-theory (ver el modulo): el bit-list de ``~i`` en cada
    alpha-corte es el complemento del de ``i`` (no ``1 - grado``), y el
    soporte (FSupp), la confianza (FConf), el factor de certeza (FCF) y el
    resto de medidas se agregan nivel a nivel con los pesos
    (alpha_i - alpha_i+1) como en `FIM.FARE.fuzzy_association_rules`.

    Argumentos:
        transactions (Sequence[Iterable[Tuple[str, float]]]): transacciones
            difusas, cada una una lista de pares (item, grado en [0, 1]).
            Un item que no aparece en una transaccion tiene grado 0 en ella
        min_supp (float): FSupp minimo de los itemsets, en (0, 1]
        num_alpha (int): numero de alpha-cortes (el articulo recomienda 10)
        metric (str): medida de `FIM.FARE.fuzzy_association_rules`
        min_threshold (float): umbral minimo de `metric`
        negate_items (Iterable[str], opcional): items cuya ausencia se
            considera (por defecto, todos)
        only_frequent_positive (bool): solo se niegan los items cuyo FSupp
            positivo alcanza `min_supp` (Han & Beheshti)
        only_negative (bool): solo reglas con al menos un item ausente
        measures (Iterable[str], opcional): medidas adicionales a devolver
            (ver `FIM.FARE.fuzzy_association_rules`)

    Retorna:
        pandas.DataFrame de reglas difusas; los items ausentes aparecen
        como '~item'.
    """
    columns = ["antecedents", "consequents", "support", "confidence", "certainty_factor"]
    total = len(transactions)
    if total == 0:
        return pd.DataFrame(columns=columns + list(measures or []))

    weights = alpha_weights(alpha_levels(num_alpha))
    vectors = [{item: alpha_cuts(degree, num_alpha) for item, degree in tx} for tx in transactions]

    vocabulary = set().union(*(v.keys() for v in vectors))
    positiveSupports = {}
    for item in vocabulary:
        perLevel = sum((v[item] for v in vectors if item in v), np.zeros(num_alpha)) / total
        positiveSupports[item] = weighted_alpha_aggregate(perLevel, weights)
    negatable = _negatable_items(vocabulary, positiveSupports, negate_items, only_frequent_positive, min_supp)

    zeros = np.zeros(num_alpha, dtype=int)
    for v in vectors:
        for item in negatable:
            v[negate(item)] = 1 - v.get(item, zeros)

    frequent = fuzzy_fpgrowth_from_vectors(vectors, min_supp, num_alpha)
    rules = fuzzy_association_rules(frequent, num_alpha, metric=metric, min_threshold=min_threshold,
                                    measures=measures)

    if only_negative and len(rules):
        rules = rules[rules.apply(_has_negated_item, axis=1)].reset_index(drop=True)
    return rules
