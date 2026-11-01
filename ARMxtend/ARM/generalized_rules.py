# -*- coding: utf-8 -*-
"""
Reglas de asociacion GENERALIZADAS (multinivel): reglas cuyos items pueden ser productos
o categorias de una taxonomia (``FIM.taxonomy``), como "ropa de abrigo => calzado".

``generalized_rules`` mina los itemsets generalizados (sin items junto a sus antecesores,
con soporte minimo opcionalmente distinto por nivel) y los convierte en reglas con
``ARM.association_rules``; como ningun itemset contiene un item y su antecesor, no salen
las reglas triviales ``item => antecesor`` (confianza 1).

Las categorias generan muchas reglas que solo repiten, a otro nivel, lo que ya dice una
regla mas general. ``interesting_rules`` conserva las "R-interesantes" de Srikant y Agrawal
(1995): una regla es interesante si no tiene antecesoras (reglas que resultan de sustituir
items por sus antecesores en el mismo conjunto de reglas) o si, respecto a cada una de sus
antecesoras CERCANAS interesantes, su soporte o su confianza reales son al menos R veces los
ESPERADOS a partir de la antecesora. Con ``X^ => Y^`` la antecesora de ``X => Y`` y ``z^`` la
antecesora de cada item ``z``:

    soporte esperado   = soporte(X^ u Y^) * prod_{z en X u Y} soporte(z) / soporte(z^)
    confianza esperada = confianza(X^ => Y^) * prod_{y en Y} soporte(y) / soporte(y^)

Es decir, la regla especializada solo aporta algo si es mejor de lo que cabria esperar si
cada item se comportase como su categoria.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..FIM.taxonomy import MinSupp, Taxonomy, ancestor_replacements, generalized_itemsets
from .association_rules import association_rules

_EPS = 1e-9


def generalized_rules(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy, min_supp: MinSupp,
                      metric: str = "confidence", min_threshold: float = 0.8, max_len: Optional[int] = None,
                      interest: Optional[float] = None) -> pd.DataFrame:
    """
    Minera reglas generalizadas (ver el docstring del modulo).

    Argumentos:
        transactions: una transaccion por elemento, cada una una coleccion de items (hojas o categorias)
        taxonomy (Taxonomy): jerarquia de items
        min_supp: soporte minimo, o ``{nivel: soporte}`` (ver ``FIM.taxonomy.generalized_itemsets``)
        metric, min_threshold: medida de ``ARM.association_rules`` y su umbral minimo
        max_len (int, opcional): maximo de items de una regla
        interest (float, opcional): si se da (>= 1), se aplica ``interesting_rules`` con ese R

    Retorna:
        pandas.DataFrame con el formato de ``ARM.association_rules`` mas la columna 'level' (el nivel
        del item mas profundo de la regla). ``attrs`` lleva 'n_transactions', 'n_candidates' (para
        ``ARM.significance``), 'taxonomy' y 'supports' (soporte de cada itemset frecuente, que usa
        ``interesting_rules``)
    """
    itemsets = generalized_itemsets(transactions, taxonomy, min_supp, max_len=max_len)
    total = itemsets.attrs["n_transactions"]
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold) if len(itemsets) \
        else pd.DataFrame(columns=["antecedents", "consequents", "antecedent support", "consequent support", "support",
                                   "confidence", "lift"])
    rules["level"] = [max(taxonomy.level(item) for item in set(a) | set(c))
                      for a, c in zip(rules["antecedents"], rules["consequents"])]
    rules.attrs["n_transactions"] = total
    rules.attrs.setdefault("n_candidates", len(rules))
    rules.attrs["taxonomy"] = taxonomy
    rules.attrs["supports"] = {itemset: float(support) for itemset, support in zip(itemsets["itemsets"], itemsets["support"])}
    return interesting_rules(rules, taxonomy, interest) if interest is not None and len(rules) else rules


def interesting_rules(rules: pd.DataFrame, taxonomy: Optional[Taxonomy] = None, interest: float = 1.1,
                      supports: Optional[Dict[FrozenSet[str], float]] = None) -> pd.DataFrame:
    """
    Las reglas R-interesantes de `rules` (ver el docstring del modulo), con R = `interest`.

    Argumentos:
        rules (pandas.DataFrame): salida de ``generalized_rules`` SIN podar (las antecesoras de una regla
            deben estar en el conjunto), con 'antecedents', 'consequents', 'support' y 'confidence'
        taxonomy (Taxonomy), supports (dict {frozenset: soporte}): por defecto los de ``rules.attrs``
        interest (float): R >= 1; 1 solo elimina las reglas peores que lo esperado

    Retorna:
        las reglas interesantes, en el mismo orden e indice, con la columna 'interest': el menor, sobre
        las antecesoras cercanas, de max(soporte real / esperado, confianza real / esperada); NaN si la
        regla no tiene antecesoras
    """
    taxonomy = taxonomy if taxonomy is not None else rules.attrs.get("taxonomy")
    supports = supports if supports is not None else rules.attrs.get("supports")
    if taxonomy is None or supports is None:
        raise ValueError("hacen falta la taxonomia y los soportes de los itemsets (rules.attrs, o los argumentos)")
    if interest < 1:
        raise ValueError("interest debe ser >= 1")
    result = rules.copy()
    result["interest"] = np.nan
    if len(rules) == 0:
        return result

    items: List[Tuple[str, ...]] = []
    sizes: List[int] = []
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        items.append(tuple(sorted(antecedent)) + tuple(sorted(consequent)))
        sizes.append(len(antecedent))
    lookup = {(frozenset(a), frozenset(c)): row for row, (a, c) in enumerate(zip(rules["antecedents"], rules["consequents"]))}
    support = rules["support"].to_numpy(dtype=float)
    confidence = rules["confidence"].to_numpy(dtype=float)

    def single(item: str) -> float:
        return supports[frozenset([item])]

    keep = np.zeros(len(rules), dtype=bool)
    ratios = np.full(len(rules), np.nan)
    for row in sorted(range(len(rules)), key=lambda r: sum(taxonomy.level(item) for item in items[r])):
        size = sizes[row]
        ancestors = []
        for combination in ancestor_replacements(items[row], taxonomy):
            antecedent, consequent = frozenset(combination[:size]), frozenset(combination[size:])
            if len(antecedent) != size or len(consequent) != len(items[row]) - size or antecedent & consequent:
                continue
            other = lookup.get((antecedent, consequent))
            if other is not None and keep[other]:
                ancestors.append((combination, other))
        close = [(combination, other) for combination, other in ancestors
                 if not any(farther != combination and all(c == f or c in taxonomy.ancestors(f)
                                                           for c, f in zip(combination, farther))
                            for farther, _ in ancestors)]
        best = np.inf
        for combination, other in close:
            scale = np.prod([single(item) / single(general) for item, general in zip(items[row], combination)])
            scale_consequent = np.prod([single(item) / single(general)
                                        for item, general in zip(items[row][size:], combination[size:])])
            best = min(best, max(support[row] / (support[other] * scale), confidence[row] / (confidence[other] * scale_consequent)))
        if close:
            ratios[row] = best
        keep[row] = not close or best >= interest - _EPS
    result["interest"] = ratios
    result = result[keep]
    result.attrs.update({key: value for key, value in rules.attrs.items()})
    return result
