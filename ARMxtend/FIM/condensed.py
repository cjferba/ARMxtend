# -*- coding: utf-8 -*-
"""
Representaciones condensadas de los itemsets frecuentes: itemsets CERRADOS
y MAXIMALES.

* Un itemset es CERRADO si ningun superconjunto suyo tiene exactamente su
  mismo soporte (Pasquier, Bastide, Taouil & Lakhal, 1999). Los cerrados
  conservan el soporte de TODOS los frecuentes (ver `support_from_closed`)
  con menos itemsets.
* Un itemset es MAXIMAL si ninguno de sus superconjuntos es frecuente
  (Bayardo, 1998). Es una representacion aun mas pequena, pero pierde los
  soportes de los subconjuntos.

Ambas funciones operan sobre cualquier diccionario {clave 'A-B-C': soporte}
producido por los algoritmos de ARMxtend, tanto crisp (soporte escalar:
``FFIM.fpgrowth``, ``FIM.apriori.DApriori``, ``FIM.Eclat.DECLAT``, ...) como
difusos (soporte = bit-list por alpha-corte: ``FFIM.fuzzy_fpgrowth``,
``FIM.Eclat.FuzzyDECLAT``, ...). En el caso difuso, dos itemsets "tienen el
mismo soporte" si coinciden en TODOS los alpha-cortes, es decir, la
clausura se define nivel a nivel, coherentemente con la RL-theory.

Basta comparar cada itemset con sus superconjuntos INMEDIATOS (un item mas):
el soporte es antimonotono, luego si algun superconjunto tuviera el mismo
soporte tambien lo tendria el itemset intermedio, que es frecuente y por
tanto esta en el diccionario.

@author Carlos Fernandez-Basso
"""
import numpy as np

from ._shared import key_to_itemset, itemset_to_key


def _same_support(first, second, tol):
    return bool(np.all(np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) <= tol))


def _immediate_subsets(freq_itemsets):
    """Genera (clave del itemset Z, clave de cada subconjunto X = Z - {item} que tambien es frecuente)."""
    for key in freq_itemsets:
        items = key_to_itemset(key)
        if len(items) < 2:
            continue
        for item in items:
            subsetKey = itemset_to_key(items - {item})
            if subsetKey in freq_itemsets:
                yield key, subsetKey


def closed_itemsets(freq_itemsets, tol=1e-12):
    """
    Itemsets frecuentes cerrados.

    Argumentos:
        freq_itemsets (dict[str, float | numpy.ndarray]): itemsets
            frecuentes con su soporte (escalar o bit-list por alpha-corte)
        tol (float): tolerancia absoluta para considerar iguales dos soportes

    Retorna:
        dict con las entradas de `freq_itemsets` que son cerradas
    """
    notClosed = set()
    for superset, subset in _immediate_subsets(freq_itemsets):
        if _same_support(freq_itemsets[superset], freq_itemsets[subset], tol):
            notClosed.add(subset)
    return {key: support for key, support in freq_itemsets.items() if key not in notClosed}


def maximal_itemsets(freq_itemsets):
    """
    Itemsets frecuentes maximales (sin ningun superconjunto frecuente).

    Argumentos:
        freq_itemsets (dict[str, float | numpy.ndarray]): itemsets frecuentes

    Retorna:
        dict con las entradas de `freq_itemsets` que son maximales
    """
    notMaximal = {subset for _, subset in _immediate_subsets(freq_itemsets)}
    return {key: support for key, support in freq_itemsets.items() if key not in notMaximal}


def support_from_closed(itemset, closed):
    """
    Recupera el soporte de un itemset frecuente a partir de los itemsets
    cerrados: es el MAXIMO (elemento a elemento, si es un bit-list) entre
    los soportes de los cerrados que lo contienen.

    Argumentos:
        itemset (str | Iterable[str]): clave 'A-B' o coleccion de items
        closed (dict): salida de `closed_itemsets`

    Retorna:
        El soporte del itemset, o None si ningun cerrado lo contiene (es
        decir, si no es frecuente).
    """
    items = key_to_itemset(itemset) if isinstance(itemset, str) else frozenset(itemset)
    supports = [support for key, support in closed.items() if items <= key_to_itemset(key)]
    if not supports:
        return None
    if np.ndim(supports[0]) == 0:
        return max(supports)
    return np.max(np.vstack(supports), axis=0)
