# -*- coding: utf-8 -*-
"""
Utilidades compartidas por los algoritmos de mineria de itemsets frecuentes
(crisp y difusos, secuenciales y en Big Data) de ARMxtend.

Estas funciones sustituyen a las multiples copias, ligeramente distintas y
con errores, de ``GenItems``/``AlphaCortes`` que existian duplicadas en
``FIM/apriori.py``, ``FIM/BD_ARE``, ``FIM/BD_FARE`` y ``FFIM``.

@author Carlos Fernandez-Basso
"""
import itertools

import numpy as np

ITEM_SEP = "-"


def itemset_to_key(itemset):
    """Convierte un itemset (iterable de items) en su clave canonica ordenada."""
    return ITEM_SEP.join(sorted(itemset))


def key_to_itemset(key):
    """Convierte una clave canonica ('A-B-C') en un frozenset de items."""
    return frozenset(key.split(ITEM_SEP))


def generate_candidates(freq_itemsets):
    """
    Genera candidatos de longitud k+1 a partir de un conjunto de itemsets
    frecuentes de longitud k, siguiendo el procedimiento apriori-gen
    (Agrawal & Srikant, 1994): dos itemsets frecuentes se combinan si
    comparten sus primeros k-1 items (una vez ordenados lexicograficamente),
    y el candidato resultante solo se conserva si TODOS sus subconjuntos de
    longitud k son tambien frecuentes (propiedad de cierre descendente).

    Argumentos:
        freq_itemsets (Iterable[str]): claves ('A-B-C') de los itemsets
            frecuentes de longitud k

    Retorna:
        Lista ordenada de claves de itemsets candidatos de longitud k+1
    """
    itemsets = sorted(key_to_itemset(k) for k in freq_itemsets)
    freq_set = set(itemsets)
    length = (len(itemsets[0]) + 1) if itemsets else 1

    sorted_items = [tuple(sorted(s)) for s in itemsets]
    candidates = set()
    n = len(sorted_items)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted_items[i], sorted_items[j]
            # Solo se combinan itemsets que comparten el mismo prefijo
            # (longitud k-1), difiriendo unicamente en el ultimo item
            if a[:-1] != b[:-1]:
                break
            candidate = frozenset(a) | frozenset(b)
            if len(candidate) != length:
                continue
            if all(frozenset(sub) in freq_set
                   for sub in itertools.combinations(candidate, length - 1)):
                candidates.add(candidate)

    return sorted(itemset_to_key(c) for c in candidates)


def alpha_cuts(value, num_alpha):
    """
    Representacion en alpha-cortes de un grado de pertenencia difuso.

    Divide el intervalo [0, 1] en `num_alpha` niveles alpha equiespaciados,
    alpha_i = i / num_alpha para i = 0..num_alpha-1 (de menor a mayor
    exigencia), y devuelve un vector binario que indica en que alpha-cortes
    se conserva el valor (value > alpha_i). Al ser los alpha_i estrictamente
    crecientes, el vector resultante es siempre no-creciente: en cuanto un
    valor deja de superar un nivel exigente, tampoco supera los siguientes,
    mas exigentes aun. Un valor de 1.0 supera siempre los `num_alpha`
    niveles (pertenece a todos los alpha-cortes); un valor de 0.0 no supera
    ninguno.

    Argumentos:
        value (float): grado de pertenencia difuso, en [0, 1]
        num_alpha (int): numero de niveles/alpha-cortes a considerar

    Retorna:
        numpy.ndarray de enteros (0/1) de longitud `num_alpha`
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError("El grado de pertenencia difuso debe estar en [0, 1], recibido: {}".format(value))
    if num_alpha < 1:
        raise ValueError("El numero de alpha-cortes debe ser >= 1, recibido: {}".format(num_alpha))

    alphas = np.arange(num_alpha) / float(num_alpha)
    return (value > alphas).astype(int)


def fuzzy_support(itemset, transaction_memberships, num_alpha):
    """
    Soporte difuso (vector de alpha-cortes) de un itemset en una transaccion.

    Argumentos:
        itemset (Iterable[str]): items del itemset a evaluar
        transaction_memberships (dict[str, numpy.ndarray]): para cada item de
            la transaccion, su vector de alpha-cortes (ya calculado con
            `alpha_cuts`)
        num_alpha (int): numero de alpha-cortes usados

    Retorna:
        numpy.ndarray: el minimo (interseccion, vía t-norma producto/min)
        de los vectores de alpha-cortes de los items del itemset presentes
        en la transaccion, o un vector de ceros si falta alguno.
    """
    result = np.ones(num_alpha, dtype=int)
    for item in itemset:
        membership = transaction_memberships.get(item)
        if membership is None:
            return np.zeros(num_alpha, dtype=int)
        result = np.minimum(result, membership)
    return result
