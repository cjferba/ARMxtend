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


def alpha_levels(num_alpha):
    """
    Conjunto Lambda = {alpha_1, ..., alpha_p} de niveles alpha equidistantes
    usado para decomponer medidas difusas en la teoria de Representacion por
    Niveles (RL-theory) de Delgado, Ruiz, Sanchez & Serrano (2011, Information
    Sciences 181) y en Fernandez-Basso, Ruiz & Martin-Bautista (2021, Int. J.
    of Approximate Reasoning 137): alpha_i = 1 - (i-1)/p, para i = 1..p, con
    1 = alpha_1 > alpha_2 > ... > alpha_p > alpha_(p+1) = 0.

    Por ejemplo, para num_alpha=4: [1, 0.75, 0.5, 0.25] (el mismo ejemplo
    numerico usado en ambos papers).

    Argumentos:
        num_alpha (int): numero de niveles alpha (p), >= 1

    Retorna:
        numpy.ndarray de longitud `num_alpha`, decreciente, con el primer
        valor igual a 1.0
    """
    if num_alpha < 1:
        raise ValueError("El numero de alpha-cortes debe ser >= 1, recibido: {}".format(num_alpha))
    return np.arange(num_alpha, 0, -1) / float(num_alpha)


def alpha_weights(levels):
    """
    Pesos (alpha_i - alpha_(i+1)) de cada nivel alpha, con alpha_(p+1) = 0,
    usados para agregar una medida calculada en cada nivel (ver `alpha_levels`,
    Ecs. (2)-(6) de Fernandez-Basso et al. 2021 / Ruiz et al. 2016). Para
    niveles equidistantes (ver `alpha_levels`) todos los pesos son iguales a
    1/num_alpha, por lo que la agregacion resultante es una media aritmetica.

    Argumentos:
        levels (numpy.ndarray): niveles alpha, en orden decreciente

    Retorna:
        numpy.ndarray de los mismos longitud que `levels`
    """
    return levels - np.append(levels[1:], 0.0)


def weighted_alpha_aggregate(values_per_level, weights):
    """
    Agregacion ponderada Σ_i weight_i * value_i de una medida calculada en
    cada nivel alpha (ver `alpha_weights`). Es el mecanismo central de la
    RL-theory para generalizar cualquier medida crisp (soporte, confianza,
    factor de certeza, ...) al caso difuso, sustituyendo la nocion de
    "elegir un alpha-corte" por una integracion sobre todos los niveles.

    Argumentos:
        values_per_level (array-like): valor de la medida crisp en cada
            nivel alpha (misma longitud que `weights`)
        weights (array-like): pesos de cada nivel (ver `alpha_weights`)

    Retorna:
        float: la medida difusa agregada
    """
    return float(np.dot(np.asarray(values_per_level, dtype=float), weights))


def alpha_cuts(value, num_alpha):
    """
    Representacion en alpha-cortes (bit-list) de un grado de pertenencia
    difuso, siguiendo el procedimiento FuzzyToArray (Algoritmo 2) de
    Fernandez-Basso, Ruiz & Martin-Bautista (2021): para cada nivel
    alpha_i en `alpha_levels(num_alpha)`, la posicion i vale 1 si
    value >= alpha_i, y 0 en otro caso.

    Por ejemplo, alpha_cuts(0.25, 4) = [0, 0, 0, 1] con niveles
    [1, 0.75, 0.5, 0.25] (ejemplo textual de ambos papers).

    Argumentos:
        value (float): grado de pertenencia difuso, en [0, 1]
        num_alpha (int): numero de niveles/alpha-cortes a considerar (p)

    Retorna:
        numpy.ndarray de enteros (0/1) de longitud `num_alpha`, no-creciente
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError("El grado de pertenencia difuso debe estar en [0, 1], recibido: {}".format(value))

    return (value >= alpha_levels(num_alpha)).astype(int)


def fuzzy_support(itemset, transaction_memberships, num_alpha):
    """
    Bit-list por alpha-corte de un itemset en una transaccion: dado que
    t(A) = min_{i in A} t(i) (Definicion 2, fuzzy transaction), el bit-list
    de A en cada nivel alpha es el minimo (interseccion logica AND, ya que
    son 0/1) de los bit-list de sus items.

    Argumentos:
        itemset (Iterable[str]): items del itemset a evaluar
        transaction_memberships (dict[str, numpy.ndarray]): para cada item de
            la transaccion, su bit-list de alpha-cortes (ver `alpha_cuts`)
        num_alpha (int): numero de alpha-cortes usados

    Retorna:
        numpy.ndarray: el bit-list de alpha-cortes del itemset, o un vector
        de ceros si falta alguno de sus items en la transaccion.
    """
    result = np.ones(num_alpha, dtype=int)
    for item in itemset:
        membership = transaction_memberships.get(item)
        if membership is None:
            return np.zeros(num_alpha, dtype=int)
        result = np.minimum(result, membership)
    return result
