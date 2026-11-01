# -*- coding: utf-8 -*-
"""
Patrones secuenciales DIFUSOS: secuencias de eventos con un grado de pertenencia
en (0, 1] ("la temperatura fue ALTA con grado 0.8, y despues la calefaccion
estuvo ENCENDIDA con grado 0.6"), de modo que un patron se cumple en una
secuencia en cierto grado y no de forma todo o nada.

Un patron es una lista ordenada de items ``<x1 x2 ... xk>``. Su grado en una
secuencia es el MEJOR de sus encajes: el maximo, sobre las posiciones
estrictamente crecientes ``p1 < p2 < ... < pk``, de la t-norma de los grados
``d(p1, x1), ..., d(pk, xk)``. El soporte difuso es la media de ese grado sobre
todas las secuencias, y se exige ``soporte >= min_supp``. Con la t-norma
``min`` (por defecto) o el producto, extender un patron nunca aumenta su grado,
de modo que la poda por soporte es correcta (la propiedad antimonotona de
PrefixSpan sigue valiendo). Si todos los grados son 1, el soporte difuso es el
soporte habitual de los patrones cuyos elementos tienen un solo item (los
patrones de ``FIM.sequential`` con elementos de varios items no se buscan aqui).

Una secuencia es una lista de elementos; cada elemento puede ser un diccionario
``{item: grado}``, una lista de pares ``(item, grado)``, un item suelto o una
coleccion de items (todos con grado 1). Los items con grado 0 se ignoran.
``fuzzy_sequences_from_memberships`` construye secuencias a partir de una tabla
de grados de pertenencia (una fila por instante), p.ej. la que da la
fuzzificacion de series temporales.

NOTA: los nombres de item no deben contener '-'.

@author Carlos Fernandez-Basso
"""
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..preprocessing.timeseries import check_item_names

TNORMS = ("min", "product")
_EPS = 1e-9

Degrees = Dict[str, float]
FuzzySequence = List[Degrees]


def _check_tnorm(tnorm: str) -> Callable[[Any, Any], Any]:
    if tnorm == "min":
        return np.minimum
    if tnorm == "product":
        return np.multiply
    raise ValueError("tnorm debe ser una de {}, recibida: '{}'".format(TNORMS, tnorm))


def _element_degrees(element: Any) -> Degrees:
    """Un elemento como diccionario {item: grado} (el grado maximo si se repite; se ignoran los grados 0)."""
    if isinstance(element, Mapping):
        pairs: Iterable[Tuple[Any, Any]] = element.items()
    elif isinstance(element, str):
        pairs = [(element, 1.0)]
    else:
        pairs = [(entry, 1.0) if isinstance(entry, str) else tuple(entry) for entry in element]
    result: Degrees = {}
    for item, degree in pairs:
        degree = float(degree)
        if not 0.0 <= degree <= 1.0:
            raise ValueError("los grados de pertenencia deben estar en [0, 1], recibido {} para '{}'".format(degree, item))
        if degree > 0:
            result[str(item)] = max(result.get(str(item), 0.0), degree)
    return result


def normalize(sequences: Iterable[Iterable[Any]]) -> List[FuzzySequence]:
    """Cada secuencia como lista de diccionarios {item: grado}, sin elementos vacios."""
    result = []
    for sequence in sequences:
        elements = [_element_degrees(element) for element in sequence]
        result.append([element for element in elements if element])
    return result


def degree_tensor(sequences: Sequence[FuzzySequence]) -> Tuple[np.ndarray, List[str]]:
    """
    Tensor de grados (secuencias, posiciones, items) y el vocabulario ordenado de items; las
    secuencias mas cortas se rellenan con ceros, que no aportan nada a ningun patron.
    """
    vocabulary = sorted({item for sequence in sequences for element in sequence for item in element})
    check_item_names(vocabulary)
    column = {item: position for position, item in enumerate(vocabulary)}
    longest = max([len(sequence) for sequence in sequences] or [0])
    tensor = np.zeros((len(sequences), max(longest, 1), len(vocabulary)))
    for row, sequence in enumerate(sequences):
        for position, element in enumerate(sequence):
            for item, degree in element.items():
                tensor[row, position, column[item]] = degree
    return tensor, vocabulary


def _allowed(best_end: np.ndarray) -> np.ndarray:
    """Para cada posicion, el mejor encaje del prefijo que termina estrictamente antes."""
    cumulative = np.maximum.accumulate(best_end, axis=1)
    allowed = np.zeros_like(cumulative)
    allowed[:, 1:] = cumulative[:, :-1]
    return allowed


def fuzzy_prefixspan(sequences: Iterable[Iterable[Any]], min_supp: float, max_len: Optional[int] = None,
                     min_len: int = 1, tnorm: str = "min") -> pd.DataFrame:
    """
    Patrones secuenciales difusos frecuentes (ver el docstring del modulo).

    Argumentos:
        sequences: una secuencia por elemento (ver el docstring del modulo)
        min_supp (float): soporte difuso minimo, en (0, 1]
        max_len (int, opcional): maximo de items de un patron
        min_len (int): minimo de items de un patron que se devuelve
        tnorm (str): 'min' (por defecto) o 'product'

    Retorna:
        pandas.DataFrame con las columnas 'support' (difuso), 'sequence' (tupla
        de ``frozenset`` de un solo item, como en ``FIM.prefixspan``) y 'length',
        por soporte decreciente y longitud creciente. Memoria: proporcional a
        secuencias x longitud maxima x items
    """
    combine = _check_tnorm(tnorm)
    if not 0 < min_supp <= 1:
        raise ValueError("min_supp debe estar en (0, 1]")
    database = normalize(sequences)
    total = len(database)
    if total == 0:
        return pd.DataFrame(columns=["support", "sequence", "length"])
    tensor, vocabulary = degree_tensor(database)
    found: List[Tuple[float, Tuple[str, ...]]] = []

    def grow(prefix: Tuple[str, ...], allowed: np.ndarray) -> None:
        extension = combine(allowed[:, :, None], tensor)               # (secuencias, posiciones, items)
        support = extension.max(axis=1).mean(axis=0)
        for item in np.flatnonzero(support >= min_supp - _EPS):
            extended = prefix + (vocabulary[item],)
            found.append((float(support[item]), extended))
            if max_len is None or len(extended) < max_len:
                grow(extended, _allowed(extension[:, :, item]))

    if max_len is None or max_len >= 1:
        grow((), np.ones(tensor.shape[:2]))
    rows = [(support, tuple(frozenset([item]) for item in pattern), len(pattern)) for support, pattern in found]
    result = pd.DataFrame(rows, columns=["support", "sequence", "length"])
    result = result[result["length"] >= min_len]
    return result.sort_values(["support", "length"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def fuzzy_pattern_degree(sequence: Iterable[Any], pattern: Iterable[str], tnorm: str = "min") -> float:
    """
    Grado en que el patron ``<x1 ... xk>`` (lista de items) se cumple en una secuencia: el
    mejor de sus encajes (ver el docstring del modulo). 0 si no aparece en orden.
    """
    combine = _check_tnorm(tnorm)
    elements = normalize([sequence])[0]
    allowed = np.ones(max(len(elements), 1))
    best = allowed
    for item in pattern:
        degrees = np.array([element.get(item, 0.0) for element in elements] or [0.0])
        best = combine(allowed, degrees)
        allowed = np.zeros_like(best)
        allowed[1:] = np.maximum.accumulate(best)[:-1]
    return float(best.max()) if len(best) else 0.0


def fuzzy_sequences_from_memberships(memberships: pd.DataFrame, by: Union[Sequence[Any], np.ndarray, pd.Series, None] = None,
                                     min_degree: float = 0.0) -> List[FuzzySequence]:
    """
    Secuencias difusas a partir de una tabla de grados de pertenencia con una fila por
    instante (ordenadas en el tiempo): cada fila es un elemento, con los items cuyo grado
    supera `min_degree`.

    Argumentos:
        memberships (pandas.DataFrame): grados en [0, 1], una columna por item
        by: a que secuencia pertenece cada fila (p.ej. el dia: ``index.date``); por
            defecto, todas las filas forman una sola secuencia. Se conserva el orden
            de las filas dentro de cada secuencia, y las secuencias salen por orden de
            primera aparicion del grupo
        min_degree (float): grado minimo (exclusivo) para que un item cuente

    Retorna:
        lista de secuencias, cada una una lista de diccionarios {item: grado}; las filas sin
        ningun item por encima de `min_degree` no producen elemento
    """
    check_item_names(memberships.columns)
    degrees = memberships.to_numpy(dtype=float)
    if np.isnan(degrees).any() or (degrees < 0).any() or (degrees > 1).any():
        raise ValueError("los grados de pertenencia deben estar en [0, 1]")
    names = [str(name) for name in memberships.columns]
    groups = np.zeros(len(memberships), dtype=int) if by is None else pd.factorize(np.asarray(by))[0]
    if len(groups) != len(memberships):
        raise ValueError("`by` debe tener una entrada por fila ({}), tiene {}".format(len(memberships), len(groups)))
    sequences: Dict[int, FuzzySequence] = {}
    for row, group in zip(degrees, groups):
        element = {names[column]: float(row[column]) for column in np.flatnonzero(row > min_degree)}
        sequence = sequences.setdefault(int(group), [])
        if element:
            sequence.append(element)
    return [sequences[group] for group in sorted(sequences)]
