# -*- coding: utf-8 -*-
"""
Datos para probar y comparar: generadores sinteticos con patrones plantados (de modo que se sabe que deberia
encontrar un algoritmo) y unos pocos conjuntos de ejemplo pequenos, incluidos en el paquete (no se descarga nada).

Generadores (todos reproducibles con `random_state`):

* ``make_transactions``: cestas de la compra al estilo del generador de IBM Quest (Agrawal y Srikant, 1994): se
  eligen unos patrones frecuentes, y cada transaccion es una mezcla de patrones (con algun item perdido) y de items
  sueltos.
* ``make_sequences``: secuencias de elementos (conjuntos de items) con patrones secuenciales plantados.
* ``make_quantities``: cantidades compradas y beneficios unitarios, para ``ARM.high_utility_rules``.
* ``make_labeled_items``: tabla booleana y clase, donde la clase depende de unas pocas condiciones, para
  ``ARM.RuleClassifier``.

Conjuntos de ejemplo (``load_dataset``): ``"han"`` (9 transacciones, el ejemplo de Han y Kamber), ``"weather"`` (14
dias de la tabla del tenis de Quinlan, con la clase), ``"sales"`` (cantidades y beneficios, el ejemplo de utilidad) y
``"routines"`` (secuencias de eventos de una casa).

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd

Seed = Union[None, int, np.random.Generator]


def _rng(random_state: Seed) -> np.random.Generator:
    return random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)


def _names(prefix: str, count: int) -> List[str]:
    width = len(str(count - 1))
    return ["{}{}".format(prefix, str(number).zfill(width)) for number in range(count)]


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if value < 1:
            raise ValueError("{} debe ser >= 1".format(name))


def _planted(rng: np.random.Generator, items: List[str], n_patterns: int, pattern_length: int) -> List[List[str]]:
    # popularidad de los items decreciente (tipo Zipf): los patrones usan sobre todo los primeros
    weights = 1.0 / np.arange(1, len(items) + 1)
    weights /= weights.sum()
    size = min(pattern_length, len(items))
    return [[items[position] for position in sorted(rng.choice(len(items), size=size, replace=False, p=weights))]
            for _ in range(n_patterns)]


def make_transactions(n_transactions: int = 1000, n_items: int = 100, avg_length: float = 8.0, n_patterns: int = 20,
                      pattern_length: int = 4, corruption: float = 0.2, random_state: Seed = None,
                      return_patterns: bool = False) -> Union[List[List[str]], Tuple[List[List[str]], List[List[str]]]]:
    """
    Transacciones sinteticas con patrones frecuentes plantados.

    Argumentos:
        n_transactions, n_items: tamano de los datos (los items se llaman ``i00``, ``i01``...)
        avg_length: items medios por transaccion (Poisson, al menos 1)
        n_patterns, pattern_length: patrones plantados y su longitud; su frecuencia sigue una exponencial, de modo
            que unos pocos son muy frecuentes y otros raros
        corruption: probabilidad de perder cada item de un patron al ponerlo en una transaccion
        return_patterns: si es verdadero, devuelve tambien los patrones plantados

    Retorna:
        Lista de transacciones (listas de items sin repetir, ordenadas); con `return_patterns` la tupla
        ``(transacciones, patrones)``.
    """
    _check_positive(n_transactions=n_transactions, n_items=n_items, n_patterns=n_patterns, pattern_length=pattern_length)
    if not 0 <= corruption < 1:
        raise ValueError("corruption debe estar en [0, 1)")
    rng = _rng(random_state)
    items = _names("i", n_items)
    patterns = _planted(rng, items, n_patterns, pattern_length)
    weights = rng.exponential(size=n_patterns)
    weights /= weights.sum()
    popularity = 1.0 / np.arange(1, n_items + 1)
    popularity /= popularity.sum()
    transactions = []
    for _ in range(n_transactions):
        target = min(n_items, max(1, rng.poisson(avg_length)))
        basket: set = set()
        while len(basket) < target:
            if rng.random() < 0.5:
                pattern = patterns[rng.choice(n_patterns, p=weights)]
                basket.update(item for item in pattern if rng.random() >= corruption)
            else:                                           # un item suelto que aun no este, segun su popularidad
                free = np.array([item not in basket for item in items])
                chance = popularity * free
                basket.add(items[rng.choice(n_items, p=chance / chance.sum())])
        transactions.append(sorted(basket))
    return (transactions, patterns) if return_patterns else transactions


def make_sequences(n_sequences: int = 500, n_items: int = 30, avg_elements: float = 6.0, avg_element_size: float = 1.5,
                   n_patterns: int = 5, pattern_length: int = 3, random_state: Seed = None,
                   return_patterns: bool = False) -> Union[list, Tuple[list, list]]:
    """
    Secuencias sinteticas (listas de elementos, cada uno una lista de items) con patrones secuenciales plantados.

    Cada secuencia es una serie de elementos al azar; con probabilidad 1/2 se intercala un patron plantado, en
    orden, con otros elementos entre medias. Un patron es una lista de items (un item por elemento).

    Retorna:
        Lista de secuencias; con `return_patterns` la tupla ``(secuencias, patrones)``.
    """
    _check_positive(n_sequences=n_sequences, n_items=n_items, n_patterns=n_patterns, pattern_length=pattern_length)
    rng = _rng(random_state)
    items = _names("e", n_items)
    patterns = [[items[position] for position in rng.choice(n_items, size=min(pattern_length, n_items), replace=False)]
                for _ in range(n_patterns)]

    def element() -> List[str]:
        size = min(n_items, max(1, rng.poisson(avg_element_size)))
        return sorted(items[position] for position in rng.choice(n_items, size=size, replace=False))

    sequences = []
    for _ in range(n_sequences):
        length = max(1, rng.poisson(avg_elements))
        sequence = [element() for _ in range(length)]
        if rng.random() < 0.5:
            pattern = patterns[rng.integers(n_patterns)]
            sequence = _interleave(sequence, pattern, rng)
        sequences.append(sequence)
    return (sequences, [[[item] for item in pattern] for pattern in patterns]) if return_patterns else sequences


def _interleave(sequence: list, pattern: List[str], rng: np.random.Generator) -> list:
    """Sequence con los items de `pattern` en elementos distintos y en el mismo orden."""
    cuts = sorted(rng.choice(len(sequence) + 1, size=len(pattern), replace=True))
    result: list = []
    done = 0
    for position, element in enumerate(sequence + [None]):
        while done < len(pattern) and cuts[done] == position:
            result.append([pattern[done]])
            done += 1
        if element is not None:
            result.append(element)
    return result


def make_quantities(n_transactions: int = 1000, n_items: int = 30, avg_length: float = 5.0, max_quantity: int = 5,
                    random_state: Seed = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Cantidades compradas (una fila por transaccion, una columna por item, 0 si no se compro) y beneficio unitario
    de cada item, para ``ARM.high_utility_rules``. Los beneficios siguen una log-normal: pocos items valen mucho.
    """
    _check_positive(n_transactions=n_transactions, n_items=n_items, max_quantity=max_quantity)
    rng = _rng(random_state)
    baskets = cast(List[List[str]], make_transactions(n_transactions, n_items, avg_length,
                                                      n_patterns=max(1, n_items // 3), pattern_length=3, random_state=rng))
    items = _names("i", n_items)
    table = np.zeros((n_transactions, n_items), dtype=int)
    position = {item: number for number, item in enumerate(items)}
    for row, basket in enumerate(baskets):
        for item in basket:
            table[row, position[item]] = rng.integers(1, max_quantity + 1)
    profits = {item: float(round(value, 2)) for item, value in zip(items, rng.lognormal(mean=1.0, sigma=0.8, size=n_items))}
    return pd.DataFrame(table, columns=items), profits


def make_labeled_items(n_samples: int = 500, n_features: int = 12, n_classes: int = 2, n_rules: int = 4,
                       noise: float = 0.05, random_state: Seed = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tabla booleana ``items`` (columnas ``f00``, ``f01``...) y la clase ``labels`` de cada fila.

    La clase la deciden `n_rules` reglas de dos condiciones, ``f_a y f_b -> clase`` (la primera que se cumpla); si
    ninguna, la clase 0. Con probabilidad `noise` la clase de una fila se cambia por otra al azar, de modo que una
    exactitud cercana a ``1 - noise`` es lo mejor que se puede esperar.
    """
    _check_positive(n_samples=n_samples, n_features=n_features, n_rules=n_rules)
    if n_classes < 2 or n_features < 2:
        raise ValueError("n_classes y n_features deben ser >= 2")
    if not 0 <= noise < 1:
        raise ValueError("noise debe estar en [0, 1)")
    rng = _rng(random_state)
    names = _names("f", n_features)
    items = pd.DataFrame(rng.random((n_samples, n_features)) < 0.4, columns=names)
    labels = np.zeros(n_samples, dtype=int)
    free = np.ones(n_samples, dtype=bool)
    for rule in range(n_rules):
        first, second = rng.choice(n_features, size=2, replace=False)
        fires = free & items.iloc[:, first].to_numpy() & items.iloc[:, second].to_numpy()
        labels[fires] = 1 + rule % (n_classes - 1)
        free &= ~fires
    flip = rng.random(n_samples) < noise
    labels[flip] = rng.integers(0, n_classes, size=int(flip.sum()))
    return items, pd.Series(labels, name="class")


class Dataset(dict):
    """Un conjunto de datos: un diccionario cuyas claves se leen tambien como atributos (``data.data``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


_HAN = [["I1", "I2", "I5"], ["I2", "I4"], ["I2", "I3"], ["I1", "I2", "I4"], ["I1", "I3"], ["I2", "I3"], ["I1", "I3"],
        ["I1", "I2", "I3", "I5"], ["I1", "I2", "I3"]]

_WEATHER = [("sunny", "hot", "high", "false", "no"), ("sunny", "hot", "high", "true", "no"),
            ("overcast", "hot", "high", "false", "yes"), ("rainy", "mild", "high", "false", "yes"),
            ("rainy", "cool", "normal", "false", "yes"), ("rainy", "cool", "normal", "true", "no"),
            ("overcast", "cool", "normal", "true", "yes"), ("sunny", "mild", "high", "false", "no"),
            ("sunny", "cool", "normal", "false", "yes"), ("rainy", "mild", "normal", "false", "yes"),
            ("sunny", "mild", "normal", "true", "yes"), ("overcast", "mild", "high", "true", "yes"),
            ("overcast", "hot", "normal", "false", "yes"), ("rainy", "mild", "high", "true", "no")]

_SALES = [{"a": 1, "c": 1, "d": 1},
          {"a": 2, "c": 6, "e": 2, "g": 5},
          {"a": 1, "b": 2, "c": 1, "d": 6, "e": 1, "f": 5},
          {"b": 4, "c": 3, "d": 3, "f": 1},
          {"b": 2, "c": 2, "e": 1, "g": 2}]

_ROUTINES = [[["alarm"], ["bathroom"], ["kitchen", "coffee"], ["leave"]],
             [["alarm"], ["kitchen", "coffee"], ["bathroom"], ["leave"]],
             [["alarm"], ["bathroom"], ["kitchen"], ["tv"]],
             [["alarm"], ["bathroom"], ["kitchen", "coffee"], ["tv"], ["bed"]],
             [["bathroom"], ["kitchen"], ["leave"]],
             [["alarm"], ["kitchen", "coffee"], ["leave"]]]


def load_dataset(name: str) -> Dataset:
    """
    Un conjunto de ejemplo del paquete (ver ``list_datasets``). Devuelve un ``Dataset`` con ``data``, ``description`` y,
    segun el caso, ``target`` (la clase), ``profits``... Cada llamada devuelve copias: se pueden modificar.
    """
    if name == "han":
        return Dataset(name=name, data=[list(basket) for basket in _HAN], description=(
            "9 transactions over 5 items (Han & Kamber, Data Mining: Concepts and Techniques, the AllElectronics "
            "example). With min_supp=2/9, {I1, I2, I5} is frequent."))
    if name == "weather":
        columns = ["outlook", "temperature", "humidity", "windy", "play"]
        table = pd.DataFrame(_WEATHER, columns=columns)
        labelled = {"{}={}".format(column, value): (table[column] == value)
                    for column in columns[:-1] for value in sorted(table[column].unique())}
        items = pd.DataFrame(labelled)
        return Dataset(name=name, data=items, target=table["play"].rename("class"), table=table, description=(
            "14 days, 4 categorical attributes (items 'outlook=sunny'...) and the class 'play' (Quinlan's tennis "
            "table). Use data and target with RuleClassifier."))
    if name == "sales":
        return Dataset(name=name, data=pd.DataFrame(_SALES).fillna(0).astype(int),
                       profits={"a": 5.0, "b": 2.0, "c": 1.0, "d": 2.0, "e": 3.0, "f": 1.0, "g": 1.0}, description=(
                           "5 sales of 7 items (quantities) and the unit profit of each: the high-utility example."))
    if name == "routines":
        return Dataset(name=name, data=[[list(element) for element in sequence] for sequence in _ROUTINES],
                       description="6 days of events of a house, one sequence per day (alarm, bathroom, kitchen...).")
    raise ValueError("conjunto desconocido '{}'; los disponibles son {}".format(name, list_datasets()))


def list_datasets() -> List[str]:
    """Nombres de los conjuntos de ejemplo de ``load_dataset``."""
    return ["han", "weather", "sales", "routines"]
