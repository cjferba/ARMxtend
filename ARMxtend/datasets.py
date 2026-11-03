# -*- coding: utf-8 -*-
"""
Data to test and compare: synthetic generators with planted patterns (so that it is known what an algorithm should
find) and a few small example datasets, included in the package (nothing is downloaded).

Generators (all reproducible with `random_state`):

* ``make_transactions``: shopping baskets in the style of the IBM Quest generator (Agrawal and Srikant, 1994): some
  frequent patterns are chosen, and each transaction is a mixture of patterns (with some item lost) and of loose
  items.
* ``make_sequences``: sequences of elements (sets of items) with planted sequential patterns.
* ``make_quantities``: purchased quantities and unit profits, for ``ARM.high_utility_rules``.
* ``make_labeled_items``: boolean table and class, where the class depends on a few conditions, for
  ``ARM.RuleClassifier``.

Example datasets (``load_dataset``): ``"han"`` (9 transactions, the example of Han and Kamber), ``"weather"`` (14 days
of Quinlan's tennis table, with the class), ``"sales"`` (quantities and profits, the utility example) and
``"routines"`` (event sequences of a house).

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
    # decreasing item popularity (Zipf-like): the patterns mostly use the first ones
    weights = 1.0 / np.arange(1, len(items) + 1)
    weights /= weights.sum()
    size = min(pattern_length, len(items))
    return [[items[position] for position in sorted(rng.choice(len(items), size=size, replace=False, p=weights))]
            for _ in range(n_patterns)]


def make_transactions(n_transactions: int = 1000, n_items: int = 100, avg_length: float = 8.0, n_patterns: int = 20,
                      pattern_length: int = 4, corruption: float = 0.2, random_state: Seed = None,
                      return_patterns: bool = False) -> Union[List[List[str]], Tuple[List[List[str]], List[List[str]]]]:
    """
    Synthetic transactions with planted frequent patterns.

    Args:
        n_transactions, n_items: size of the data (the items are called ``i00``, ``i01``...)
        avg_length: average items per transaction (Poisson, at least 1)
        n_patterns, pattern_length: planted patterns and their length; their frequency follows an exponential, so a few
            are very frequent and others rare
        corruption: probability of losing each item of a pattern when putting it in a transaction
        return_patterns: if true, also return the planted patterns

    Returns:
        List of transactions (lists of items without repeats, sorted); with `return_patterns` the tuple
        ``(transactions, patterns)``.
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
            else:                                           # a loose item not yet present, according to its popularity
                free = np.array([item not in basket for item in items])
                chance = popularity * free
                basket.add(items[rng.choice(n_items, p=chance / chance.sum())])
        transactions.append(sorted(basket))
    return (transactions, patterns) if return_patterns else transactions


def make_sequences(n_sequences: int = 500, n_items: int = 30, avg_elements: float = 6.0, avg_element_size: float = 1.5,
                   n_patterns: int = 5, pattern_length: int = 3, random_state: Seed = None,
                   return_patterns: bool = False) -> Union[list, Tuple[list, list]]:
    """
    Synthetic sequences (lists of elements, each a list of items) with planted sequential patterns.

    Each sequence is a series of random elements; with probability 1/2 a planted pattern is interleaved, in order, with
    other elements in between. A pattern is a list of items (one item per element).

    Returns:
        List of sequences; with `return_patterns` the tuple ``(sequences, patterns)``.
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
    """Sequence with the items of `pattern` in distinct elements and in the same order."""
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
    Purchased quantities (one row per transaction, one column per item, 0 if not bought) and unit profit of each item,
    for ``ARM.high_utility_rules``. The profits follow a log-normal: few items are worth a lot.
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
    Boolean table ``items`` (columns ``f00``, ``f01``...) and the class ``labels`` of each row.

    The class is decided by `n_rules` rules of two conditions, ``f_a and f_b -> class`` (the first one that holds); if
    none holds, class 0. With probability `noise` the class of a row is changed to another at random, so an accuracy
    close to ``1 - noise`` is the best that can be expected.
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
    """A dataset: a dictionary whose keys are also read as attributes (``data.data``)."""

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
    An example dataset of the package (see ``list_datasets``). It returns a ``Dataset`` with ``data``, ``description``
    and, depending on the case, ``target`` (the class), ``profits``... Each call returns copies: they can be modified.
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
    raise ValueError("unknown dataset '{}'; the available ones are {}".format(name, list_datasets()))


def list_datasets() -> List[str]:
    """Names of the example datasets of ``load_dataset``."""
    return ["han", "weather", "sales", "routines"]
