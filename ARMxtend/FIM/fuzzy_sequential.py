# -*- coding: utf-8 -*-
"""
FUZZY sequential patterns: sequences of events with a membership degree in (0, 1] ("the temperature was HIGH to
degree 0.8, and then the heating was ON to degree 0.6"), so that a pattern holds in a sequence to some degree rather
than all or nothing.

A pattern is an ordered list of items ``<x1 x2 ... xk>``. Its degree in a sequence is the BEST of its matches: the
maximum, over the strictly increasing positions ``p1 < p2 < ... < pk``, of the t-norm of the degrees
``d(p1, x1), ..., d(pk, xk)``. The fuzzy support is the mean of that degree over all the sequences, and
``support >= min_supp`` is required. With the ``min`` t-norm (default) or the product, extending a pattern never
raises its degree, so pruning by support is correct (the anti-monotone property of PrefixSpan still holds). If all
the degrees are 1, the fuzzy support is the usual support of the patterns whose elements have a single item (the
patterns of ``FIM.sequential`` with multi-item elements are not searched here).

A sequence is a list of elements; each element can be a dictionary ``{item: degree}``, a list of ``(item, degree)``
pairs, a single item or a collection of items (all with degree 1). Items with degree 0 are ignored.
``fuzzy_sequences_from_memberships`` builds sequences from a table of membership degrees (one row per instant), e.g.
the one given by the fuzzification of time series.

NOTE: item names must not contain '-'.

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
    raise ValueError("tnorm must be one of {}, received: '{}'".format(TNORMS, tnorm))


def _element_degrees(element: Any) -> Degrees:
    """An element as a dictionary {item: degree} (the highest degree if repeated; degrees of 0 are ignored)."""
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
            raise ValueError("membership degrees must be in [0, 1], received {} for '{}'".format(degree, item))
        if degree > 0:
            result[str(item)] = max(result.get(str(item), 0.0), degree)
    return result


def normalize(sequences: Iterable[Iterable[Any]]) -> List[FuzzySequence]:
    """Each sequence as a list of dictionaries {item: degree}, with no empty elements."""
    result = []
    for sequence in sequences:
        elements = [_element_degrees(element) for element in sequence]
        result.append([element for element in elements if element])
    return result


def degree_tensor(sequences: Sequence[FuzzySequence]) -> Tuple[np.ndarray, List[str]]:
    """
    Tensor of degrees (sequences, positions, items) and the sorted vocabulary of items; the shorter sequences are
    padded with zeros, which contribute nothing to any pattern.
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
    """For each position, the best match of the prefix that ends strictly before."""
    cumulative = np.maximum.accumulate(best_end, axis=1)
    allowed = np.zeros_like(cumulative)
    allowed[:, 1:] = cumulative[:, :-1]
    return allowed


def fuzzy_prefixspan(sequences: Iterable[Iterable[Any]], min_supp: float, max_len: Optional[int] = None,
                     min_len: int = 1, tnorm: str = "min") -> pd.DataFrame:
    """
    Frequent fuzzy sequential patterns (see the module docstring).

    Args:
        sequences: one sequence per element (see the module docstring)
        min_supp (float): minimum fuzzy support, in (0, 1]
        max_len (int, optional): maximum number of items of a pattern
        min_len (int): minimum number of items of a returned pattern
        tnorm (str): 'min' (default) or 'product'

    Returns:
        pandas.DataFrame with the columns 'support' (fuzzy), 'sequence' (tuple of single-item ``frozenset``, as in
        ``FIM.prefixspan``) and 'length', by decreasing support and increasing length. Memory: proportional to
        sequences x maximum length x items
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
    Degree to which the pattern ``<x1 ... xk>`` (list of items) holds in a sequence: the best of its matches (see the
    module docstring). 0 if it does not appear in order.
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
    Fuzzy sequences from a table of membership degrees with one row per instant (ordered in time): each row is an
    element, with the items whose degree exceeds `min_degree`.

    Args:
        memberships (pandas.DataFrame): degrees in [0, 1], one column per item
        by: which sequence each row belongs to (e.g. the day: ``index.date``); by default, all the rows form a single
            sequence. The order of the rows inside each sequence is kept, and the sequences come out in order of
            first appearance of the group
        min_degree (float): minimum degree (exclusive) for an item to count

    Returns:
        list of sequences, each a list of dictionaries {item: degree}; rows with no item above `min_degree` produce
        no element
    """
    check_item_names(memberships.columns)
    degrees = memberships.to_numpy(dtype=float)
    if np.isnan(degrees).any() or (degrees < 0).any() or (degrees > 1).any():
        raise ValueError("membership degrees must be in [0, 1]")
    names = [str(name) for name in memberships.columns]
    groups = np.zeros(len(memberships), dtype=int) if by is None else pd.factorize(np.asarray(by))[0]
    if len(groups) != len(memberships):
        raise ValueError("`by` must have one entry per row ({}), it has {}".format(len(memberships), len(groups)))
    sequences: Dict[int, FuzzySequence] = {}
    for row, group in zip(degrees, groups):
        element = {names[column]: float(row[column]) for column in np.flatnonzero(row > min_degree)}
        sequence = sequences.setdefault(int(group), [])
        if element:
            sequence.append(element)
    return [sequences[group] for group in sorted(sequences)]
