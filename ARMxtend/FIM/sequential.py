# -*- coding: utf-8 -*-
"""
Frequent sequential patterns: subsequences of events that repeat, in that order, in many sequences (daily routines
of a sensorized house, patient trajectories, usage sessions, ...).

A SEQUENCE is a time-ordered list of ELEMENTS, and each element is a set of items that happen at the same time:
``[{"kitchen"}, {"door", "light"}, {"living_room"}]``. An element can also be written as a single item
(``"kitchen"``). A pattern ``<{a, b} {c}>`` is contained in a sequence if there are two elements, one before the
other, that contain ``{a, b}`` and ``{c}`` respectively (there may be other elements between them). The support of a
pattern is the fraction of sequences that contain it, counting each sequence only once.

`prefixspan` implements PrefixSpan (Pei, Han, Mortazavi-Asl, Wang, Pinto, Chen, Dayal & Hsu, 2004, "Mining
sequential patterns by pattern-growth: the PrefixSpan approach", IEEE Transactions on Knowledge and Data
Engineering, 16(11)) with projection by the position of the first match of each prefix. `prefixspan_spark` runs the
same computation on Spark with the MLlib PrefixSpan (`pyspark.ml.fpm`), for sequences that do not fit on one
machine.

It also includes utilities to build sequences from an event log (`sequences_from_events`) or from a table of items
per instant (`sequences_from_items`), and `format_sequence` to display them.

@author Carlos Fernandez-Basso
"""
from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

Element = FrozenSet[str]
SequenceType = Tuple[Element, ...]


def _normalize(sequences: Iterable[Iterable[Union[str, Iterable[str]]]]) -> List[List[Element]]:
    """Convert each sequence into a list of elements (frozenset of items), with no empty elements."""
    result = []
    for sequence in sequences:
        elements = []
        for element in sequence:
            items = frozenset([element]) if isinstance(element, str) else frozenset(element)
            if items:
                elements.append(items)
        result.append(elements)
    return result


def format_sequence(sequence: Sequence[Iterable[str]]) -> str:
    """Readable text of a pattern: ``({'a','b'}, {'c'})`` -> ``'<{a, b} {c}>'``."""
    return "<" + " ".join("{" + ", ".join(sorted(element)) + "}" for element in sequence) + ">"


def contains_sequence(sequence: Iterable[Iterable[str]], pattern: Iterable[Iterable[str]]) -> bool:
    """True if `pattern` is contained in `sequence` (in order, each element of the pattern inside a different one)."""
    elements = [frozenset(e) for e in sequence]
    position = 0
    for wanted in pattern:
        wanted = frozenset(wanted)
        while position < len(elements) and not wanted <= elements[position]:
            position += 1
        if position == len(elements):
            return False
        position += 1
    return True


def prefixspan(sequences: Iterable[Iterable[Union[str, Iterable[str]]]], min_supp: float,
               max_len: Optional[int] = None, min_len: int = 1) -> pd.DataFrame:
    """
    Frequent sequential patterns (PrefixSpan).

    Args:
        sequences: one sequence per element; each sequence, a list of elements (sets of items, or a single item)
        min_supp (float): minimum relative support (fraction of sequences), in (0, 1]
        max_len (int, optional): maximum number of items of a pattern (sum of the sizes of its elements); no limit
            by default
        min_len (int): minimum number of items of the returned patterns

    Returns:
        pandas.DataFrame with one row per pattern and the columns 'support', 'sequence' (tuple of ``frozenset``, one
        element each) and 'length' (number of items), sorted by decreasing support and length.
    """
    database = _normalize(sequences)
    total = len(database)
    if total == 0:
        return pd.DataFrame({"support": [], "sequence": [], "length": []})
    minCount = min_supp * total - 1e-9
    sortedElements = [[tuple(sorted(element)) for element in sequence] for sequence in database]

    found: List[Tuple[int, SequenceType]] = []

    def grow(prefix: List[Tuple[str, ...]], length: int, projected: List[Tuple[int, int, int]]) -> None:
        if max_len is not None and length >= max_len:
            return
        sExtensions: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        iExtensions: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        lastElement = set(prefix[-1]) if prefix else None
        lastItem = prefix[-1][-1] if prefix else ""

        for sequenceIndex, previous, last in projected:
            elements = database[sequenceIndex]
            firstAfter: Dict[str, int] = {}
            for position in range(last + 1, len(elements)):
                for item in sortedElements[sequenceIndex][position]:
                    firstAfter.setdefault(item, position)
            for item, position in firstAfter.items():
                sExtensions[item].append((sequenceIndex, last, position))
            if lastElement is not None:
                firstWith: Dict[str, int] = {}
                for position in range(previous + 1, len(elements)):
                    if lastElement <= elements[position]:
                        for item in sortedElements[sequenceIndex][position]:
                            if item > lastItem:
                                firstWith.setdefault(item, position)
                for item, position in firstWith.items():
                    iExtensions[item].append((sequenceIndex, previous, position))

        for item in sorted(iExtensions):
            if len(iExtensions[item]) >= minCount:
                extended = prefix[:-1] + [prefix[-1] + (item,)]
                found.append((len(iExtensions[item]), tuple(frozenset(e) for e in extended)))
                grow(extended, length + 1, iExtensions[item])
        for item in sorted(sExtensions):
            if len(sExtensions[item]) >= minCount:
                extended = prefix + [(item,)]
                found.append((len(sExtensions[item]), tuple(frozenset(e) for e in extended)))
                grow(extended, length + 1, sExtensions[item])

    grow([], 0, [(index, -1, -1) for index in range(total)])

    rows = [(count / total, pattern, sum(len(element) for element in pattern)) for count, pattern in found]
    result = pd.DataFrame(rows, columns=["support", "sequence", "length"])
    result = result[result["length"] >= min_len]
    return result.sort_values(["support", "length"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _sorted_elements(sequence: Iterable[Union[str, Iterable[str]]]) -> List[List[str]]:
    """A sequence as a list of elements (sorted lists of items without repeats), with no empty elements."""
    result = []
    for element in sequence:
        items = sorted({element} if isinstance(element, str) else set(element))
        if items:
            result.append(items)
    return result


def prefixspan_spark(spark: Any, sequences: Any, min_supp: float, max_pattern_length: int = 10,
                     sequence_column: str = "sequence") -> pd.DataFrame:
    """
    Frequent sequential patterns on Spark: it uses the MLlib PrefixSpan (``pyspark.ml.fpm.PrefixSpan``). Requires
    the ``spark`` extra.

    Args:
        spark: a ``SparkSession`` or a ``SparkContext``
        sequences: the sequences, in the same format as `prefixspan`, as a Python list, as an RDD or as a Spark
            ``DataFrame`` with a `sequence_column` column of type ``array<array<string>>`` (for example read from
            Parquet). The RDD and the DataFrame are NOT collected on the driver: they are normalized on the executors
        min_supp (float): minimum relative support, in (0, 1]
        max_pattern_length (int): maximum number of items of a pattern
        sequence_column (str): column with the sequences when `sequences` is a DataFrame

    Returns:
        pandas.DataFrame with the columns 'support', 'sequence' and 'length', like `prefixspan`.
    """
    from pyspark.ml.fpm import PrefixSpan
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F

    session = spark if isinstance(spark, SparkSession) else SparkSession.builder.getOrCreate()
    if isinstance(sequences, DataFrame):
        column = F.col(sequence_column)
        frame = sequences.select(F.filter(F.transform(column, lambda element: F.array_sort(F.array_distinct(element))),
                                          lambda element: F.size(element) > 0).alias("sequence"))
    elif hasattr(sequences, "map") and hasattr(sequences, "collect"):
        frame = session.createDataFrame(sequences.map(lambda sequence: (_sorted_elements(sequence),)),
                                        "sequence array<array<string>>")
    else:
        rows = [(_sorted_elements(sequence),) for sequence in sequences]
        frame = session.createDataFrame(rows, "sequence array<array<string>>")
    total = frame.count()
    if total == 0:
        return pd.DataFrame({"support": [], "sequence": [], "length": []})

    patterns = PrefixSpan(minSupport=min_supp, maxPatternLength=max_pattern_length,
                          sequenceCol="sequence").findFrequentSequentialPatterns(frame).collect()

    table = [(row["freq"] / total, tuple(frozenset(element) for element in row["sequence"]),
              sum(len(element) for element in row["sequence"])) for row in patterns]
    result = pd.DataFrame(table, columns=["support", "sequence", "length"])
    return result.sort_values(["support", "length"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def sequences_from_events(events: pd.DataFrame, sequence_column: str, time_column: str, item_column: str,
                          bin: Optional[Union[str, pd.Timedelta]] = None) -> List[List[Element]]:
    """
    Build sequences from an event log (one row per event).

    Args:
        events (pandas.DataFrame): log with one row per event
        sequence_column (str): column that identifies the sequence (user, patient, session, day, ...)
        time_column (str): instant of the event (sortable)
        item_column (str): item of the event
        bin (str | pandas.Timedelta, optional): if given, the events of the same sequence within the same interval
            (``'5min'``, ``'1h'``) form a single element; otherwise, the events with exactly the same instant form
            an element. Requires `time_column` to be of date type

    Returns:
        list[list[frozenset[str]]]: one sequence per value of `sequence_column` (in the order of its values), with
        the elements ordered in time.
    """
    for column in (sequence_column, time_column, item_column):
        if column not in events.columns:
            raise ValueError("column '{}' does not exist in the events".format(column))
    table = events[[sequence_column, time_column, item_column]].dropna()
    moment = pd.to_datetime(table[time_column]).dt.floor(bin) if bin is not None else table[time_column]
    table = table.assign(_moment=moment.to_numpy())
    result = []
    for _, group in table.groupby(sequence_column, sort=True):
        elements = [frozenset(items.astype(str)) for _, items in group.sort_values("_moment").groupby("_moment",
                                                                                                    sort=True)[item_column]]
        result.append(elements)
    return result


def sequences_from_items(items: pd.DataFrame, by: Any = None) -> List[List[Element]]:
    """
    Build sequences from a table of boolean items (one row per instant): each row with some item is an element, and
    empty rows are omitted.

    Args:
        items (pandas.DataFrame): boolean items ordered in time
        by: how to split the table into sequences: a sequence or ``Series`` of the same length, or a function that
            receives the index and returns it (e.g. ``lambda idx: idx.date`` for one sequence per day); by default,
            the whole table is a single sequence

    Returns:
        list[list[frozenset[str]]]: one sequence per group, in the order of the keys
    """
    names = np.array(items.columns, dtype=object)
    present = items.to_numpy(dtype=bool)
    if by is None:
        groups = np.zeros(len(items), dtype=int)
    else:
        groups = np.asarray(by(items.index) if callable(by) else by)
        if len(groups) != len(items):
            raise ValueError("`by` must have one entry per row ({}), it has {}".format(len(items), len(groups)))
    result = []
    for key in sorted(set(groups.tolist())):
        rows = present[groups == key]
        result.append([frozenset(names[row]) for row in rows if row.any()])
    return result
