# -*- coding: utf-8 -*-
"""
Patrones secuenciales frecuentes: subsecuencias de eventos que se repiten, en
ese orden, en muchas secuencias (rutinas diarias de una casa sensorizada,
trayectorias de pacientes, sesiones de uso, ...).

Una SECUENCIA es una lista ordenada en el tiempo de ELEMENTOS, y cada elemento
es un conjunto de items que ocurren a la vez: ``[{"cocina"}, {"puerta",
"luz"}, {"salon"}]``. Un elemento puede escribirse tambien como un solo item
(``"cocina"``). Un patron ``<{a, b} {c}>`` esta contenido en una secuencia si
existen dos elementos, uno antes que otro, que contienen ``{a, b}`` y ``{c}``
respectivamente (entre ellos puede haber otros elementos). El soporte de un
patron es la fraccion de secuencias que lo contienen, contando cada secuencia
una sola vez.

`prefixspan` implementa PrefixSpan (Pei, Han, Mortazavi-Asl, Wang, Pinto,
Chen, Dayal & Hsu, 2004, "Mining sequential patterns by pattern-growth: the
PrefixSpan approach", IEEE Transactions on Knowledge and Data Engineering,
16(11)) con la proyeccion por posicion del primer encaje de cada prefijo.
`prefixspan_spark` ejecuta el mismo calculo en Spark con el PrefixSpan de MLlib
(`pyspark.ml.fpm`), para secuencias que no caben en una maquina.

Tambien incluye utilidades para construir secuencias a partir de un registro
de eventos (`sequences_from_events`) o de una tabla de items por instante
(`sequences_from_items`), y `format_sequence` para mostrarlas.

@author Carlos Fernandez-Basso
"""
from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

Element = FrozenSet[str]
SequenceType = Tuple[Element, ...]


def _normalize(sequences: Iterable[Iterable[Union[str, Iterable[str]]]]) -> List[List[Element]]:
    """Convierte cada secuencia en una lista de elementos (frozenset de items), sin elementos vacios."""
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
    """Texto legible de un patron: ``({'a','b'}, {'c'})`` -> ``'<{a, b} {c}>'``."""
    return "<" + " ".join("{" + ", ".join(sorted(element)) + "}" for element in sequence) + ">"


def contains_sequence(sequence: Iterable[Iterable[str]], pattern: Iterable[Iterable[str]]) -> bool:
    """True si `pattern` esta contenido en `sequence` (en orden, cada elemento del patron dentro de uno distinto)."""
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
    Patrones secuenciales frecuentes (PrefixSpan).

    Argumentos:
        sequences: una secuencia por elemento; cada secuencia, una lista de
            elementos (conjuntos de items, o un item suelto)
        min_supp (float): soporte minimo relativo (fraccion de secuencias), en (0, 1]
        max_len (int, opcional): numero maximo de items de un patron (suma de
            los tamanos de sus elementos); por defecto, sin limite
        min_len (int): numero minimo de items de los patrones devueltos

    Retorna:
        pandas.DataFrame con una fila por patron y las columnas 'support',
        'sequence' (tupla de ``frozenset``, un elemento cada uno) y 'length'
        (numero de items), ordenado por soporte decreciente y longitud.
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


def prefixspan_spark(spark: Any, sequences: Any, min_supp: float, max_pattern_length: int = 10) -> pd.DataFrame:
    """
    Patrones secuenciales frecuentes en Spark: usa el PrefixSpan de MLlib
    (``pyspark.ml.fpm.PrefixSpan``). Requiere el extra ``spark``.

    Argumentos:
        spark: un ``SparkSession`` o un ``SparkContext``
        sequences: las secuencias, en el mismo formato que `prefixspan`, como
            lista de Python o como RDD
        min_supp (float): soporte minimo relativo, en (0, 1]
        max_pattern_length (int): numero maximo de items de un patron

    Retorna:
        pandas.DataFrame con las columnas 'support', 'sequence' y 'length',
        como `prefixspan`.
    """
    from pyspark.ml.fpm import PrefixSpan
    from pyspark.sql import SparkSession

    session = spark if isinstance(spark, SparkSession) else SparkSession.builder.getOrCreate()
    local = sequences.collect() if hasattr(sequences, "collect") else list(sequences)
    database = _normalize(local)
    if not database:
        return pd.DataFrame({"support": [], "sequence": [], "length": []})

    rows = [([sorted(element) for element in sequence],) for sequence in database]
    frame = session.createDataFrame(rows, ["sequence"])
    patterns = PrefixSpan(minSupport=min_supp, maxPatternLength=max_pattern_length,
                          sequenceCol="sequence").findFrequentSequentialPatterns(frame).collect()

    total = len(database)
    table = [(row["freq"] / total, tuple(frozenset(element) for element in row["sequence"]),
              sum(len(element) for element in row["sequence"])) for row in patterns]
    result = pd.DataFrame(table, columns=["support", "sequence", "length"])
    return result.sort_values(["support", "length"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def sequences_from_events(events: pd.DataFrame, sequence_column: str, time_column: str, item_column: str,
                          bin: Optional[Union[str, pd.Timedelta]] = None) -> List[List[Element]]:
    """
    Construye secuencias a partir de un registro de eventos (una fila por evento).

    Argumentos:
        events (pandas.DataFrame): registro con una fila por evento
        sequence_column (str): columna que identifica la secuencia (usuario,
            paciente, sesion, dia, ...)
        time_column (str): instante del evento (ordenable)
        item_column (str): item del evento
        bin (str | pandas.Timedelta, opcional): si se da, los eventos de una
            misma secuencia dentro del mismo intervalo (``'5min'``, ``'1h'``)
            forman un unico elemento; si no, forman un elemento los eventos
            con exactamente el mismo instante. Requiere que `time_column`
            sea de tipo fecha

    Retorna:
        list[list[frozenset[str]]]: una secuencia por valor de `sequence_column`
        (en el orden de sus valores), con los elementos ordenados en el tiempo.
    """
    for column in (sequence_column, time_column, item_column):
        if column not in events.columns:
            raise ValueError("la columna '{}' no existe en los eventos".format(column))
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
    Construye secuencias a partir de una tabla de items booleanos (una fila por
    instante): cada fila con algun item es un elemento, y las filas vacias se
    omiten.

    Argumentos:
        items (pandas.DataFrame): items booleanos ordenados en el tiempo
        by: como partir la tabla en secuencias: una secuencia o ``Series`` de
            su misma longitud, o una funcion que recibe el indice y la
            devuelve (p.ej. ``lambda idx: idx.date`` para una secuencia por
            dia); por defecto, toda la tabla es una sola secuencia

    Retorna:
        list[list[frozenset[str]]]: una secuencia por grupo, en el orden de las claves
    """
    names = np.array(items.columns, dtype=object)
    present = items.to_numpy(dtype=bool)
    if by is None:
        groups = np.zeros(len(items), dtype=int)
    else:
        groups = np.asarray(by(items.index) if callable(by) else by)
        if len(groups) != len(items):
            raise ValueError("`by` debe tener una entrada por fila ({}), tiene {}".format(len(items), len(groups)))
    result = []
    for key in sorted(set(groups.tolist())):
        rows = present[groups == key]
        result.append([frozenset(names[row]) for row in rows if row.any()])
    return result
