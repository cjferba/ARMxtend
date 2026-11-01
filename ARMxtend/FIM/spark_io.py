# -*- coding: utf-8 -*-
"""
Entrada desde DataFrames de Spark (por ejemplo, ficheros Parquet) para los algoritmos Big Data
(``DApriori``, ``DAprioriTID``, ``DECLAT``, ``FuzzyDAprioriTID``, ``FuzzyDECLAT``), que trabajan con RDDs.

* ``transactions_from_dataframe``: columna ``array<string>`` (o texto con separador) -> ``RDD[str]``
  ('a,b,c'), la entrada de los algoritmos crisp.
* ``fuzzy_transactions_from_dataframe``: columna ``array<struct<item,degree>>`` o dos columnas paralelas
  (items y grados) -> ``RDD[list[(item, grado)]]``, la entrada de los algoritmos difusos.
* ``read_transactions``: lee un fichero (Parquet por defecto) y llama a la anterior.

No se recoge nada en el driver: la conversion se hace en los ejecutores. Los nombres de item no pueden
contener ',' ni '-'.

@author Carlos Fernandez-Basso
"""
from typing import Any, Optional


def _check(name: str) -> None:
    if not name:
        raise ValueError("falta el nombre de una columna")


def transactions_from_dataframe(frame: Any, column: str = "items", sep: Optional[str] = None) -> Any:
    """
    RDD de transacciones 'a,b,c' desde una columna de un DataFrame.

    Argumentos:
        frame: ``DataFrame`` de Spark
        column (str): columna ``array<...>`` (los elementos se convierten a texto) o, si se da `sep`, texto
            con los items separados por `sep`
        sep (str, opcional): separador de la columna de texto
    """
    _check(column)
    if column not in frame.columns:
        raise ValueError("la columna '{}' no existe; hay: {}".format(column, ", ".join(frame.columns)))
    dtype = dict(frame.dtypes)[column]
    if sep is None and not dtype.startswith("array"):
        raise ValueError("la columna '{}' es {}; use `sep` para una columna de texto".format(column, dtype))
    if sep is not None and dtype.startswith("array"):
        raise ValueError("`sep` solo vale para columnas de texto, no para {}".format(dtype))

    def join(row):
        value = row[0]
        if value is None:
            return ""
        items = value if sep is None else str(value).split(sep)
        return ",".join(sorted({str(item).strip() for item in items if item is not None and str(item).strip()}))

    return frame.select(column).rdd.map(join)


def fuzzy_transactions_from_dataframe(frame: Any, items_column: str = "items", degrees_column: Optional[str] = None) -> Any:
    """
    RDD de transacciones difusas ``[(item, grado), ...]`` desde un DataFrame.

    Argumentos:
        frame: ``DataFrame`` de Spark
        items_column (str): ``array<struct<item, degree>>`` (los dos primeros campos del struct) si no se da
            `degrees_column`; si se da, un ``array<string>`` de items
        degrees_column (str, opcional): ``array<double>`` con los grados, del mismo tamano que los items
    """
    for name in (items_column, degrees_column):
        if name is not None and name not in frame.columns:
            raise ValueError("la columna '{}' no existe; hay: {}".format(name, ", ".join(frame.columns)))
    if degrees_column is None:
        def convert(row):
            return [(str(pair[0]), float(pair[1])) for pair in (row[0] or [])]
        return frame.select(items_column).rdd.map(convert)

    def zip_columns(row):
        items, degrees = row[0] or [], row[1] or []
        if len(items) != len(degrees):
            raise ValueError("items y grados deben tener el mismo tamano en cada fila")
        return [(str(item), float(degree)) for item, degree in zip(items, degrees)]
    return frame.select(items_column, degrees_column).rdd.map(zip_columns)


def read_transactions(spark: Any, path: str, format: str = "parquet", column: str = "items", sep: Optional[str] = None,
                      degrees_column: Optional[str] = None, fuzzy: bool = False) -> Any:
    """
    Lee un fichero (``parquet``, ``json``, ``csv``, ...) con ``spark.read`` y devuelve el RDD de transacciones
    (crisp, o difusas con ``fuzzy=True``; ver las funciones anteriores).
    """
    reader = spark.read.format(format)
    if format == "csv":
        reader = reader.option("header", True)
    frame = reader.load(path)
    if fuzzy:
        return fuzzy_transactions_from_dataframe(frame, column, degrees_column)
    return transactions_from_dataframe(frame, column, sep)
