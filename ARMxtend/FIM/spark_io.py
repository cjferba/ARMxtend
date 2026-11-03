# -*- coding: utf-8 -*-
"""
Input from Spark DataFrames (for example Parquet files) for the Big Data algorithms (``DApriori``, ``DAprioriTID``,
``DECLAT``, ``FuzzyDAprioriTID``, ``FuzzyDECLAT``), which work with RDDs.

* ``transactions_from_dataframe``: ``array<string>`` column (or text with a separator) -> ``RDD[str]`` ('a,b,c'),
  the input of the crisp algorithms.
* ``fuzzy_transactions_from_dataframe``: ``array<struct<item,degree>>`` column or two parallel columns (items and
  degrees) -> ``RDD[list[(item, degree)]]``, the input of the fuzzy algorithms.
* ``read_transactions``: reads a file (Parquet by default) and calls the previous one.

Nothing is collected on the driver: the conversion is done on the executors. Item names cannot contain ',' or '-'.

@author Carlos Fernandez-Basso
"""
from typing import Any, Optional


def _check(name: str) -> None:
    if not name:
        raise ValueError("a column name is missing")


def transactions_from_dataframe(frame: Any, column: str = "items", sep: Optional[str] = None) -> Any:
    """
    RDD of 'a,b,c' transactions from a column of a DataFrame.

    Args:
        frame: Spark ``DataFrame``
        column (str): ``array<...>`` column (the elements are converted to text) or, if `sep` is given, text with the
            items separated by `sep`
        sep (str, optional): separator of the text column
    """
    _check(column)
    if column not in frame.columns:
        raise ValueError("column '{}' does not exist; available: {}".format(column, ", ".join(frame.columns)))
    dtype = dict(frame.dtypes)[column]
    if sep is None and not dtype.startswith("array"):
        raise ValueError("column '{}' is {}; use `sep` for a text column".format(column, dtype))
    if sep is not None and dtype.startswith("array"):
        raise ValueError("`sep` only applies to text columns, not to {}".format(dtype))

    def join(row):
        value = row[0]
        if value is None:
            return ""
        items = value if sep is None else str(value).split(sep)
        return ",".join(sorted({str(item).strip() for item in items if item is not None and str(item).strip()}))

    return frame.select(column).rdd.map(join)


def fuzzy_transactions_from_dataframe(frame: Any, items_column: str = "items", degrees_column: Optional[str] = None) -> Any:
    """
    RDD of fuzzy transactions ``[(item, degree), ...]`` from a DataFrame.

    Args:
        frame: Spark ``DataFrame``
        items_column (str): ``array<struct<item, degree>>`` (the first two fields of the struct) if `degrees_column`
            is not given; if it is, an ``array<string>`` of items
        degrees_column (str, optional): ``array<double>`` with the degrees, of the same size as the items
    """
    for name in (items_column, degrees_column):
        if name is not None and name not in frame.columns:
            raise ValueError("column '{}' does not exist; available: {}".format(name, ", ".join(frame.columns)))
    if degrees_column is None:
        def convert(row):
            return [(str(pair[0]), float(pair[1])) for pair in (row[0] or [])]
        return frame.select(items_column).rdd.map(convert)

    def zip_columns(row):
        items, degrees = row[0] or [], row[1] or []
        if len(items) != len(degrees):
            raise ValueError("items and degrees must have the same size in every row")
        return [(str(item), float(degree)) for item, degree in zip(items, degrees)]
    return frame.select(items_column, degrees_column).rdd.map(zip_columns)


def read_transactions(spark: Any, path: str, format: str = "parquet", column: str = "items", sep: Optional[str] = None,
                      degrees_column: Optional[str] = None, fuzzy: bool = False) -> Any:
    """
    Read a file (``parquet``, ``json``, ``csv``, ...) with ``spark.read`` and return the RDD of transactions (crisp,
    or fuzzy with ``fuzzy=True``; see the previous functions).
    """
    reader = spark.read.format(format)
    if format == "csv":
        reader = reader.option("header", True)
    frame = reader.load(path)
    if fuzzy:
        return fuzzy_transactions_from_dataframe(frame, column, degrees_column)
    return transactions_from_dataframe(frame, column, sep)
