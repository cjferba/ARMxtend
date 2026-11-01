# -*- coding: utf-8 -*-
"""
Aplicacion de reglas ya minadas a una tabla de items booleanos, compartida por
la deteccion de anomalias (``ARM.anomaly``), la prediccion (``ARM.prediction``)
y la clasificacion (``ARM.classification``).

Los items pueden llevar retardo (``item@lagk``, ver ``preprocessing.timeseries``):
``x@lag2`` en la fila ``t`` es el valor de ``x`` en la fila ``t - 2``. Para las
primeras filas, sin datos anteriores, el item con retardo se toma como ausente.

@author Carlos Fernandez-Basso
"""
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from ..preprocessing.timeseries import split_lag


class ItemTable:
    """Una tabla de items booleanos con acceso a los items con retardo y a las conjunciones de items."""

    def __init__(self, items: pd.DataFrame) -> None:
        values = items.to_numpy()
        if values.dtype != bool:
            if not np.isin(values, (0, 1)).all():
                raise ValueError("`items` debe ser una tabla de items booleanos (o 0/1); para datos difusos o "
                                 "numericos use antes preprocessing.timeseries.to_items")
            values = values.astype(bool)
        self.rows = len(items)
        self.values = values
        self.columns: Dict[str, int] = {str(name): position for position, name in enumerate(items.columns)}
        self._cache: Dict[str, np.ndarray] = {}

    def item(self, name: str) -> np.ndarray:
        """Vector booleano (una entrada por fila) con el valor del item `name`, con retardo si lo lleva."""
        if name not in self._cache:
            base, lag = split_lag(name)
            if base not in self.columns:
                raise ValueError("el item '{}' de las reglas no es una columna de `items`".format(base))
            column = self.values[:, self.columns[base]]
            if lag == 0:
                shifted = column
            else:
                shifted = np.zeros(self.rows, dtype=bool)
                if lag < self.rows:
                    shifted[lag:] = column[:self.rows - lag]
            self._cache[name] = shifted
        return self._cache[name]

    def holds(self, names: Iterable[str]) -> np.ndarray:
        """Filas en las que se cumplen todos los items de `names`."""
        result = np.ones(self.rows, dtype=bool)
        for name in names:
            result &= self.item(name)
        return result

    def evaluable(self, names: Iterable[str]) -> np.ndarray:
        """Filas con datos anteriores suficientes para leer todos los items de `names` (retardo <= fila)."""
        reach = max([split_lag(name)[1] for name in names] or [0])
        return np.arange(self.rows) >= reach


def max_lag(itemsets: Iterable[Iterable[str]]) -> int:
    """El mayor retardo de los items de los conjuntos dados (0 si no hay ninguno)."""
    return max([split_lag(item)[1] for itemset in itemsets for item in itemset] or [0])


def require_columns(rules: pd.DataFrame, *names: str) -> None:
    missing = [name for name in names if name not in rules.columns]
    if missing:
        raise ValueError("a `rules` le faltan las columnas {}".format(missing))


def describe_rule(antecedent: Iterable[str], consequent: Iterable[str]) -> str:
    """Texto legible de una regla: 'door (t-1), motion (t-1) => light (t)'."""
    from .temporal_rules import format_lagged_item

    def side(items: Iterable[str]) -> str:
        return ", ".join(sorted(format_lagged_item(item) for item in items))

    return "{} => {}".format(side(antecedent), side(consequent))
