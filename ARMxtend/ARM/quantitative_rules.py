# -*- coding: utf-8 -*-
"""
Reglas de asociacion CUANTITATIVAS por intervalos (Srikant y Agrawal, 1996, "Mining
quantitative association rules in large relational tables"): reglas sobre tablas con
atributos numericos y categoricos, como ``edad en [30, 45) => ingresos en [40000, 90000)``
y ``sexo=F, edad en [20, 30) => compra=si``.

Cada atributo numerico se parte en ``n_intervals`` intervalos BASE de igual frecuencia
(equi-depth) y se forman los RANGOS de intervalos base contiguos cuyo soporte no supera
``max_supp`` (un rango demasiado ancho es verdad casi siempre y no dice nada). Los rangos
forman una jerarquia por inclusion -- ``[30, 40)`` esta dentro de ``[30, 50)`` -- y por eso
todo se reduce a las reglas generalizadas de ``ARM.generalized_rules``, con una taxonomia
cuyos "antecesores" de un intervalo son los rangos que lo contienen, y con la restriccion
de que una regla no mencione dos rangos del mismo atributo. La poda R-interesante de Srikant
y Agrawal elimina entonces las reglas cuyo rango se puede ensanchar o estrechar sin que la
regla cambie de verdad.

* ``IntervalPartition``: ajusta (``fit``) los intervalos con unos datos y transforma
  (``transform``) filas -- tambien nuevas -- en items.
* ``quantitative_rules``: tabla -> reglas.
* ``partial_completeness_intervals``: cuantos intervalos hacen falta para no perder mas de
  cierta "completitud parcial" K (formula del articulo).

Los items categoricos son ``columna=valor``; los rangos, ``columna=[bajo,alto)`` (``]`` en
el ultimo intervalo; el signo menos se escribe con U+2212 porque los items no admiten '-').

@author Carlos Fernandez-Basso
"""
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..FIM.taxonomy import MinSupp, Taxonomy
from .generalized_rules import generalized_rules

_EPS = 1e-9


def partial_completeness_intervals(n_quantitative: int, min_supp: float, completeness: float = 1.5) -> int:
    """
    Intervalos base necesarios por atributo, ``2 n / (min_supp (K - 1))``, para que los rangos formados
    no pierdan mas de una "completitud parcial" K (> 1) respecto a los mejores posibles (Srikant y Agrawal, 1996).

    Argumentos:
        n_quantitative (int): maximo de atributos numericos en una regla
        min_supp (float): soporte minimo, en (0, 1]
        completeness (float): nivel K de completitud parcial, > 1 (cuanto mas cerca de 1, mas intervalos)
    """
    if n_quantitative < 1 or not 0 < min_supp <= 1 or completeness <= 1:
        raise ValueError("hace falta n_quantitative >= 1, min_supp en (0, 1] y completeness > 1")
    return int(math.ceil(2 * n_quantitative / (min_supp * (completeness - 1))))


def _number(value: float) -> str:
    return "{:g}".format(value).replace("-", "−")


class IntervalPartition(object):
    """
    Particion de los atributos de una tabla en items: intervalos base y rangos para los numericos,
    ``columna=valor`` para los demas.

    Argumentos:
        n_intervals (int): intervalos base por atributo numerico (menos si hay valores repetidos)
        method (str): 'quantile' (igual frecuencia, por defecto) o 'uniform' (igual anchura)
        max_supp (float): soporte maximo de un rango de varios intervalos base, en (0, 1]
        columns (list, opcional): columnas a usar (por defecto, todas)
        quantitative (list, opcional): columnas que se tratan como numericas (por defecto, las numericas
            con mas de `n_intervals` valores distintos; las demas son categoricas)

    Atributos tras ``fit``: ``taxonomy_`` (jerarquia de rangos), ``groups_`` (rango -> columna, para
    ``exclusive``), ``intervals_`` (``DataFrame`` con item, column, low, high, support y n_base de cada rango),
    ``edges_`` (bordes de cada columna numerica) y ``categorical_`` (columnas categoricas).
    """

    def __init__(self, n_intervals: int = 8, method: str = "quantile", max_supp: float = 0.5,
                 columns: Optional[Sequence[str]] = None, quantitative: Optional[Sequence[str]] = None):
        if n_intervals < 1:
            raise ValueError("n_intervals debe ser >= 1")
        if method not in ("quantile", "uniform"):
            raise ValueError("method debe ser 'quantile' o 'uniform', recibido: '{}'".format(method))
        if not 0 < max_supp <= 1:
            raise ValueError("max_supp debe estar en (0, 1]")
        self.n_intervals, self.method, self.max_supp = n_intervals, method, max_supp
        self.columns, self.quantitative = columns, quantitative

    def fit(self, table: pd.DataFrame) -> "IntervalPartition":
        columns = list(self.columns) if self.columns is not None else list(table.columns)
        missing = [name for name in columns if name not in table.columns]
        if missing:
            raise ValueError("columnas inexistentes: {}".format(", ".join(map(str, missing))))
        if self.quantitative is not None:
            numeric = list(self.quantitative)
            wrong = [name for name in numeric if name not in columns or not pd.api.types.is_numeric_dtype(table[name])]
            if wrong:
                raise ValueError("las columnas {} no son numericas o no estan en `columns`".format(wrong))
        else:
            numeric = [name for name in columns if pd.api.types.is_numeric_dtype(table[name])
                       and not pd.api.types.is_bool_dtype(table[name]) and table[name].nunique() > self.n_intervals]
        self.categorical_: List[str] = [name for name in columns if name not in numeric]
        if not numeric and not self.categorical_:
            raise ValueError("no hay columnas que usar")
        total = len(table)
        self.edges_: Dict[str, np.ndarray] = {}
        rows: List[Dict[str, Any]] = []
        pairs: List[Any] = []
        for name in numeric:
            values = table[name].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                raise ValueError("la columna '{}' no tiene valores".format(name))
            if self.method == "quantile":
                edges = np.unique(np.quantile(values, np.linspace(0, 1, self.n_intervals + 1)))
            else:
                edges = np.unique(np.linspace(values.min(), values.max(), self.n_intervals + 1))
            if len(edges) == 1:
                edges = np.array([edges[0], edges[0]])
            self.edges_[str(name)] = edges
            count = len(edges) - 1
            base = np.bincount(self._index(values, edges), minlength=count) / total
            cumulative = np.concatenate([[0.0], np.cumsum(base)])
            kept = {}
            for first in range(count):
                for last in range(first, count):
                    support = float(cumulative[last + 1] - cumulative[first])
                    if first == last or support <= self.max_supp + _EPS:
                        kept[(first, last)] = self._label(str(name), edges, first, last)
                        rows.append({"item": kept[(first, last)], "column": str(name), "low": float(edges[first]),
                                     "high": float(edges[last + 1]), "support": support, "n_base": last - first + 1})
            for (first, last), item in kept.items():
                for parent in ((first - 1, last), (first, last + 1)):
                    if parent in kept:
                        pairs.append((item, kept[parent]))
        self.intervals_ = pd.DataFrame(rows, columns=["item", "column", "low", "high", "support", "n_base"])
        self.taxonomy_ = Taxonomy(pairs)
        self.groups_: Dict[str, str] = dict(zip(self.intervals_["item"], self.intervals_["column"]))
        self._numeric = list(numeric)
        return self

    @staticmethod
    def _index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Indice del intervalo base de cada valor; los valores fuera del rango ajustado caen en el primero o el ultimo."""
        return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)

    @staticmethod
    def _label(name: str, edges: np.ndarray, first: int, last: int) -> str:
        closing = "]" if last == len(edges) - 2 else ")"
        return "{}=[{},{}{}".format(name, _number(edges[first]), _number(edges[last + 1]), closing)

    def transform(self, table: pd.DataFrame) -> List[List[str]]:
        """Los items base (intervalo o ``columna=valor``) de cada fila; los valores ausentes no dan item."""
        if not hasattr(self, "taxonomy_"):
            raise ValueError("hay que llamar a fit antes de transform")
        missing = [name for name in self._numeric + self.categorical_ if name not in table.columns]
        if missing:
            raise ValueError("faltan las columnas {}".format(missing))
        baskets: List[List[str]] = [[] for _ in range(len(table))]
        for name in self._numeric:
            edges = self.edges_[str(name)]
            values = table[name].to_numpy(dtype=float)
            present = ~np.isnan(values)
            index = np.full(len(values), -1)
            index[present] = self._index(values[present], edges)
            labels = [self._label(str(name), edges, k, k) for k in range(len(edges) - 1)]
            for row, k in enumerate(index):
                if k >= 0:
                    baskets[row].append(labels[k])
        for name in self.categorical_:
            for row, value in enumerate(table[name]):
                if not pd.isna(value):
                    baskets[row].append("{}={}".format(name, value))
        return baskets


def quantitative_rules(table: pd.DataFrame, min_supp: MinSupp = 0.1, n_intervals: int = 8, max_supp: float = 0.5,
                       columns: Optional[Sequence[str]] = None, quantitative: Optional[Sequence[str]] = None,
                       method: str = "quantile", metric: str = "confidence", min_threshold: float = 0.8,
                       max_len: Optional[int] = None, interest: Optional[float] = None,
                       partition: Optional[IntervalPartition] = None) -> pd.DataFrame:
    """
    Reglas cuantitativas de una tabla (ver el docstring del modulo).

    Argumentos:
        table (pandas.DataFrame): una fila por registro
        min_supp: soporte minimo (o ``{nivel: soporte}``, ver ``ARM.generalized_rules``)
        n_intervals, max_supp, columns, quantitative, method: ver ``IntervalPartition``
        metric, min_threshold, max_len: ver ``ARM.generalized_rules``
        interest (float, opcional): si se da (>= 1), solo las reglas R-interesantes de ese R: las que
            superan en ese factor lo esperado a partir de la regla con rangos mas anchos
        partition (IntervalPartition, opcional): una particion ya ajustada (por ejemplo, con otros datos);
            si se da, se ignoran `n_intervals`, `max_supp`, `columns`, `quantitative` y `method`

    Retorna:
        pandas.DataFrame con el formato de ``ARM.generalized_rules``; ``attrs['partition']`` guarda la
        particion y ``attrs['taxonomy']`` y ``attrs['supports']`` lo necesario para ``interesting_rules``
    """
    if partition is None:
        partition = IntervalPartition(n_intervals, method, max_supp, columns, quantitative).fit(table)
    baskets = partition.transform(table)
    rules = generalized_rules(baskets, partition.taxonomy_, min_supp, metric=metric, min_threshold=min_threshold,
                              max_len=max_len, interest=interest, exclusive=partition.groups_)
    rules.attrs["partition"] = partition
    return rules
