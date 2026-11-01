# -*- coding: utf-8 -*-
"""
Taxonomias de items y minado de itemsets GENERALIZADOS (multinivel).

Una taxonomia ("es-un") ordena los items en jerarquias: ``Jacket -> Outerwear ->
Clothes``. Con ella una regla puede hablar de categorias ("quien compra ropa de
abrigo compra calzado") ademas de productos, y asi aparecen reglas que ningun
producto concreto alcanza por si solo (Srikant y Agrawal, 1995, "Mining generalized
association rules"; Han y Fu, 1995, "Discovery of multiple-level association rules
from large databases").

* ``Taxonomy``: un grafo dirigido sin ciclos hijo -> padre (un item puede tener varios
  padres). ``ancestors``, ``descendants``, ``level`` (la raiz es el nivel 1; el nivel
  de un item es 1 + el mayor nivel de sus padres, de modo que un antecesor siempre
  tiene un nivel estrictamente menor), ``at_level``...
* ``extend_transactions``: anade a cada transaccion los antecesores de sus items.
* ``generalize_transactions``: sustituye cada item por su antecesor de un nivel dado
  (el enfoque de Han y Fu: minar nivel a nivel).
* ``generalized_itemsets``: itemsets frecuentes sobre las transacciones extendidas, sin
  generar nunca un itemset con un item y alguno de sus antecesores (su soporte es el del
  itemset sin el antecesor y solo anade ruido; es la optimizacion de Cumulate). El soporte
  minimo puede ser distinto en cada nivel (``min_supp={1: 0.3, 2: 0.1}``): un itemset debe
  alcanzar el MAYOR de los minimos de los niveles de sus items, lo que mantiene la
  propiedad antimonotona (anadir un item no baja el umbral).

Opcionalmente, ``exclusive`` declara grupos de items que no pueden aparecer juntos en un itemset.

NOTA: los nombres de item no deben contener '-'.

@author Carlos Fernandez-Basso
"""
from itertools import product
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from ..preprocessing.timeseries import check_item_names

_EPS = 1e-9

Parents = Union[Mapping[str, Union[str, Iterable[str]]], Iterable[Tuple[str, str]], pd.DataFrame]
MinSupp = Union[float, Mapping[int, float]]


class Taxonomy(object):
    """
    Jerarquia de items hijo -> padre(s), sin ciclos.

    Argumentos:
        parents: ``{hijo: padre}``, ``{hijo: [padre1, padre2]}``, una lista de pares
            ``(hijo, padre)`` o un ``DataFrame`` cuyas dos primeras columnas son hijo y padre.
            Los items que solo aparecen como padres son raices.
    """

    def __init__(self, parents: Parents):
        direct: Dict[str, Set[str]] = {}
        for child, parent in self._pairs(parents):
            child, parent = str(child), str(parent)
            if child == parent:
                raise ValueError("el item '{}' no puede ser su propio padre".format(child))
            direct.setdefault(child, set()).add(parent)
            direct.setdefault(parent, set())
        check_item_names(list(direct))
        self._parents: Dict[str, FrozenSet[str]] = {item: frozenset(found) for item, found in direct.items()}
        self._children: Dict[str, Set[str]] = {item: set() for item in self._parents}
        for child, found in self._parents.items():
            for parent in found:
                self._children[parent].add(child)
        self._ancestors: Dict[str, FrozenSet[str]] = {}
        self._levels: Dict[str, int] = {}
        for item in self._parents:
            self._visit(item, ())

    @staticmethod
    def _pairs(parents: Parents) -> List[Tuple[Any, Any]]:
        if isinstance(parents, pd.DataFrame):
            if parents.shape[1] < 2:
                raise ValueError("el DataFrame de la taxonomia necesita dos columnas: hijo y padre")
            return list(zip(parents.iloc[:, 0], parents.iloc[:, 1]))
        if isinstance(parents, Mapping):
            pairs = []
            for child, found in parents.items():
                for parent in ([found] if isinstance(found, str) else found):
                    pairs.append((child, parent))
            return pairs
        return [tuple(pair) for pair in parents]            # type: ignore[misc]

    def _visit(self, item: str, path: Tuple[str, ...]) -> None:
        if item in self._ancestors:
            return
        if item in path:
            raise ValueError("la taxonomia tiene un ciclo: {}".format(" -> ".join(path[path.index(item):] + (item,))))
        found: Set[str] = set()
        level = 1
        for parent in self._parents[item]:
            self._visit(parent, path + (item,))
            found.add(parent)
            found |= self._ancestors[parent]
            level = max(level, self._levels[parent] + 1)
        self._ancestors[item] = frozenset(found)
        self._levels[item] = level

    @classmethod
    def from_paths(cls, paths: Iterable[str], sep: str = "/") -> "Taxonomy":
        """Taxonomia a partir de caminos ``'Clothes/Outerwear/Jacket'`` (de la raiz a la hoja)."""
        pairs: List[Tuple[str, str]] = []
        for path in paths:
            names = [name.strip() for name in str(path).split(sep) if name.strip()]
            pairs.extend(zip(names[1:], names[:-1]))
        return cls(pairs)

    # ------------------------------------------------------------------ consultas

    @property
    def items(self) -> List[str]:
        """Todos los items de la taxonomia, ordenados."""
        return sorted(self._parents)

    def __contains__(self, item: object) -> bool:
        return item in self._parents

    def __len__(self) -> int:
        return len(self._parents)

    def parents(self, item: str) -> FrozenSet[str]:
        """Padres directos (vacio si el item es raiz o no esta en la taxonomia)."""
        return self._parents.get(str(item), frozenset())

    def children(self, item: str) -> FrozenSet[str]:
        """Hijos directos."""
        return frozenset(self._children.get(str(item), ()))

    def ancestors(self, item: str) -> FrozenSet[str]:
        """Todos los antecesores (padres, abuelos...), sin el propio item."""
        return self._ancestors.get(str(item), frozenset())

    def descendants(self, item: str) -> FrozenSet[str]:
        """Todos los descendientes (hijos, nietos...), sin el propio item."""
        found: Set[str] = set()
        pending = [str(item)]
        while pending:
            for child in self._children.get(pending.pop(), ()):
                if child not in found:
                    found.add(child)
                    pending.append(child)
        return frozenset(found)

    def is_ancestor(self, ancestor: str, item: str) -> bool:
        """``True`` si `ancestor` es un antecesor (estricto) de `item`."""
        return str(ancestor) in self.ancestors(item)

    def level(self, item: str) -> int:
        """Nivel del item: 1 para las raices y para los items que no estan en la taxonomia."""
        return self._levels.get(str(item), 1)

    @property
    def depth(self) -> int:
        """Mayor nivel de la taxonomia."""
        return max(self._levels.values(), default=1)

    def roots(self) -> List[str]:
        return sorted(item for item, found in self._parents.items() if not found)

    def leaves(self) -> List[str]:
        return sorted(item for item, found in self._children.items() if not found)

    def at_level(self, item: str, level: int) -> FrozenSet[str]:
        """
        El item visto desde `level`: el propio item si su nivel es <= `level`; si no, sus antecesores mas
        profundos de nivel <= `level` (el de ese nivel en cada camino; varios si el item tiene varios padres).
        """
        item = str(item)
        if self.level(item) <= level:
            return frozenset([item])
        eligible = [a for a in self.ancestors(item) if self._levels[a] <= level]
        return frozenset(a for a in eligible if not any(a in self.ancestors(b) for b in eligible))

    def to_frame(self) -> pd.DataFrame:
        """Pares ``child, parent`` (los items raiz aparecen solo como padres)."""
        rows = [(child, parent) for child in sorted(self._parents) for parent in sorted(self._parents[child])]
        return pd.DataFrame(rows, columns=["child", "parent"])

    def __repr__(self) -> str:
        return "Taxonomy({} items, {} levels, roots={})".format(len(self), self.depth, self.roots())


def _as_baskets(transactions: Iterable[Iterable[Any]]) -> List[FrozenSet[str]]:
    return [frozenset(str(item) for item in transaction) for transaction in transactions]


def extend_transactions(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy) -> List[FrozenSet[str]]:
    """Cada transaccion con los antecesores de todos sus items (la base de las reglas generalizadas)."""
    extended = []
    for basket in _as_baskets(transactions):
        full = set(basket)
        for item in basket:
            full |= taxonomy.ancestors(item)
        extended.append(frozenset(full))
    return extended


def generalize_transactions(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy, level: int) -> List[FrozenSet[str]]:
    """Cada item sustituido por su antecesor de nivel `level` (los de nivel menor o igual se quedan)."""
    if level < 1:
        raise ValueError("level debe ser >= 1")
    return [frozenset(general for item in basket for general in taxonomy.at_level(item, level))
            for basket in _as_baskets(transactions)]


def _level_thresholds(min_supp: MinSupp, levels: Sequence[int]) -> Dict[int, float]:
    """Soporte minimo de cada nivel: el del mayor nivel listado que no excede al suyo (o el del primero listado)."""
    if isinstance(min_supp, Mapping):
        if not min_supp:
            raise ValueError("min_supp no puede ser un diccionario vacio")
        listed = {int(level): float(value) for level, value in min_supp.items()}
    else:
        listed = {1: float(min_supp)}
    for value in listed.values():
        if not 0 < value <= 1:
            raise ValueError("min_supp debe estar en (0, 1]")
    ordered = sorted(listed)
    return {level: listed[max([k for k in ordered if k <= level] or [ordered[0]])] for level in levels}


def generalized_itemsets(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy, min_supp: MinSupp,
                         max_len: Optional[int] = None, exclusive: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """
    Itemsets frecuentes generalizados (ver el docstring del modulo).

    Argumentos:
        transactions: una transaccion por elemento, cada una una coleccion de items (hojas o categorias)
        taxonomy (Taxonomy): jerarquia de items; los items que no estan en ella son raices
        min_supp: soporte minimo relativo en (0, 1], o ``{nivel: soporte}`` (un soporte menor en los
            niveles profundos es lo habitual: "soporte reducido" de Han y Fu). Un nivel no listado usa
            el del mayor nivel listado por debajo; uno anterior al primero listado, el del primero
        max_len (int, opcional): maximo de items de un itemset
        exclusive (dict, opcional): ``{item: grupo}``; dos items del mismo grupo no aparecen juntos en un
            itemset (p. ej. dos rangos del mismo atributo, ver ``ARM.quantitative_rules``)

    Retorna:
        pandas.DataFrame con las columnas 'support' e 'itemsets' (``frozenset``), por soporte decreciente
        (el formato de ``ARM.association_rules``). ``result.attrs['n_transactions']`` guarda el numero de
        transacciones. Memoria: proporcional a transacciones x itemsets frecuentes del nivel mas ancho
    """
    baskets = _as_baskets(transactions)
    total = len(baskets)
    if max_len is not None and max_len < 1:
        raise ValueError("max_len debe ser >= 1")
    empty = pd.DataFrame({"support": [], "itemsets": []})
    empty.attrs["n_transactions"] = total
    _level_thresholds(min_supp, [1])                                          # valida min_supp
    if total == 0:
        return empty

    vocabulary = sorted({item for basket in baskets for item in basket}
                        | {ancestor for basket in baskets for item in basket for ancestor in taxonomy.ancestors(item)})
    column = {item: position for position, item in enumerate(vocabulary)}
    thresholds = _level_thresholds(min_supp, sorted({taxonomy.level(item) for item in vocabulary}))
    threshold = np.array([thresholds[taxonomy.level(item)] for item in vocabulary])

    present = np.zeros((total, len(vocabulary)), dtype=bool)
    for row, basket in enumerate(baskets):
        for item in basket:
            present[row, column[item]] = True
            for ancestor in taxonomy.ancestors(item):
                present[row, column[ancestor]] = True

    related = np.zeros((len(vocabulary), len(vocabulary)), dtype=bool)         # item y antecesor suyo
    for item in vocabulary:
        for ancestor in taxonomy.ancestors(item):
            if ancestor in column:
                related[column[item], column[ancestor]] = related[column[ancestor], column[item]] = True
    members: Dict[Any, List[int]] = {}
    for item, group in (exclusive or {}).items():
        if item in column:
            members.setdefault(group, []).append(column[item])
    for columns in members.values():
        related[np.ix_(columns, columns)] = True
        related[columns, columns] = False

    counts = present.sum(axis=0)
    needed = np.ceil(threshold * total - _EPS)
    found: List[Tuple[Tuple[int, ...], float]] = []
    level: Dict[Tuple[int, ...], np.ndarray] = {}
    for position in np.flatnonzero(counts >= needed):
        level[(int(position),)] = present[:, position]
        found.append(((int(position),), float(counts[position]) / total))

    size = 1
    while level and (max_len is None or size < max_len):
        ordered = sorted(level)
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for itemset in ordered:
            groups.setdefault(itemset[:-1], []).append(itemset[-1])
        known = set(level)
        following: Dict[Tuple[int, ...], np.ndarray] = {}
        for prefix, tails in groups.items():
            for first in range(len(tails)):
                for second in range(first + 1, len(tails)):
                    a, b = tails[first], tails[second]
                    if related[a, b]:
                        continue
                    candidate = prefix + (a, b)
                    if any(candidate[:k] + candidate[k + 1:] not in known for k in range(len(prefix))):
                        continue
                    vector = level[prefix + (a,)] & present[:, b]
                    count = int(vector.sum())
                    if count >= np.ceil(max(threshold[list(candidate)]) * total - _EPS):
                        following[candidate] = vector
                        found.append((candidate, count / total))
        level = following
        size += 1

    result = pd.DataFrame({"support": [support for _, support in found],
                           "itemsets": [frozenset(vocabulary[i] for i in itemset) for itemset, _ in found]})
    result = result.sort_values("support", ascending=False, kind="mergesort").reset_index(drop=True)
    result.attrs["n_transactions"] = total
    return result


def ancestor_replacements(items: Sequence[str], taxonomy: Taxonomy, only: Optional[Set[str]] = None) -> List[Tuple[str, ...]]:
    """
    Todas las tuplas que resultan de sustituir cada item por si mismo o por un antecesor, salvo la identidad. Con
    `only`, solo se consideran los antecesores de ese conjunto (los que aparecen en alguna regla).
    """
    options = [[item] + sorted(a for a in taxonomy.ancestors(item) if only is None or a in only) for item in items]
    return [combination for combination in product(*options) if combination != tuple(items)]
