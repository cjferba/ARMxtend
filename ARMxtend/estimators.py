# -*- coding: utf-8 -*-
"""
Estimadores con la interfaz de scikit-learn (``fit`` / ``transform`` / ``get_params``), sin depender de
scikit-learn:

* ``TransactionEncoder``: transacciones (listas de items) <-> tabla booleana, una columna por item. Es el formato
  de entrada de casi todo ARMxtend (``ARM.association_rules`` lee itemsets, ``anomaly``/``prediction``/
  ``classification`` leen tablas booleanas) y el de ``mlxtend.preprocessing.TransactionEncoder``.
* ``RuleMiner``: extrae las reglas de una coleccion de transacciones, con el algoritmo elegido (`method`):
  ``"fpgrowth"`` (umbral de soporte `min_supp`), ``"top_k"`` (las `k` de mayor soporte) o ``"rare"`` (combinaciones
  raras de items frecuentes, `max_supp`). Deja ``itemsets_`` y ``rules_`` tras ``fit``; ``transform`` convierte
  transacciones en una tabla de reglas que se cumplen, util como atributos de otro modelo en un ``Pipeline``.

Los estimadores guardan sus parametros sin tocarlos (``ParamsMixin``): ``sklearn.base.clone`` y ``GridSearchCV``
funcionan con ellos, y ``ARM.RuleClassifier`` (CBA) sigue el mismo protocolo.

@author Carlos Fernandez-Basso
"""
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .ARM._apply import ItemTable
from .base import ParamsMixin

Transactions = Union[Iterable[Iterable[Any]], pd.DataFrame]


def as_transactions(data: Transactions) -> List[List[Any]]:
    """Transacciones como lista de listas: una tabla booleana (o 0/1) da, por fila, las columnas que valen verdadero."""
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy()
        if values.dtype != bool:
            if not np.isin(values, (0, 1)).all():
                raise ValueError("una tabla de entrada debe ser booleana (o 0/1): una columna por item")
            values = values.astype(bool)
        names = list(data.columns)
        return [[names[position] for position in np.flatnonzero(row)] for row in values]
    return [list(transaction) for transaction in data]


class TransactionEncoder(ParamsMixin):
    """
    Transacciones -> tabla booleana (una fila por transaccion, una columna por item, en orden alfabetico).

    Argumentos:
        handle_unknown: que hacer en ``transform`` con items que no se vieron en ``fit``: ``"ignore"`` (se descartan,
            por defecto) o ``"error"``

    Atributos tras ``fit``: ``columns_`` (lista ordenada de items).
    """

    def __init__(self, handle_unknown: str = "ignore") -> None:
        self.handle_unknown = handle_unknown

    def fit(self, X: Transactions, y: Any = None) -> "TransactionEncoder":
        if self.handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown debe ser 'ignore' o 'error'")
        items = {item for transaction in as_transactions(X) for item in transaction}
        try:
            self.columns_ = sorted(items)
        except TypeError:
            self.columns_ = sorted(items, key=str)
        return self

    def transform(self, X: Transactions) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise ValueError("el codificador no esta ajustado: llame antes a fit")
        position = {item: index for index, item in enumerate(self.columns_)}
        transactions = as_transactions(X)
        table = np.zeros((len(transactions), len(self.columns_)), dtype=bool)
        for row, transaction in enumerate(transactions):
            for item in transaction:
                if item in position:
                    table[row, position[item]] = True
                elif self.handle_unknown == "error":
                    raise ValueError("item no visto en fit: {!r}".format(item))
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(table, columns=self.columns_, index=index)

    def fit_transform(self, X: Transactions, y: Any = None) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def inverse_transform(self, table: pd.DataFrame) -> List[List[Any]]:
        """De la tabla booleana a las transacciones (items en orden alfabetico)."""
        return as_transactions(table)


class RuleMiner(ParamsMixin):
    """
    Extrae reglas de asociacion de transacciones (una lista de listas de items o una tabla booleana).

    Argumentos:
        method: ``"fpgrowth"`` (todas las reglas con soporte >= `min_supp`), ``"top_k"`` (las `k` de mayor soporte,
            sin umbral) o ``"rare"`` (reglas entre items frecuentes que juntos son raros; `max_supp` define que es
            frecuente)
        min_supp: soporte minimo (con ``"fpgrowth"``; con ``"top_k"`` y ``"rare"`` es un suelo opcional)
        k: numero de reglas (``"top_k"``)
        max_supp: umbral frecuente/raro (``"rare"``)
        min_count: apariciones minimas de una combinacion rara (``"rare"``)
        metric, min_threshold: medida de interes y su umbral minimo (ver ``ARM.association_rules``); con
            ``"top_k"`` y ``"rare"`` el umbral se aplica a la confianza (`metric` debe ser ``"confidence"``)
        min_lift: lift minimo, opcional
        max_len: items maximos de una regla (``"top_k"``, ``"rare"``)
        alpha: si no es None, se conservan solo las reglas significativas a ese nivel (``ARM.significant_rules``),
            con la `correction` y el contraste `test` indicados
        prune: si es verdadero, se quitan las reglas redundantes (``ARM.prune_redundant_rules``)
        algorithm: con ``"fpgrowth"``, como se buscan los itemsets: ``"fpgrowth"`` (``FFIM.fpgrowth``) o ``"dense"``
            (``FIM.dense_itemsets``, mejor con pocos items y muchas transacciones); mismo resultado

    Atributos tras ``fit``: ``itemsets_`` (soporte e itemsets; solo con ``"fpgrowth"``), ``rules_`` (el formato de
    ``ARM.association_rules``), ``n_transactions_``.
    """

    METHODS = ("fpgrowth", "top_k", "rare")

    def __init__(self, method: str = "fpgrowth", min_supp: float = 0.1, k: Optional[int] = None,
                 max_supp: Optional[float] = None, min_count: int = 2, metric: str = "confidence",
                 min_threshold: float = 0.6, min_lift: Optional[float] = None, max_len: Optional[int] = None,
                 alpha: Optional[float] = None, correction: str = "fdr_bh", test: str = "fisher",
                 prune: bool = False, algorithm: str = "fpgrowth") -> None:
        self.method = method
        self.algorithm = algorithm
        self.min_supp = min_supp
        self.k = k
        self.max_supp = max_supp
        self.min_count = min_count
        self.metric = metric
        self.min_threshold = min_threshold
        self.min_lift = min_lift
        self.max_len = max_len
        self.alpha = alpha
        self.correction = correction
        self.test = test
        self.prune = prune

    def fit(self, X: Transactions, y: Any = None) -> "RuleMiner":
        from .ARM import association_rules, prune_redundant_rules, rare_rules, significant_rules, top_k_rules
        from .FFIM import fpgrowth
        from .FIM._shared import key_to_itemset
        from .FIM.dense import dense_itemsets

        if self.method not in self.METHODS:
            raise ValueError("method debe ser uno de {}".format(self.METHODS))
        transactions = as_transactions(X)
        self.n_transactions_ = len(transactions)
        self.itemsets_ = pd.DataFrame({"support": [], "itemsets": []})
        if self.method != "fpgrowth" and self.metric != "confidence":
            raise ValueError("con method='{}' la medida es la confianza (metric='confidence')".format(self.method))
        if self.method == "fpgrowth":
            if self.algorithm == "dense":
                self.itemsets_ = dense_itemsets(transactions, self.min_supp)
            elif self.algorithm == "fpgrowth":
                frequent = fpgrowth(transactions, self.min_supp)
                self.itemsets_ = pd.DataFrame({"support": list(frequent.values()),
                                               "itemsets": [key_to_itemset(key) for key in frequent]})
            else:
                raise ValueError("algorithm debe ser 'fpgrowth' o 'dense'")
            rules = association_rules(self.itemsets_, metric=self.metric, min_threshold=self.min_threshold)
            if self.min_lift is not None:
                rules = rules[rules["lift"] >= self.min_lift].reset_index(drop=True)
        elif self.method == "top_k":
            if self.k is None:
                raise ValueError("method='top_k' necesita el parametro k")
            rules = top_k_rules(transactions, self.k, min_conf=self.min_threshold, min_lift=self.min_lift,
                                max_len=self.max_len, min_supp=self.min_supp)
        else:
            if self.max_supp is None:
                raise ValueError("method='rare' necesita el parametro max_supp")
            rules = rare_rules(transactions, self.max_supp, min_supp=self.min_supp, min_count=self.min_count,
                               min_conf=self.min_threshold, min_lift=self.min_lift, max_len=self.max_len)
        if self.alpha is not None and len(rules):
            rules = significant_rules(rules, self.alpha, self.correction, n_transactions=self.n_transactions_,
                                      method=self.test)
        if self.prune and len(rules):
            rules = prune_redundant_rules(rules)
        self.rules_ = rules
        return self

    def transform(self, X: Transactions) -> pd.DataFrame:
        """
        Tabla booleana con una columna por regla de ``rules_`` (``0``, ``1``...): verdadero cuando la transaccion cumple
        el antecedente de la regla. Si hay items de las reglas que no estan en `X`, la regla no se cumple.
        """
        if not hasattr(self, "rules_"):
            raise ValueError("el estimador no esta ajustado: llame antes a fit")
        transactions = as_transactions(X)
        known = sorted({item for _, rule in self.rules_.iterrows() for item in rule["antecedents"]}, key=str)
        encoded = TransactionEncoder(handle_unknown="ignore").fit([known]).transform(transactions)
        table = ItemTable(encoded)
        columns = {position: table.holds(antecedent) for position, antecedent in enumerate(self.rules_["antecedents"])}
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(columns, index=index if index is not None else range(len(transactions)), dtype=bool)

    def fit_transform(self, X: Transactions, y: Any = None) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Texto de cada regla de ``rules_``, en el orden de las columnas de ``transform``."""
        return ["{} => {}".format(",".join(sorted(map(str, a))), ",".join(sorted(map(str, c))))
                for a, c in zip(self.rules_["antecedents"], self.rules_["consequents"])]
