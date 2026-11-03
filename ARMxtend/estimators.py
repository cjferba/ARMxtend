# -*- coding: utf-8 -*-
"""
Estimators with the scikit-learn interface (``fit`` / ``transform`` / ``get_params``), without depending on
scikit-learn:

* ``TransactionEncoder``: transactions (lists of items) <-> boolean table, one column per item. It is the input
  format of almost all of ARMxtend (``ARM.association_rules`` reads itemsets, ``anomaly``/``prediction``/
  ``classification`` read boolean tables) and that of ``mlxtend.preprocessing.TransactionEncoder``.
* ``RuleMiner``: extracts the rules of a collection of transactions, with the chosen algorithm (`method`):
  ``"fpgrowth"`` (support threshold `min_supp`), ``"top_k"`` (the `k` of highest support) or ``"rare"`` (rare
  combinations of frequent items, `max_supp`). It leaves ``itemsets_`` and ``rules_`` after ``fit``; ``transform``
  converts transactions into a table of rules that hold, useful as features of another model in a ``Pipeline``.

The estimators store their parameters untouched (``ParamsMixin``): ``sklearn.base.clone`` and ``GridSearchCV`` work
with them, and ``ARM.RuleClassifier`` (CBA) follows the same protocol.

@author Carlos Fernandez-Basso
"""
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .ARM._apply import ItemTable
from .base import ParamsMixin

Transactions = Union[Iterable[Iterable[Any]], pd.DataFrame]


def as_transactions(data: Transactions) -> List[List[Any]]:
    """Transactions as a list of lists: a boolean (or 0/1) table gives, per row, the columns that are true."""
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy()
        if values.dtype != bool:
            if not np.isin(values, (0, 1)).all():
                raise ValueError("an input table must be boolean (or 0/1): one column per item")
            values = values.astype(bool)
        names = list(data.columns)
        return [[names[position] for position in np.flatnonzero(row)] for row in values]
    return [list(transaction) for transaction in data]


class TransactionEncoder(ParamsMixin):
    """
    Transactions -> boolean table (one row per transaction, one column per item, in alphabetical order).

    Args:
        handle_unknown: what to do in ``transform`` with items not seen in ``fit``: ``"ignore"`` (they are dropped,
            default) or ``"error"``

    Attributes after ``fit``: ``columns_`` (sorted list of items).
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
            raise ValueError("the encoder is not fitted: call fit first")
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
        """From the boolean table to transactions (items in alphabetical order)."""
        return as_transactions(table)


class RuleMiner(ParamsMixin):
    """
    Extract association rules from transactions (a list of lists of items or a boolean table).

    Args:
        method: ``"fpgrowth"`` (all the rules with support >= `min_supp`), ``"top_k"`` (the `k` of highest support, no
            threshold) or ``"rare"`` (rules between frequent items that are rare together; `max_supp` defines what is
            frequent)
        min_supp: minimum support (with ``"fpgrowth"``; with ``"top_k"`` and ``"rare"`` it is an optional floor)
        k: number of rules (``"top_k"``)
        max_supp: frequent/rare threshold (``"rare"``)
        min_count: minimum occurrences of a rare combination (``"rare"``)
        metric, min_threshold: interest measure and its minimum threshold (see ``ARM.association_rules``); with
            ``"top_k"`` and ``"rare"`` the threshold applies to confidence (`metric` must be ``"confidence"``)
        min_lift: minimum lift, optional
        max_len: maximum items of a rule (``"top_k"``, ``"rare"``)
        alpha: if not None, only the rules significant at that level are kept (``ARM.significant_rules``), with the
            indicated `correction` and `test`
        prune: if true, redundant rules are removed (``ARM.prune_redundant_rules``)
        algorithm: with ``"fpgrowth"``, how the itemsets are searched: ``"fpgrowth"`` (``FFIM.fpgrowth``) or ``"dense"``
            (``FIM.dense_itemsets``, better with few items and many transactions); same result

    Attributes after ``fit``: ``itemsets_`` (support and itemsets; only with ``"fpgrowth"``), ``rules_`` (the format
    of ``ARM.association_rules``), ``n_transactions_``.
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
            raise ValueError("with method='{}' the measure is confidence (metric='confidence')".format(self.method))
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
        Boolean table with one column per rule of ``rules_`` (``0``, ``1``...): true when the transaction satisfies the
        antecedent of the rule. If there are items of the rules that are not in `X`, the rule does not hold.
        """
        if not hasattr(self, "rules_"):
            raise ValueError("the estimator is not fitted: call fit first")
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
        """Text of each rule of ``rules_``, in the order of the columns of ``transform``."""
        return ["{} => {}".format(",".join(sorted(map(str, a))), ",".join(sorted(map(str, c))))
                for a, c in zip(self.rules_["antecedents"], self.rules_["consequents"])]
