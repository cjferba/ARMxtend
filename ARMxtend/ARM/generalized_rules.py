# -*- coding: utf-8 -*-
"""
GENERALIZED (multilevel) association rules: rules whose items can be products or categories of a taxonomy
(``FIM.taxonomy``), such as "outerwear => footwear".

``generalized_rules`` mines the generalized itemsets (with no items next to their ancestors, with a minimum support
optionally different per level) and converts them into rules with ``ARM.association_rules``; since no itemset
contains an item and its ancestor, the trivial rules ``item => ancestor`` (confidence 1) do not come out.

Categories generate many rules that merely repeat, at another level, what a more general rule already says.
``interesting_rules`` keeps the "R-interesting" ones of Srikant and Agrawal (1995): a rule is interesting if it has no
ancestors (rules that result from replacing items by their ancestors in the same set of rules) or if, with respect to
each of its CLOSE interesting ancestors, its real support or confidence are at least R times those EXPECTED from the
ancestor. With ``X^ => Y^`` the ancestor of ``X => Y`` and ``z^`` the ancestor of each item ``z``:

    expected support    = support(X^ u Y^) * prod_{z in X u Y} support(z) / support(z^)
    expected confidence = confidence(X^ => Y^) * prod_{y in Y} support(y) / support(y^)

That is, the specialized rule only contributes something if it is better than would be expected if each item behaved
like its category.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..FIM.taxonomy import MinSupp, Taxonomy, ancestor_replacements, generalized_itemsets
from .association_rules import association_rules

_EPS = 1e-9


def generalized_rules(transactions: Iterable[Iterable[Any]], taxonomy: Taxonomy, min_supp: MinSupp,
                      metric: str = "confidence", min_threshold: float = 0.8, max_len: Optional[int] = None,
                      interest: Optional[float] = None, exclusive: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """
    Mine generalized rules (see the module docstring).

    Args:
        transactions: one transaction per element, each a collection of items (leaves or categories)
        taxonomy (Taxonomy): hierarchy of items
        min_supp: minimum support, or ``{level: support}`` (see ``FIM.taxonomy.generalized_itemsets``)
        metric, min_threshold: measure of ``ARM.association_rules`` and its minimum threshold
        max_len (int, optional): maximum number of items of a rule
        interest (float, optional): if given (>= 1), ``interesting_rules`` is applied with that R
        exclusive (dict, optional): ``{item: group}``, items of the same group that cannot go together (see
            ``FIM.taxonomy.generalized_itemsets``)

    Returns:
        pandas.DataFrame with the format of ``ARM.association_rules`` plus the column 'level' (the level of the deepest
        item of the rule). ``attrs`` holds 'n_transactions', 'n_candidates' (for ``ARM.significance``), 'taxonomy' and
        'supports' (support of each frequent itemset, which ``interesting_rules`` uses)
    """
    itemsets = generalized_itemsets(transactions, taxonomy, min_supp, max_len=max_len, exclusive=exclusive)
    total = itemsets.attrs["n_transactions"]
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold) if len(itemsets) \
        else pd.DataFrame(columns=["antecedents", "consequents", "antecedent support", "consequent support", "support",
                                   "confidence", "lift"])
    rules["level"] = [max(taxonomy.level(item) for item in set(a) | set(c))
                      for a, c in zip(rules["antecedents"], rules["consequents"])]
    rules.attrs["n_transactions"] = total
    rules.attrs.setdefault("n_candidates", len(rules))
    rules.attrs["taxonomy"] = taxonomy
    rules.attrs["supports"] = {itemset: float(support) for itemset, support in zip(itemsets["itemsets"], itemsets["support"])}
    return interesting_rules(rules, taxonomy, interest) if interest is not None and len(rules) else rules


def interesting_rules(rules: pd.DataFrame, taxonomy: Optional[Taxonomy] = None, interest: float = 1.1,
                      supports: Optional[Dict[FrozenSet[str], float]] = None) -> pd.DataFrame:
    """
    The R-interesting rules of `rules` (see the module docstring), with R = `interest`.

    Args:
        rules (pandas.DataFrame): output of ``generalized_rules`` WITHOUT pruning (the ancestors of a rule must be in
            the set), with 'antecedents', 'consequents', 'support' and 'confidence'
        taxonomy (Taxonomy), supports (dict {frozenset: support}): by default those of ``rules.attrs``
        interest (float): R >= 1; 1 only removes the rules that are worse than expected

    Returns:
        the interesting rules, in the same order and index, with the column 'interest': the smallest, over the close
        ancestors, of max(real support / expected, real confidence / expected); NaN if the rule has no ancestors
    """
    taxonomy = taxonomy if taxonomy is not None else rules.attrs.get("taxonomy")
    supports = supports if supports is not None else rules.attrs.get("supports")
    if taxonomy is None or supports is None:
        raise ValueError("the taxonomy and the supports of the itemsets are needed (rules.attrs, or the arguments)")
    if interest < 1:
        raise ValueError("interest debe ser >= 1")
    result = rules.copy()
    result["interest"] = np.nan
    if len(rules) == 0:
        return result

    items: List[Tuple[str, ...]] = []
    sizes: List[int] = []
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        items.append(tuple(sorted(antecedent)) + tuple(sorted(consequent)))
        sizes.append(len(antecedent))
    lookup = {(frozenset(a), frozenset(c)): row for row, (a, c) in enumerate(zip(rules["antecedents"], rules["consequents"]))}
    support = rules["support"].to_numpy(dtype=float)
    confidence = rules["confidence"].to_numpy(dtype=float)

    def single(item: str) -> float:
        return supports[frozenset([item])]

    used = {item for combination in items for item in combination}
    keep = np.zeros(len(rules), dtype=bool)
    ratios = np.full(len(rules), np.nan)
    for row in sorted(range(len(rules)), key=lambda r: sum(taxonomy.level(item) for item in items[r])):
        size = sizes[row]
        ancestors = []
        for combination in ancestor_replacements(items[row], taxonomy, used):
            antecedent, consequent = frozenset(combination[:size]), frozenset(combination[size:])
            if len(antecedent) != size or len(consequent) != len(items[row]) - size or antecedent & consequent:
                continue
            other = lookup.get((antecedent, consequent))
            if other is not None and keep[other]:
                ancestors.append((combination, other))
        close = [(combination, other) for combination, other in ancestors
                 if not any(farther != combination and all(c == f or c in taxonomy.ancestors(f)
                                                           for c, f in zip(combination, farther))
                            for farther, _ in ancestors)]
        best = np.inf
        for combination, other in close:
            scale = np.prod([single(item) / single(general) for item, general in zip(items[row], combination)])
            scale_consequent = np.prod([single(item) / single(general)
                                        for item, general in zip(items[row][size:], combination[size:])])
            best = min(best, max(support[row] / (support[other] * scale), confidence[row] / (confidence[other] * scale_consequent)))
        if close:
            ratios[row] = best
        keep[row] = not close or best >= interest - _EPS
    result["interest"] = ratios
    result = result[keep]
    result.attrs.update({key: value for key, value in rules.attrs.items()})
    return result
