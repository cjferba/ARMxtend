# -*- coding: utf-8 -*-
"""
Rule rows with the standard measures (support, confidence, lift and the rest of ``_measures``),
computed from the supports of the rule, the antecedent and the consequent. Shared by the generators of
rules that do not go through ``association_rules``.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, FrozenSet, List

from ._measures import MEASURES

RULE_COLUMNS: List[str] = ["antecedents", "consequents", "antecedent support", "consequent support", "support",
                           "confidence", "lift"] + [name for name in MEASURES if name not in ("confidence", "lift")]


def rule_row(antecedent: FrozenSet[str], consequent: FrozenSet[str], support: float, support_x: float,
             support_y: float) -> Dict[str, Any]:
    """A rule ``X => Y`` with all the measures, computed from the three supports."""
    row: Dict[str, Any] = {"antecedents": antecedent, "consequents": consequent, "antecedent support": support_x,
                           "consequent support": support_y, "support": support}
    for name, function in MEASURES.items():
        row[name] = float(function(support, support_x, support_y))
    return row
