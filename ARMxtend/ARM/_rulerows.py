# -*- coding: utf-8 -*-
"""
Filas de reglas con las medidas estandar (soporte, confianza, lift y las demas de ``_measures``),
a partir de los soportes de la regla, del antecedente y del consecuente. Lo comparten los generadores de
reglas que no pasan por ``association_rules``.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, FrozenSet, List

from ._measures import MEASURES

RULE_COLUMNS: List[str] = ["antecedents", "consequents", "antecedent support", "consequent support", "support",
                           "confidence", "lift"] + [name for name in MEASURES if name not in ("confidence", "lift")]


def rule_row(antecedent: FrozenSet[str], consequent: FrozenSet[str], support: float, support_x: float,
             support_y: float) -> Dict[str, Any]:
    """Una regla ``X => Y`` con todas las medidas, a partir de los tres soportes."""
    row: Dict[str, Any] = {"antecedents": antecedent, "consequents": consequent, "antecedent support": support_x,
                           "consequent support": support_y, "support": support}
    for name, function in MEASURES.items():
        row[name] = float(function(support, support_x, support_y))
    return row
