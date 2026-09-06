# -*- coding: utf-8 -*-
"""
Test de regresion para el puente SARE.extractAssociationRules, que conecta
el arbol de itemsets frecuentes (FIT) de FIMoTS con ARM.association_rules.

Se construye a mano una FIMoTS_Structure con los itemsets frecuentes ya
conocidos (y verificados en otros tests) del dataset de ejemplo (Tabla 2) de
Fernandez-Basso, Ruiz & Martin-Bautista (2024), sin necesidad de ejecutar el
algoritmo FIMoTS completo (que requiere Spark Streaming).
"""
import pytest

from ARMxtend.SARE import extractAssociationRules
from ARMxtend.SARE.FIMoTS_Node import FIMoTS_Node
from ARMxtend.SARE.FIMoTS_Tree import FIMoTS_Tree
from ARMxtend.SARE.FIMoTS_List import FIMoTS_List
from ARMxtend.SARE.FIMoTS_Bounds import FIMoTS_Bounds
from ARMxtend.SARE.FIMoTS_Structure import FIMoTS_Structure

FREQ_ITEMSETS = {
    "A": 0.5, "B": 0.8, "C": 0.7, "D": 0.6,
    "A-B": 0.3, "A-C": 0.4, "B-C": 0.5, "B-D": 0.5, "C-D": 0.3,
}


def _build_fimots_structure(freq_itemsets):
    nodeMap = {}
    for itemsetKey, support in freq_itemsets.items():
        items = sorted(itemsetKey.split("-"))
        nodeMap[itemsetKey] = FIMoTS_Node(itemPrefix=items, relativeSupport=support)

    tree = FIMoTS_Tree(nodeMap)
    frequentBounds = FIMoTS_List([FIMoTS_Bounds(itemNodes=list(freq_itemsets.keys()))])
    return FIMoTS_Structure(itemsetsTree=tree, frequentItemsetsBounds=frequentBounds,
                            infrequentItemsetsBounds=FIMoTS_List([]))


def test_extract_association_rules_matches_bd_are_generate_rules():
    from ARMxtend.FIM.BD_ARE import generate_rules

    structure = _build_fimots_structure(FREQ_ITEMSETS)
    rulesDf = extractAssociationRules(structure, min_threshold=0.7)

    expectedRules = generate_rules(FREQ_ITEMSETS, min_conf=0.7)
    actualRules = {
        (frozenset(row["antecedents"]) - frozenset(), tuple(sorted(row["consequents"]))[0]
         if len(row["consequents"]) == 1 else None): row["confidence"]
        for _, row in rulesDf.iterrows()
    }

    assert len(rulesDf) == len(expectedRules)
    for _, row in rulesDf.iterrows():
        antecedentKey = "-".join(sorted(row["antecedents"]))
        consequentKey = "-".join(sorted(row["consequents"]))
        assert (antecedentKey, consequentKey) in expectedRules
        assert row["confidence"] == pytest.approx(expectedRules[(antecedentKey, consequentKey)])


def test_extract_association_rules_empty_when_no_frequent_itemsets():
    structure = _build_fimots_structure({})
    rulesDf = extractAssociationRules(structure, min_threshold=0.7)
    assert rulesDf.empty
