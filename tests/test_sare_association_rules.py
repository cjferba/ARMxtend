# -*- coding: utf-8 -*-
"""
Regression test for the SARE.extractAssociationRules bridge, which connects
the frequent itemset tree (FIT) of FIMoTS with ARM.association_rules.

A FIMoTS_Structure is built by hand with the already known frequent itemsets
(verified in other tests) of the example dataset (Table 2) of
Fernandez-Basso, Ruiz & Martin-Bautista (2024), without needing to run the
full FIMoTS algorithm (which requires Spark Streaming).
"""
import pytest

pytest.importorskip("pyspark")  # SARE imports pyspark when loaded (it is part of the "spark" extra)

from ARMxtend.SARE import extractAssociationRules  # noqa: E402
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
