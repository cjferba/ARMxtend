# -*- coding: utf-8 -*-
"""
Tests de regresion para ARMxtend.VizARM.AREtoGraph, siguiendo la
metodologia VizARE (grafo tipado item/rule, Algoritmo 1) de:
    Fernandez-Basso, C., Ruiz, M.D., Molina-Solana, M., Martin-Bautista,
    M.J. (2026). "VizARE: An Intermediate Representation to Support the
    Visualization of Association Rules in Data Mining". Future Internet,
    18(7), 374.
"""
import json

import pandas as pd
import pytest

from ARMxtend.ARM.association_rules import association_rules
from ARMxtend.VizARM import AREtoGraph


@pytest.fixture
def rules_df():
    df = pd.DataFrame({
        "support": [0.5, 0.8, 0.7, 0.6, 0.3, 0.4, 0.5, 0.5, 0.3],
        "itemsets": [frozenset(x) for x in
                    [["A"], ["B"], ["C"], ["D"], ["A", "B"], ["A", "C"], ["B", "C"], ["B", "D"], ["C", "D"]]],
    })
    return association_rules(df, metric="confidence", min_threshold=0.7)


def test_from_dataframe_builds_typed_item_and_rule_nodes(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df, rule_measures=("confidence", "lift"))

    itemNodes = {node for node, data in graph.graph.nodes(data=True) if data["kind"] == "item"}
    ruleNodes = {node for node, data in graph.graph.nodes(data=True) if data["kind"] == "rule"}

    assert itemNodes == {"A", "B", "C", "D"}
    assert ruleNodes == {"rule::A=>C", "rule::C=>B", "rule::D=>B"}

    # A -> C: antecedent edge A -> rule, consequent edge rule -> C (not a
    # direct A -> C edge, unlike a flat item-to-item graph).
    assert graph.graph.edges[("A", "rule::A=>C")]["relation"] == "antecedent"
    assert graph.graph.edges[("rule::A=>C", "C")]["relation"] == "consequent"
    assert not graph.graph.has_edge("A", "C")

    # The rule's measures live on the rule node itself, not on an edge.
    ruleData = graph.graph.nodes["rule::A=>C"]
    assert ruleData["confidence"] == pytest.approx(0.8)
    assert ruleData["lift"] == pytest.approx(1.142857142857143)


def test_add_rule_supports_multi_item_antecedent_and_consequent():
    graph = AREtoGraph()
    ruleId = graph.add_rule(["A", "B"], ["C", "D"], measures={"support": 0.4, "confidence": 0.9})

    assert set(graph.graph.predecessors(ruleId)) == {"A", "B"}
    assert set(graph.graph.successors(ruleId)) == {"C", "D"}
    assert all(graph.graph.edges[(item, ruleId)]["relation"] == "antecedent" for item in ("A", "B"))
    assert all(graph.graph.edges[(ruleId, item)]["relation"] == "consequent" for item in ("C", "D"))


def test_item_group_defaults_to_attribute_prefix():
    graph = AREtoGraph()
    graph.add_rule(["temperature_cold"], ["occupation_low"], measures={"support": 0.3})
    assert graph.graph.nodes["temperature_cold"]["group"] == "temperature"
    assert graph.graph.nodes["occupation_low"]["group"] == "occupation"


def test_export_jgf_round_trips_nodes_and_edges(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df, rule_measures=("confidence",))
    jgf = json.loads(graph.exportGraph(type=0))

    assert jgf["graph"]["directed"] is True
    assert jgf["graph"]["nodes"]["rule::A=>C"]["metadata"]["kind"] == "rule"
    assert jgf["graph"]["nodes"]["rule::A=>C"]["metadata"]["confidence"] == pytest.approx(0.8)
    assert jgf["graph"]["nodes"]["A"]["metadata"]["kind"] == "item"

    edgeRelations = {(edge["source"], edge["target"]): edge["relation"] for edge in jgf["graph"]["edges"]}
    assert edgeRelations[("A", "rule::A=>C")] == "antecedent"
    assert edgeRelations[("rule::A=>C", "C")] == "consequent"


def test_export_graphml_and_dot_are_non_empty_strings(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df)
    graphml = graph.exportGraph(type=1)
    dot = graph.exportGraph(type=2)
    assert "<graphml" in graphml
    assert dot.startswith("digraph ARM {")
    assert '"A" -> "rule::A=>C"' in dot


def test_export_graph_object(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df)
    import networkx as nx
    assert isinstance(graph.exportGraph(type=3), nx.DiGraph)


def test_export_graph_invalid_type_raises(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df)
    with pytest.raises(ValueError):
        graph.exportGraph(type=99)


def test_summarize_by_consequent_aggregates_measures(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df, rule_measures=("support", "confidence"))
    summaryIds = graph.summarize(signature="consequent", measures=("confidence",))

    # C->B (confidence .7142857) and D->B (confidence .8333333) share consequent {B}.
    bGroup = next(
        graph.graph.nodes[sid] for sid in summaryIds
        if graph.graph.nodes[sid]["members"] == ["rule::C=>B", "rule::D=>B"]
        or set(graph.graph.nodes[sid]["members"]) == {"rule::C=>B", "rule::D=>B"}
    )
    assert bGroup["rule_count"] == 2
    assert bGroup["confidence_min"] == pytest.approx(0.7142857142857143)
    assert bGroup["confidence_max"] == pytest.approx(0.8333333333333334)
    assert bGroup["confidence_mean"] == pytest.approx((0.7142857142857143 + 0.8333333333333334) / 2)

    # Summary nodes connect to the same item nodes as the rules they summarize.
    assert set(graph.graph.predecessors([sid for sid in summaryIds if graph.graph.nodes[sid]["rule_count"] == 1][0])) == {"A"}


def test_summarize_invalid_signature_raises(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df)
    with pytest.raises(ValueError):
        graph.summarize(signature="not-a-signature")
