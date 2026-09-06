# -*- coding: utf-8 -*-
"""Tests de regresion para ARMxtend.VizARM.AREtoGraph."""
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


def test_from_dataframe_builds_expected_nodes_and_edges(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df, rule_measures=("confidence", "lift"))
    assert set(graph.graph.nodes) == {"A", "B", "C", "D"}
    assert set(graph.graph.edges) == {("A", "C"), ("C", "B"), ("D", "B")}
    edgeData = graph.graph.edges[("A", "C")]
    assert edgeData["confidence"] == pytest.approx(0.8)


def test_export_graphml_and_dot_are_non_empty_strings(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df)
    graphml = graph.exportGraph(type=0)
    dot = graph.exportGraph(type=1)
    assert "<graphml" in graphml
    assert dot.startswith("digraph ARM {")
    assert '"A" -> "C"' in dot


def test_export_graph_object(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df)
    import networkx as nx
    assert isinstance(graph.exportGraph(type=2), nx.DiGraph)


def test_export_graph_invalid_type_raises(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df)
    with pytest.raises(ValueError):
        graph.exportGraph(type=99)
