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
import re

import numpy as np
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


# --- visor HTML interactivo, Cypher y documentos ------------------------------------------


def _embedded(html, element_id):
    match = re.search(r'<script type="application/json" id="{}">(.*?)</script>'.format(element_id), html, re.S)
    return json.loads(match.group(1))


def test_html_viewer_embeds_the_jgf_and_is_self_contained(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df, rule_measures=("support", "confidence", "lift"))
    html = graph.to_html(title="My rules", measure="lift", min_value=1.0)

    assert html.lstrip().startswith("<!DOCTYPE html>") and "<title>My rules</title>" in html
    assert _embedded(html, "vizare-data") == graph.to_jgf_dict()
    assert _embedded(html, "vizare-options") == {"measure": "lift", "min_value": 1.0}
    # sin placeholders sin sustituir ni dependencias externas (CDN, fuentes, imagenes remotas)
    assert "__DATA__" not in html and "__TITLE__" not in html and "__OPTIONS__" not in html
    assert "http://" not in html and "https://" not in html
    assert "src=" not in html and "<link" not in html


def test_html_viewer_writes_the_file(rules_df, tmp_path):
    target = tmp_path / "rules.html"
    html = AREtoGraph.from_dataframe(rules_df).to_html(target)
    assert target.read_text(encoding="utf-8") == html


def test_html_viewer_escapes_hostile_labels_and_non_finite_numbers():
    graph = AREtoGraph()
    graph.add_rule(["</script><img src=x onerror=alert(1)>"], ["B&C"],
                   measures={"support": float("nan"), "confidence": np.float64(0.5)}, rule_id="r<1>")
    html = graph.to_html(title="<b>x</b>")

    assert "<img" not in html and "</script><img" not in html
    assert "<title>&lt;b&gt;x&lt;/b&gt;</title>" in html
    data = _embedded(html, "vizare-data")["graph"]
    assert "</script><img src=x onerror=alert(1)>" in data["nodes"]               # el dato sobrevive intacto
    assert data["nodes"]["r<1>"]["metadata"]["support"] is None                   # NaN -> null (JSON valido)
    assert data["nodes"]["r<1>"]["metadata"]["confidence"] == 0.5
    # el JGF normal tambien es JSON estricto
    json.loads(graph.to_jgf(), parse_constant=lambda c: pytest.fail("NaN/Infinity en el JSON: " + c))


def test_max_rules_keeps_the_best_rules_with_their_items(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df, rule_measures=("support", "confidence", "lift"))
    html = graph.to_html(max_rules=1, sort_by="lift")
    nodes = _embedded(html, "vizare-data")["graph"]["nodes"]

    ruleIds = [i for i, n in nodes.items() if n["metadata"]["kind"] == "rule"]
    assert ruleIds == ["rule::A=>C"]                                   # la de mayor lift (1.14)
    assert {i for i, n in nodes.items() if n["metadata"]["kind"] == "item"} == {"A", "C"}


def test_max_rules_shrinks_summary_members(rules_df):
    graph = AREtoGraph.from_dataframe(rules_df, rule_measures=("support", "confidence"))
    graph.summarize(signature="consequent", measures=("confidence",))
    nodes = _embedded(graph.to_html(max_rules=2, sort_by="confidence"), "vizare-data")["graph"]["nodes"]

    keptRules = {i for i, n in nodes.items() if n["metadata"]["kind"] == "rule"}
    assert len(keptRules) == 2
    for node in nodes.values():
        if node["metadata"]["kind"] == "summary":
            assert set(node["metadata"]["members"]) <= keptRules
            assert node["metadata"]["rule_count"] == len(node["metadata"]["members"]) > 0


def test_cypher_export_creates_nodes_and_relationships(rules_df):
    cypher = AREtoGraph.from_dataframe(rules_df, rule_measures=("confidence",)).to_cypher()
    statements = cypher.splitlines()

    assert "CREATE (:Item {id: 'A', name: 'A', kind: 'item', group: 'A'});" in statements
    assert any(s.startswith("CREATE (:Rule {id: 'rule::A=>C'") and "confidence: 0.8" in s for s in statements)
    assert "MATCH (a {id: 'A'}), (b {id: 'rule::A=>C'}) CREATE (a)-[:ANTECEDENT]->(b);" in statements
    assert "MATCH (a {id: 'rule::A=>C'}), (b {id: 'C'}) CREATE (a)-[:CONSEQUENT]->(b);" in statements
    assert all(s.endswith(";") for s in statements)


def test_cypher_escapes_quotes_and_backslashes():
    graph = AREtoGraph()
    graph.add_rule(["it's"], ["back\\slash"], measures={"support": 0.1})
    cypher = graph.to_cypher()
    assert "id: 'it\\'s'" in cypher and "id: 'back\\\\slash'" in cypher


def test_documents_export_is_ready_for_a_document_database(rules_df):
    documents = AREtoGraph.from_dataframe(rules_df, rule_measures=("confidence",)).to_documents()

    byId = {d["_id"]: d for d in documents["nodes"]}
    assert byId["rule::A=>C"]["kind"] == "rule" and byId["rule::A=>C"]["confidence"] == pytest.approx(0.8)
    assert byId["A"]["kind"] == "item"
    assert {"source": "A", "target": "rule::A=>C", "relation": "antecedent"} in documents["edges"]
    json.dumps(documents)                                              # serializable tal cual
    assert len(AREtoGraph.from_dataframe(rules_df).to_documents(max_rules=1)["nodes"]) == 3   # 1 regla + 2 items
