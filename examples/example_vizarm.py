# -*- coding: utf-8 -*-
"""
Example: transform a set of association rules (ARM.association_rules)
into the typed (item/rule) VizARE graph, following:

    Fernandez-Basso, C., Ruiz, M.D., Molina-Solana, M., Martin-Bautista,
    M.J. (2026). "VizARE: An Intermediate Representation to Support the
    Visualization of Association Rules in Data Mining". Future Internet,
    18(7), 374.

and export it to the proposed intermediate format (JSON Graph Format), as well as
GraphML/DOT (Gephi/Cytoscape/Graphviz/NetworkX) and to the optional
summary layer (Section 4.2) to inspect large sets of rules.
"""
import pandas as pd

from ARMxtend.ARM.association_rules import association_rules
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM._shared import key_to_itemset
from ARMxtend.VizARM import AREtoGraph

TRANSACTIONS = [
    "A,B,C", "B,D", "A,C,D", "B,C", "A,C",
    "B,D", "A,B,C", "B,C,D", "A,B,D", "B,C,D",
]


def main():
    transactions = [line.split(",") for line in TRANSACTIONS]
    freqItemsets = fpgrowth(transactions, min_supp=0.3)
    itemsetsDf = pd.DataFrame({
        "support": list(freqItemsets.values()),
        "itemsets": [key_to_itemset(key) for key in freqItemsets],
    })
    rulesDf = association_rules(itemsetsDf, metric="confidence", min_threshold=0.7)

    graph = AREtoGraph.from_dataframe(rulesDf, rule_measures=("support", "confidence", "lift"))
    itemNodes = [n for n, d in graph.graph.nodes(data=True) if d["kind"] == "item"]
    ruleNodes = [n for n, d in graph.graph.nodes(data=True) if d["kind"] == "rule"]
    print("Grafo: {} nodos item, {} nodos rule, {} aristas".format(
        len(itemNodes), len(ruleNodes), graph.graph.number_of_edges()))

    # Each rule is a node of its own, connected to its items by edges
    # 'antecedent'/'consequent' -- there is no direct item -> item edge.
    for ruleId in ruleNodes:
        data = graph.graph.nodes[ruleId]
        antecedent = sorted(graph.graph.predecessors(ruleId))
        consequent = sorted(graph.graph.successors(ruleId))
        print("  {}: {} -> {}  support={:.2f} confidence={:.2f} lift={:.2f}".format(
            ruleId, antecedent, consequent, data["support"], data["confidence"], data["lift"]))

    print("\nJGF (formato intermedio propuesto, primeras lineas):")
    print("\n".join(graph.to_jgf().splitlines()[:12]))

    # Summary layer (Section 4.2): groups the rules with the same consequent.
    summaryIds = graph.summarize(signature="consequent", measures=("confidence", "lift"))
    print("\nSummary by consequent ({} groups):".format(len(summaryIds)))
    for summaryId in summaryIds:
        data = graph.graph.nodes[summaryId]
        print("  {} rules -> {}  confidence in [{:.2f}, {:.2f}] (mean {:.2f})".format(
            data["rule_count"], sorted(graph.graph.successors(summaryId)),
            data["confidence_min"], data["confidence_max"], data["confidence_mean"]))

    # Self-contained interactive viewer: a single HTML file, no network or dependencies
    # (with path="rules.html" it is written to disk and opened in any browser).
    html = graph.to_html(title="ARMxtend - VizARE")
    print("\nVisor HTML interactivo: {} bytes, grafo JGF embebido".format(len(html)))

    return graph


if __name__ == "__main__":
    main()
