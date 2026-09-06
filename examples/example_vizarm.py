# -*- coding: utf-8 -*-
"""
Ejemplo: transformar un conjunto de reglas de asociacion (ARM.association_rules)
en el grafo tipado (item/rule) VizARE, siguiendo:

    Fernandez-Basso, C., Ruiz, M.D., Molina-Solana, M., Martin-Bautista,
    M.J. (2026). "VizARE: An Intermediate Representation to Support the
    Visualization of Association Rules in Data Mining". Future Internet,
    18(7), 374.

y exportarlo al formato intermedio propuesto (JSON Graph Format), ademas de
GraphML/DOT (Gephi/Cytoscape/Graphviz/NetworkX) y a la capa de resumen
opcional (Seccion 4.2) para inspeccionar grandes conjuntos de reglas.
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

    # Cada regla es un nodo propio, conectado a sus items por aristas
    # 'antecedent'/'consequent' -- no hay arista directa item -> item.
    for ruleId in ruleNodes:
        data = graph.graph.nodes[ruleId]
        antecedent = sorted(graph.graph.predecessors(ruleId))
        consequent = sorted(graph.graph.successors(ruleId))
        print("  {}: {} -> {}  support={:.2f} confidence={:.2f} lift={:.2f}".format(
            ruleId, antecedent, consequent, data["support"], data["confidence"], data["lift"]))

    print("\nJGF (formato intermedio propuesto, primeras lineas):")
    print("\n".join(graph.to_jgf().splitlines()[:12]))

    # Capa de resumen (Seccion 4.2): agrupa las reglas con el mismo consecuente.
    summaryIds = graph.summarize(signature="consequent", measures=("confidence", "lift"))
    print("\nResumen por consecuente ({} grupos):".format(len(summaryIds)))
    for summaryId in summaryIds:
        data = graph.graph.nodes[summaryId]
        print("  {} reglas -> {}  confidence en [{:.2f}, {:.2f}] (media {:.2f})".format(
            data["rule_count"], sorted(graph.graph.successors(summaryId)),
            data["confidence_min"], data["confidence_max"], data["confidence_mean"]))

    return graph


if __name__ == "__main__":
    main()
