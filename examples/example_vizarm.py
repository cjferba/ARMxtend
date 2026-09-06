# -*- coding: utf-8 -*-
"""
Ejemplo: exportar un conjunto de reglas de asociacion (ARM.association_rules)
a un grafo dirigido (VizARM.AREtoGraph), en formato GraphML y DOT, listo
para abrir en Gephi/Cytoscape/Graphviz.
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

    graph = AREtoGraph.from_dataframe(rulesDf, rule_measures=("confidence", "lift"))
    print("Grafo: {} nodos (items), {} aristas (reglas)".format(
        graph.graph.number_of_nodes(), graph.graph.number_of_edges()))

    graphml = graph.exportGraph(type=0)
    print("\nGraphML (primeras lineas):")
    print("\n".join(graphml.splitlines()[:8]))

    dot = graph.exportGraph(type=1)
    print("\nDOT:")
    print(dot)

    return graph


if __name__ == "__main__":
    main()
