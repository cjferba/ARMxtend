# `VizARM.AREtoGraph`

Exports a set of association rules as a directed graph -- one node per item, one edge per rule,
with the rule's assessment measures (confidence, lift, certainty factor, ...) as edge attributes --
in a standard format that can be opened directly in [Gephi](https://gephi.org/),
[Cytoscape](https://cytoscape.org/) or [Graphviz](https://graphviz.org/).

## Building the graph

```python
AREtoGraph.from_dataframe(rules_df, rule_measures=("confidence", "lift"))
```

The most convenient entry point: builds the graph directly from a rules `pandas.DataFrame`, in the
same format returned by [`ARM.association_rules`](ARM/association_rules.md) or
[`FIM.FARE.fuzzy_association_rules`](FIM/FARE.md) (columns `antecedents`/`consequents` as
`frozenset`s of items, plus one column per measure named in `rule_measures`).

```python
from ARMxtend.ARM import association_rules
from ARMxtend.VizARM import AREtoGraph

rules = association_rules(freq_itemsets_df, metric="confidence", min_threshold=0.5)
graph = AREtoGraph.from_dataframe(rules, rule_measures=("confidence", "certainty_factor"))
```

Alternatively, `AREtoGraph(path, MeasuresRules, MeasuresItems=None, SepRule=";", SepFI=";",
SepItems=",")` loads rules from a CSV file with `antecedents`/`consequents` columns (items joined
by `SepItems`) plus one column per measure in `MeasuresRules`; `load_item_measures(path)` attaches
per-item measures (e.g. support) from a second CSV.

## Exporting

```python
graph.exportGraph(type=0)   # GraphML string (Gephi / Cytoscape)
graph.exportGraph(type=1)   # DOT string (Graphviz)
graph.exportGraph(type=2)   # the underlying networkx.DiGraph
```

```python
dot = graph.exportGraph(type=1)
print(dot)
# digraph ARM {
#     "bread";
#     "milk";
#     "bread" -> "milk" [confidence="0.6", certainty_factor="-0.25"];
# }
```

Write the result to a `.graphml`/`.dot` file to open it in your graph visualization tool of choice,
or work with the `networkx.DiGraph` (`type=2`) directly for further analysis (centrality, community
detection, etc.) using [networkx](https://networkx.org/).
