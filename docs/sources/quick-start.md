# Quick Start

All the snippets on this page are runnable as-is (they are also covered by the test suite under
`tests/`). See [Installation](installation.md) first.

## 1. Crisp association rules

`ARM.association_rules` takes a `pandas.DataFrame` of frequent itemsets (columns `support` and
`itemsets`, as produced by `FFIM.fpgrowth` -- see below) and returns the rules whose `metric`
reaches `min_threshold`. Besides the usual `confidence`/`lift`/`leverage`/`conviction`, it also
supports `certainty_factor` (Shortliffe & Buchanan's CF), which unlike confidence takes the base
rate of the consequent into account and is bounded in `[-1, 1]`:

```python
import pandas as pd
from ARMxtend.ARM import association_rules

df = pd.DataFrame({
    "support": [0.5, 0.8, 0.7, 0.3],
    "itemsets": [frozenset(["bread"]), frozenset(["milk"]), frozenset(["butter"]),
                frozenset(["bread", "milk"])],
})
rules = association_rules(df, metric="confidence", min_threshold=0.5)
print(rules[["antecedents", "consequents", "support", "confidence", "certainty_factor"]])
#   antecedents consequents  support  confidence  certainty_factor
# 0     (bread)      (milk)      0.3         0.6             -0.25
```

Note the negative certainty factor: although "bread -> milk" clears the 0.5 confidence bar, milk is
already present in 80% of transactions on its own, so bread's presence actually *decreases* the
belief in milk relative to its base rate -- a case confidence alone does not reveal.

## 2. Frequent itemsets with FP-Growth

`FFIM.fpgrowth` mines frequent itemsets directly from raw transactions (no candidate generation),
returning a `dict {itemset_key: support}` ready to feed into `association_rules` after converting
it to the expected DataFrame shape (`FIM._shared.key_to_itemset` undoes the `'A-B-C'` key encoding):

```python
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM._shared import key_to_itemset
from ARMxtend.ARM import association_rules
import pandas as pd

transactions = [
    ["bread", "milk"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "cola"],
]
freq_itemsets = fpgrowth(transactions, min_supp=0.5)

df = pd.DataFrame({"support": list(freq_itemsets.values()),
                   "itemsets": [key_to_itemset(k) for k in freq_itemsets]})
rules = association_rules(df, metric="confidence", min_threshold=0.7)
```

## 3. Fuzzy association rules

Fuzzy itemsets and rules generalize the crisp case to items with a *degree* of membership in
`[0, 1]` (e.g. "cold" or "low humidity" instead of a hard threshold). `FFIM.fuzzy_fpgrowth` mines
fuzzy frequent itemsets by decomposing the fuzzy database into `num_alpha` alpha-cuts (10 is a good
default, see [Fernandez-Basso, Ruiz & Martin-Bautista, 2021](cite.md)); `FIM.FARE.fuzzy_association_rules`
then derives the fuzzy support (FSupp), confidence (FConf) and certainty factor (FCF), integrating
the crisp measure at each alpha-cut weighted by the width of that cut (see the
[FARE user guide](user_guide/FIM/FARE.md) for the exact formulas):

```python
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules

# each transaction: a list of (item, membership degree in [0, 1]) pairs
fuzzy_transactions = [
    [("cold", 1.0), ("low_humidity", 0.8)],
    [("cold", 0.9), ("low_humidity", 0.6)],
    [("warm", 0.7), ("low_humidity", 0.9)],
    [("cold", 0.6), ("low_humidity", 0.7)],
    [("warm", 1.0), ("low_humidity", 0.2)],
]
freq_itemsets = fuzzy_fpgrowth(fuzzy_transactions, min_supp=0.3, num_alpha=10)
rules = fuzzy_association_rules(freq_itemsets, num_alpha=10, metric="confidence", min_threshold=0.5)
print(rules[["antecedents", "consequents", "support", "confidence", "certainty_factor"]])
```

`FIM.Eclat.FuzzyDECLAT` and `FIM.BD_FARE.FuzzyDAprioriTID` mine the same kind of fuzzy frequent
itemsets, but distributed on Spark (see section 5 below).

## 4. Meta-association rules

When you have already mined association rules independently from *several* datasets (e.g. one per
store, sensor, or time period), `ARM.meta_rules` finds "rules about rules": which primary rules
tend to co-occur across a high enough proportion of the datasets, optionally together with extra
attributes of each dataset. This is useful both to summarize a large number of per-dataset rules and
to spot cross-dataset patterns invisible in any single dataset (see
[Ruiz et al., 2016](cite.md)).

```python
from ARMxtend.ARM.meta_rules import mine_primary_rule_measures, crisp_meta_association_rules, \
    fuzzy_meta_association_rules

store1 = [["bread", "milk"]] * 6 + [["bread"]] * 2 + [["milk"]] * 2
store2 = [["bread", "milk"]] * 5 + [["bread"]] * 3 + [["milk"]] * 2
store3 = [["cola", "chips"]] * 7 + [["cola"]] * 3

# Step 1: mine primary crisp rules (and their confidence) independently per dataset
rule_measures = mine_primary_rule_measures([store1, store2, store3], min_supp=0.3, min_conf=0.5)
# -> [{'milk=>bread': 0.75, 'bread=>milk': 0.75}, {'milk=>bread': 0.71, 'bread=>milk': 0.62}, {'chips=>cola': 1.0, 'cola=>chips': 0.7}]

# Step 2a: crisp meta-rules -- only presence/absence of each primary rule matters
crisp_meta = crisp_meta_association_rules([set(m) for m in rule_measures], min_supp=0.5, min_conf=0.9)
print(crisp_meta[["antecedents", "consequents", "support", "confidence"]])
# bread=>milk <-> milk=>bread, support=0.67 (found in 2/3 datasets); chips<->cola rules are
# dropped, since they were only found in 1/3 datasets (below min_supp)

# Step 2b: fuzzy meta-rules -- also weighs how strong (confident) each primary rule was
fuzzy_meta = fuzzy_meta_association_rules(rule_measures, num_alpha=10, min_supp=0.3, min_conf=0.5)
print(fuzzy_meta[["antecedents", "consequents", "support", "confidence"]])
```

## 5. Big Data (Spark)

`FIM.apriori`, `FIM.Eclat` and `FIM.BD_ARE`/`FIM.BD_FARE` mirror the algorithms above but distribute
the counting across a Spark cluster (`pip install -e ".[spark]"`, see [Installation](installation.md)).
They take a `pyspark.SparkContext` and an `RDD` of transactions instead of a plain Python list:

```python
from pyspark import SparkContext, SparkConf
from ARMxtend.FIM.apriori import DApriori
from ARMxtend.FIM.BD_ARE import association_rules_bd

sc = SparkContext(conf=SparkConf().setAppName("armxtend-quickstart").setMaster("local[*]"))
transactions = sc.parallelize([
    "bread,milk",
    "bread,diapers,beer,eggs",
    "milk,diapers,beer,cola",
    "bread,milk,diapers,beer",
    "bread,milk,diapers,cola",
])

freq_itemsets = DApriori.run(sc, transactions, min_supp=0.5)   # or DAprioriTID, or Eclat.DECLAT
rules = association_rules_bd(sc, freq_itemsets, min_conf=0.7)
```

`SFIM` mines frequent itemsets over a Spark Streaming sliding window (see `SFIM.main`), and
`SARE.extractAssociationRules` extracts association rules from its current frequent-itemset tree.

## 6. Visualizing rules

`VizARM.AREtoGraph` turns a rules DataFrame into a directed graph (GraphML or DOT), ready to open in
Gephi, Cytoscape or Graphviz:

```python
from ARMxtend.VizARM import AREtoGraph

graph = AREtoGraph.from_dataframe(rules, rule_measures=("confidence", "certainty_factor"))
dot = graph.exportGraph(type=1)   # 0: GraphML, 1: DOT, 2: the networkx.DiGraph itself
```

## 7. Fuzzifying numeric attributes

`preprocessing.FuzzyLib` turns a numeric column into a set of overlapping fuzzy labels (a
triangular Ruspini partition), ready to be used as items in the fuzzy pipelines above:

```python
import pandas as pd
from ARMxtend.preprocessing import FuzzyLib

fuzzy_lib = FuzzyLib()
fuzzy_lib.data = pd.DataFrame({"temperature": [15, 19, 22, 25, 28]})
added_columns = fuzzy_lib.Fuzzification(["temperature"], [[18, 22, 26]], [["cold", "comfort", "warm"]])
# adds temperature_cold, temperature_comfort, temperature_warm columns, degrees summing to 1 per row
```
