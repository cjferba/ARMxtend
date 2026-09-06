# Examples

Every example on this page lives as a runnable, self-contained script under
[`examples/`](https://github.com/cjferba/ARMxtend/tree/master/examples) in the repository, and is executed on every
push by [`tests/test_examples.py`](https://github.com/cjferba/ARMxtend/blob/master/tests/test_examples.py) as part of
the [CI workflow](https://github.com/cjferba/ARMxtend/actions) (see the badge on the project [README](https://github.com/cjferba/ARMxtend#readme)).
This means the code and the output shown below are guaranteed to stay in sync with the current
version of the library -- if a change to `ARMxtend` breaks one of these examples, CI turns red.

Run any of them locally with:

```bash
pip install -e .
python examples/example_association_rules.py
```

## Crisp association rules (`ARM.association_rules` + `FFIM.fpgrowth`)

Mines frequent itemsets with FP-Growth and then crisp association rules (support/confidence/lift),
on the classic 10-transaction, 4-item dataset used throughout the FIM papers cited in
[Citing ARMxtend](cite.md).

```python
import pandas as pd

from ARMxtend.ARM.association_rules import association_rules
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM._shared import key_to_itemset

TRANSACTIONS = [
    "A,B,C", "B,D", "A,C,D", "B,C", "A,C",
    "B,D", "A,B,C", "B,C,D", "A,B,D", "B,C,D",
]


def main():
    transactions = [line.split(",") for line in TRANSACTIONS]

    freqItemsets = fpgrowth(transactions, min_supp=0.3)
    print("Itemsets frecuentes (min_supp=0.3):")
    for itemsetKey, support in sorted(freqItemsets.items()):
        print("  {:<10} support={:.2f}".format(itemsetKey, support))

    itemsetsDf = pd.DataFrame({
        "support": list(freqItemsets.values()),
        "itemsets": [key_to_itemset(key) for key in freqItemsets],
    })
    rulesDf = association_rules(itemsetsDf, metric="confidence", min_threshold=0.7)

    print("\nReglas de asociacion (min_confidence=0.7):")
    for _, rule in rulesDf.iterrows():
        antecedent = ",".join(sorted(rule["antecedents"]))
        consequent = ",".join(sorted(rule["consequents"]))
        print("  {} -> {:<6} support={:.2f} confidence={:.2f} lift={:.2f}".format(
            antecedent, consequent, rule["support"], rule["confidence"], rule["lift"]))

    return freqItemsets, rulesDf


if __name__ == "__main__":
    main()
```

(full source: [`examples/example_association_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_association_rules.py))

Output:

```text
Itemsets frecuentes (min_supp=0.3):
  A          support=0.50
  A-B        support=0.30
  A-C        support=0.40
  B          support=0.80
  B-C        support=0.50
  B-D        support=0.50
  C          support=0.70
  C-D        support=0.30
  D          support=0.60

Reglas de asociacion (min_confidence=0.7):
  A -> C      support=0.40 confidence=0.80 lift=1.14
  D -> B      support=0.50 confidence=0.83 lift=1.04
  C -> B      support=0.50 confidence=0.71 lift=0.89
```

## Fuzzy association rules (`FFIM.fuzzy_fpgrowth` + `FIM.FARE`)

Reproduces, alpha-cut by alpha-cut, the worked example of Delgado, Ruiz, Sanchez & Serrano (2011)
-- see [`FIM.FARE`](user_guide/FIM/FARE.md) -- for the rule `{i1,i3} -> {i4}`: FSupp = 0.266,
FConf = 0.5, FCF = 0.33. This is exactly what [`tests/test_examples.py`](https://github.com/cjferba/ARMxtend/blob/master/tests/test_examples.py)
asserts, so any regression in the alpha-cut aggregation breaks CI immediately.

```python
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules

# t1..t6, grados de pertenencia de i1..i5 (Tabla 2 del articulo)
FUZZY_TRANSACTIONS = [
    [("i1", 1.0), ("i2", 0.2), ("i3", 1.0), ("i4", 0.8), ("i5", 0.9)],
    [("i1", 1.0), ("i2", 1.0), ("i3", 0.8)],
    [("i1", 0.4), ("i2", 0.1), ("i3", 0.7), ("i4", 0.6)],
    [("i1", 0.6), ("i3", 0.4), ("i4", 0.4), ("i5", 0.5)],
    [("i1", 0.4), ("i2", 0.1)],
    [("i2", 1.0)],
]

NUM_ALPHA = 5  # alpha-cortes {1, 0.8, 0.6, 0.4, 0.2}, como en el articulo


def main():
    freqItemsets = fuzzy_fpgrowth(FUZZY_TRANSACTIONS, min_supp=0.1, num_alpha=NUM_ALPHA)
    rulesDf = fuzzy_association_rules(freqItemsets, NUM_ALPHA, metric="support", min_threshold=0.0)

    target = next(
        row for _, row in rulesDf.iterrows()
        if set(row["antecedents"]) == {"i1", "i3"} and set(row["consequents"]) == {"i4"}
    )
    print("Comprobacion contra el articulo para {i1,i3} -> {i4}:")
    print("  esperado  FSupp=0.266 FConf=0.5 FCF=0.33")
    print("  obtenido  FSupp={:.3f} FConf={:.3f} FCF={:.3f}".format(
        target["support"], target["confidence"], target["certainty_factor"]))

    return freqItemsets, rulesDf


if __name__ == "__main__":
    main()
```

(full source, including every printed itemset/rule: [`examples/example_fuzzy_association_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_fuzzy_association_rules.py))

Output (abridged -- the full run lists every fuzzy rule found):

```text
Comprobacion contra el articulo para {i1,i3} -> {i4}:
  esperado  FSupp=0.266 FConf=0.5 FCF=0.33
  obtenido  FSupp=0.267 FConf=0.500 FCF=0.330
```

## Meta-association rules (`ARM.meta_rules`)

Mines primary rules independently in three small datasets, then meta-rules describing which
primary rules co-occur across datasets -- crisp (presence/absence) and fuzzy (weighted by each
rule's own support/confidence). See [`ARM.meta_rules`](user_guide/ARM/meta_rules.md) for the
underlying model (Ruiz et al., 2016).

```python
from ARMxtend.ARM.meta_rules import (
    mine_primary_rule_measures,
    crisp_meta_association_rules,
    fuzzy_meta_association_rules,
)

# Tres datasets D1, D2, D3 en los que A -> B es una regla fuerte y
# consistente (aparece siempre que aparece A), y C -> D solo aparece,
# debilmente, en D3.
DATASETS = [
    [["A", "B"], ["A", "B"], ["A", "B"], ["C"]],
    [["A", "B"], ["A", "B"], ["B"], ["C"]],
    [["A", "B"], ["A", "B", "C", "D"], ["C", "D"], ["B"]],
]


def main():
    ruleMeasuresPerDataset = mine_primary_rule_measures(DATASETS, min_supp=0.4, min_conf=0.5)

    crispMetaRules = crisp_meta_association_rules(
        ruleMeasuresPerDataset, min_supp=0.6, min_conf=0.6)

    fuzzyMetaRules = fuzzy_meta_association_rules(
        ruleMeasuresPerDataset, num_alpha=5, min_supp=0.3, min_conf=0.5)

    return ruleMeasuresPerDataset, crispMetaRules, fuzzyMetaRules


if __name__ == "__main__":
    main()
```

(full source, with the printing of every rule set: [`examples/example_meta_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_meta_rules.py))

Output:

```text
Reglas primarias minadas en cada dataset:
  D1: {'A=>B': 1.0, 'B=>A': 1.0}
  D2: {'A=>B': 1.0, 'B=>A': 0.6666666666666666}
  D3: {'C=>D': 1.0, 'D=>C': 1.0, 'A=>B': 1.0, 'B=>A': 0.6666666666666666}

Meta-reglas crisp (co-ocurrencia de reglas entre datasets):
  {'A=>B'} -> {'B=>A'}  support=1.00 confidence=1.00
  {'B=>A'} -> {'A=>B'}  support=1.00 confidence=1.00

Meta-reglas difusas (ponderadas por la fuerza de cada regla primaria):
  {'A=>B'} -> {'B=>A'}  FSupp=0.73 FConf=0.73
  {'B=>A'} -> {'A=>B'}  FSupp=0.73 FConf=1.00
  {'C=>D'} -> {'D=>C'}  FSupp=0.33 FConf=1.00
  {'D=>C'} -> {'C=>D'}  FSupp=0.33 FConf=1.00
  ...
```

Note how the crisp variant only sees `A=>B` and `B=>A` co-occurring in all three datasets (a
boolean fact), while the fuzzy variant additionally distinguishes that the `A=>B` <-> `B=>A`
association (FSupp=0.73) is far more prominent than the `C=>D` <-> `D=>C` one (FSupp=0.33, since
`C=>D`/`D=>C` were only mined at all in dataset D3).

## Preprocessing: fuzzifying a numeric attribute (`preprocessing.FuzzyLib`)

Turns a numeric `temperature` column into a Ruspini triangular fuzzy partition (`cold` /
`comfortable` / `warm`), the usual preprocessing step before feeding data into any of the fuzzy
mining algorithms above. See [`preprocessing.FuzzyLib`](user_guide/preprocessing/FuzzyLib.md).

```python
import pandas as pd

from ARMxtend.preprocessing import FuzzyLib


def main():
    fuzzyLib = FuzzyLib()
    fuzzyLib.data = pd.DataFrame({
        "temperature": [17.0, 19.5, 21.0, 23.0, 25.5, 28.0],
        "sensor_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
    })
    fuzzyLib.atributes = list(fuzzyLib.data.columns)

    addedColumns = fuzzyLib.Fuzzification(
        Atri=["temperature"],
        thresholds=[[18, 21, 25]],
        FuzzyLabel=[["cold", "comfortable", "warm"]],
    )

    print("Columnas difusas anadidas:", addedColumns)
    print(fuzzyLib.GetData().to_string(index=False))
    return fuzzyLib.GetData()


if __name__ == "__main__":
    main()
```

(full source: [`examples/example_preprocessing.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_preprocessing.py))

Output:

```text
Columnas difusas anadidas: ['temperature_cold', 'temperature_comfortable', 'temperature_warm']

 temperature sensor_id  temperature_cold  temperature_comfortable  temperature_warm
        17.0        s1               1.0                      0.0               0.0
        19.5        s2               0.5                      0.5               0.0
        21.0        s3               0.0                      1.0               0.0
        23.0        s4               0.0                      0.5               0.5
        25.5        s5               0.0                      0.0               1.0
        28.0        s6               0.0                      0.0               1.0
```

## Visualization: the VizARE typed graph (`VizARM.AREtoGraph`)

Takes the crisp rules mined in the first example and transforms them into the **VizARE** typed
graph (item nodes + rule nodes, connected by `antecedent`/`consequent` edges) of Fernandez-Basso,
Ruiz, Molina-Solana & Martin-Bautista (2026), exports it to the paper's proposed intermediate
format (JSON Graph Format), and builds the optional rule-summarization layer (Section 4.2). See
[`VizARM.AREtoGraph`](user_guide/VizARM.md).

```python
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
    # Each rule is its own node -- e.g. rule::A=>C -- linked to item nodes A
    # (antecedent edge) and C (consequent edge); there is no direct A -> C edge.
    print(graph.to_jgf())

    summaryIds = graph.summarize(signature="consequent", measures=("confidence", "lift"))
    return graph, summaryIds


if __name__ == "__main__":
    main()
```

(full source: [`examples/example_vizarm.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_vizarm.py))

Output (abridged):

```text
Grafo: 4 nodos item, 3 nodos rule, 6 aristas
  rule::A=>C: ['A'] -> ['C']  support=0.40 confidence=0.80 lift=1.14
  rule::D=>B: ['D'] -> ['B']  support=0.50 confidence=0.83 lift=1.04
  rule::C=>B: ['C'] -> ['B']  support=0.50 confidence=0.71 lift=0.89

JGF (formato intermedio propuesto, primeras lineas):
{
  "graph": {
    "directed": true,
    "type": "association-rules",
    "nodes": {
      "rule::A=>C": {
        "label": "A -> C",
        "metadata": {
          "kind": "rule",
          "group": "rule",
          "support": 0.4,
          "confidence": 0.8,

Resumen por consecuente (2 grupos):
  1 reglas -> ['C']  confidence en [0.80, 0.80] (media 0.80)
  2 reglas -> ['B']  confidence en [0.71, 0.83] (media 0.77)
```

GraphML (Gephi/Cytoscape/NetworkX) and DOT (Graphviz) exports are still available as
`graph.exportGraph(type=1)`/`type=2`, for tools that don't consume JGF directly.

## Big Data and streaming algorithms

The Spark-based algorithms (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`, `SARE`,
`SFIM`) are exercised in `tests/test_fim_apriori_eclat.py`, `tests/test_sare_association_rules.py`
and the rest of the `tests/` suite against `tests/spark_stub.py`, a small local double of the Spark
RDD API -- see the [User Guide](USER_GUIDE_INDEX.md) for each algorithm's reference and the
`pip install -e ".[spark]"` note in the project [README](https://github.com/cjferba/ARMxtend#readme)
for running them against a real cluster.
