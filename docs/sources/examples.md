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
    print("Frequent itemsets (min_supp=0.3):")
    for itemsetKey, support in sorted(freqItemsets.items()):
        print("  {:<10} support={:.2f}".format(itemsetKey, support))

    itemsetsDf = pd.DataFrame({
        "support": list(freqItemsets.values()),
        "itemsets": [key_to_itemset(key) for key in freqItemsets],
    })
    rulesDf = association_rules(itemsetsDf, metric="confidence", min_threshold=0.7)

    print("\nAssociation rules (min_confidence=0.7):")
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
Frequent itemsets (min_supp=0.3):
  A          support=0.50
  A-B        support=0.30
  A-C        support=0.40
  B          support=0.80
  B-C        support=0.50
  B-D        support=0.50
  C          support=0.70
  C-D        support=0.30
  D          support=0.60

Association rules (min_confidence=0.7):
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

# t1..t6, membership degrees of i1..i5 (Table 2 of the paper)
FUZZY_TRANSACTIONS = [
    [("i1", 1.0), ("i2", 0.2), ("i3", 1.0), ("i4", 0.8), ("i5", 0.9)],
    [("i1", 1.0), ("i2", 1.0), ("i3", 0.8)],
    [("i1", 0.4), ("i2", 0.1), ("i3", 0.7), ("i4", 0.6)],
    [("i1", 0.6), ("i3", 0.4), ("i4", 0.4), ("i5", 0.5)],
    [("i1", 0.4), ("i2", 0.1)],
    [("i2", 1.0)],
]

NUM_ALPHA = 5  # alpha-cuts {1, 0.8, 0.6, 0.4, 0.2}, as in the paper


def main():
    freqItemsets = fuzzy_fpgrowth(FUZZY_TRANSACTIONS, min_supp=0.1, num_alpha=NUM_ALPHA)
    rulesDf = fuzzy_association_rules(freqItemsets, NUM_ALPHA, metric="support", min_threshold=0.0)

    target = next(
        row for _, row in rulesDf.iterrows()
        if set(row["antecedents"]) == {"i1", "i3"} and set(row["consequents"]) == {"i4"}
    )
    print("Check against the paper for {i1,i3} -> {i4}:")
    print("  expected  FSupp=0.266 FConf=0.5 FCF=0.33")
    print("  obtenido  FSupp={:.3f} FConf={:.3f} FCF={:.3f}".format(
        target["support"], target["confidence"], target["certainty_factor"]))

    return freqItemsets, rulesDf


if __name__ == "__main__":
    main()
```

(full source, including every printed itemset/rule: [`examples/example_fuzzy_association_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_fuzzy_association_rules.py))

Output (abridged -- the full run lists every fuzzy rule found):

```text
Check against the paper for {i1,i3} -> {i4}:
  expected  FSupp=0.266 FConf=0.5 FCF=0.33
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

# Three datasets D1, D2, D3 in which A -> B is a strong and
# consistent rule (it appears whenever A appears), and C -> D only appears,
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
Primary rules mined in each dataset:
  D1: {'A=>B': 1.0, 'B=>A': 1.0}
  D2: {'A=>B': 1.0, 'B=>A': 0.6666666666666666}
  D3: {'C=>D': 1.0, 'D=>C': 1.0, 'A=>B': 1.0, 'B=>A': 0.6666666666666666}

Crisp meta-rules (co-occurrence of rules between datasets):
  {'A=>B'} -> {'B=>A'}  support=1.00 confidence=1.00
  {'B=>A'} -> {'A=>B'}  support=1.00 confidence=1.00

Fuzzy meta-rules (weighted by the strength of each primary rule):
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

    print("Fuzzy columns added:", addedColumns)
    print(fuzzyLib.GetData().to_string(index=False))
    return fuzzyLib.GetData()


if __name__ == "__main__":
    main()
```

(full source: [`examples/example_preprocessing.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_preprocessing.py))

Output:

```text
Fuzzy columns added: ['temperature_cold', 'temperature_comfortable', 'temperature_warm']

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
    graph.to_html("rules.html")    # interactive viewer, self-contained
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

Summary by consequent (2 groups):
  1 rules -> ['C']  confidence in [0.80, 0.80] (mean 0.80)
  2 rules -> ['B']  confidence in [0.71, 0.83] (mean 0.77)

Visor HTML interactivo: 25950 bytes, grafo JGF embebido
```

`graph.to_html("rules.html")` writes that interactive viewer to disk -- a single file you can open in any
browser (see [`VizARM.AREtoGraph`](user_guide/VizARM.md#interactive-viewer)).

GraphML (Gephi/Cytoscape/NetworkX) and DOT (Graphviz) exports are still available as
`graph.exportGraph(type=1)`/`type=2`, for tools that don't consume JGF directly.

## Other kinds of rules and tools

Rules with **absent items**, **exception** and **anomalous** rules, **gradual** rules, **closed /
maximal** itemsets with **redundant-rule pruning**, and **automatic fuzzification** feeding the fuzzy
miners with extra interest measures. See [`ARM.negative_rules`](user_guide/ARM/negative_rules.md),
[`ARM.exception_rules`](user_guide/ARM/exception_rules.md),
[`ARM.gradual_rules`](user_guide/ARM/gradual_rules.md),
[`FIM.condensed`](user_guide/FIM/condensed.md), [`ARM.redundancy`](user_guide/ARM/redundancy.md) and
[`preprocessing.FuzzyLib`](user_guide/preprocessing/FuzzyLib.md).

```python
from ARMxtend.ARM import (
    anomalous_rules, exception_rules, gradual_rules, negative_association_rules, prune_redundant_rules,
)
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM import closed_itemsets, maximal_itemsets
from ARMxtend.FIM.FARE import fuzzy_association_rules
from ARMxtend.preprocessing import FuzzyLib

# quien toma cafe no toma te
negative = negative_association_rules(drinks, min_supp=0.25, min_threshold=0.95)

# "fever -> rest" fails when there is also "must_work" (exception) or ends in "hospital" (anomaly)
exceptions = exception_rules(sickness, min_supp=0.3, min_conf=0.8, min_exception_supp=0.03)
anomalies = anomalous_rules(sickness, min_supp=0.3, min_conf=0.7, min_anomalous_supp=0.05,
                            min_anomalous_conf=0.6)

# the more age and the less exercise, the more cholesterol -- directly on numeric columns
gradual = gradual_rules(health, min_supp=0.25, min_conf=0.8)

# representaciones condensadas y poda
closed, maximal = closed_itemsets(frequent), maximal_itemsets(frequent)
pruned = prune_redundant_rules(all_rules)

# automatic fuzzification (quantiles) + fuzzy rules with additional measures
fuzzy = FuzzyLib()
fuzzy.data = health[["age", "cholesterol"]].copy()
fuzzy.AutoFuzzification(n_labels=3, method="quantile")
itemsets = fuzzy_fpgrowth(fuzzy.GetFuzzyTransactions(), min_supp=0.1, num_alpha=10)
rules = fuzzy_association_rules(itemsets, 10, metric="zhangs_metric", min_threshold=0.4,
                                measures=["zhangs_metric", "kulczynski"])
```

(full source, with the data of each case: [`examples/example_rule_types.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_rule_types.py))

Output:

```text
Rules with absent items:
  {~coffee, ~sugar} -> {tea}  support=0.36 confidence=1.00
  {tea, ~sugar} -> {~coffee}  support=0.36 confidence=1.00
  {~coffee} -> {tea}  support=0.43 confidence=1.00
  {tea} -> {~coffee}  support=0.43 confidence=1.00
  {sugar, ~tea} -> {coffee}  support=0.43 confidence=1.00
  {coffee, sugar} -> {~tea}  support=0.43 confidence=1.00
  {~tea} -> {coffee}  support=0.57 confidence=1.00
  {coffee} -> {~tea}  support=0.57 confidence=1.00

Exceptions to strong common rules:
  {must_work} -> rest (conf 0.88), but with {fever} ~rest occurs: confidence=1.00 (reference 0.20)
  {fever} -> rest (conf 0.80), but with {must_work} ~rest occurs: confidence=1.00 (reference 0.12)
Anomalias (A y no b -> C):
  {fever} -> rest: in 15 cases with ~rest, {hospital} appears (confidence=0.67)

Gradual rules:
  cuanto age↑, exercise↓ -> cholesterol↑  support=0.51 confidence=0.98
  cuanto cholesterol↑, exercise↑ -> age↑  support=0.29 confidence=0.97
  cuanto age↑ -> cholesterol↑  support=0.80 confidence=0.80

Frequent itemsets: 8, closed: ['a-b', 'a-b-c', 'd'], maximal: ['a-b-c', 'd']
Rules: 12 -> 5 after removing the redundant ones

Fuzzy rules with Zhang >= 0.4:
  {age_low} -> {cholesterol_low}  FSupp=0.12 FConf=0.37 zhang=0.74 kulczynski=0.55
  {cholesterol_low} -> {age_low}  FSupp=0.12 FConf=0.73 zhang=0.63 kulczynski=0.55
  {age_high} -> {cholesterol_high}  FSupp=0.12 FConf=0.48 zhang=0.49 kulczynski=0.51
```

## Time series and sensors: lagged rules and rule drift (`ARM.temporal_rules`, `ARM.rule_drift`)

Sixty days of hourly readings of a simulated house -- outdoor temperature, door and heater. The heater comes on one
hour after the door is opened, but only from 17 h on; in the second half of the period it also comes on after a
cold hour. The script turns the series into items ([`preprocessing.timeseries`](user_guide/preprocessing/timeseries.md)),
mines lagged rules ([`ARM.temporal_rules`](user_guide/ARM/temporal_rules.md)), measures the same rule hour by hour
and follows the rules over 15-day windows ([`ARM.rule_drift`](user_guide/ARM/rule_drift.md)).

```python
from ARMxtend.ARM import (
    compare_rule_sets, prune_redundant_rules, rule_drift, rule_profile, temporal_association_rules, window_rules,
)
from ARMxtend.preprocessing.timeseries import calendar_items, discretize, to_items

# temperature -> low/medium/high; door and heating are events; + time of day
items = to_items(discretize(house, ["temp"], n_symbols=3, method="quantile"))
with_calendar = items.join(calendar_items(items.index, ["part_of_day"]))

# rules with lag: what happened an hour ago for the heating to be on now
rules = temporal_association_rules(with_calendar, max_lag=1, min_supp=0.02, min_threshold=0.9,
                                   targets=["heater_on"], exclude_persistence=True)
rules = prune_redundant_rules(rules)

# the same rule, hour by hour: it only holds from 17 h on
profile = rule_profile(items, ["door_open@lag1"], ["heater_on"], lambda stamps: stamps.hour)

# drift: rules with lag in windows of 15 days; which ones emerge, stay or vanish
windows = window_rules(with_calendar, "15D", miner=lambda w: temporal_association_rules(
    w, max_lag=1, min_supp=0.03, min_threshold=0.85, targets=["heater_on"], exclude_persistence=True))
drift = rule_drift(windows)
```

(full source, with the simulated data: [`examples/example_temporal_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_temporal_rules.py))

Output:

```text
Items: temp=high, temp=low, temp=medium, door_open, heater_on

Rules with lag (consequent: heater_on):
  door_open (t-1), part_of_day=evening (t), temp=high (t-1) -> heater_on (t)  support=0.022 confidence=1.00 lift=4.4
  door_open (t-1), part_of_day=evening (t) -> heater_on (t)  support=0.051 confidence=0.95 lift=4.1
  door_open (t-1), part_of_day=evening (t), part_of_day=evening (t-1) -> heater_on (t)  support=0.043 confidence=0.94 lift=4.1
  door_open (t-1), part_of_day=evening (t), temp=medium (t) -> heater_on (t)  support=0.030 confidence=0.93 lift=4.1

Confidence of 'door open 1 hour ago -> heating on' by hour:
  04h: 0.55
  10h: 0.00
  16h: 0.00
  17h: 1.00
  20h: 0.86
  23h: 0.92

Windows: 4 | rule states: {'emerging': 52, 'vanishing': 6, 'intermittent': 3, 'stable': 2}
  emerging: temp=low (t-1) -> heater_on (t) (since window 2, mean confidence 0.99)
First versus last window: {'new': 52, 'lost': 6, 'changed': 2}
```

## Event logs: sequential patterns and rules (`FIM.sequential`, `ARM.sequential_rules`)

The events of a person in a smart home over 120 days (alarm, bathroom, kitchen, coffee, TV, door and random phone
calls). Rushed days (most workdays) end with the door right after the kitchen; relaxed days have coffee and then TV.
The script builds one sequence per day from the event log, mines the routines with PrefixSpan
([`FIM.sequential`](user_guide/FIM/sequential.md)) and the "if X, then later Y" rules
([`ARM.sequential_rules`](user_guide/ARM/sequential_rules.md)); the lift separates the informative rule
(`coffee => tv`) from the trivial ones that hold every day.

```python
from ARMxtend.ARM import sequential_rules
from ARMxtend.FIM import format_sequence, prefixspan, sequences_from_events

sequences = sequences_from_events(events, "day", "time", "event")      # one sequence per day

patterns = prefixspan(sequences, min_supp=0.3, min_len=4)             # frequent routines
every = sequential_rules(sequences, min_supp=0.2, min_conf=0.0, max_antecedent=1, max_consequent=1)
rules = every[every["confidence"] >= 0.7].sort_values("lift", ascending=False)
```

(full source: [`examples/example_sequences.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_sequences.py))

Output:

```text
Sequences: 120 | first day: {alarm} {bathroom} {kitchen} {door} {phone}

Frequent patterns of 4 or more events (support >= 0.3):
  <{alarm} {bathroom} {kitchen} {phone}>  support=0.67
  <{alarm} {bathroom} {kitchen} {door}>  support=0.59
  <{alarm} {bathroom} {door} {phone}>  support=0.41
  <{alarm} {kitchen} {door} {phone}>  support=0.41

Sequential rules with confidence >= 0.7, by decreasing lift:
  coffee => tv  support=0.28 confidence=0.77 lift=2.44
  bathroom => kitchen  support=1.00 confidence=1.00 lift=1.00
  alarm => kitchen  support=1.00 confidence=1.00 lift=1.00
  alarm => bathroom  support=1.00 confidence=1.00 lift=1.00

confidence of coffee => tv: 0.77; of tv => coffee: 0.00
```

## Fuzzy sequential patterns and rules (`FIM.fuzzy_sequential`, `ARM.fuzzy_sequential_rules`)

Eighty days of a house, eight readings of temperature and heating power per day. The colder the morning, the more
power is used in the evening, and the relation is gradual. The readings are fuzzified (`FuzzyLib.AutoFuzzification`),
each day becomes a fuzzy sequence, and [`fuzzy_prefixspan`](user_guide/FIM/fuzzy_sequential.md) and
[`fuzzy_sequential_rules`](user_guide/ARM/fuzzy_sequential_rules.md) mine the patterns and the rules; each day fulfils
the rule `temp_low => power_high` to its own degree, and fixed thresholds are compared with the fuzzy sets.

```python
from ARMxtend.ARM import fuzzy_rule_degree, fuzzy_sequential_rules
from ARMxtend.FIM import fuzzy_prefixspan, fuzzy_sequences_from_memberships
from ARMxtend.preprocessing import FuzzyLib

library = FuzzyLib()
library.data = readings
names = library.AutoFuzzification(Atri=["temp", "power"], n_labels=3, method="uniform")
sequences = fuzzy_sequences_from_memberships(library.data[names], by=library.data["day"], min_degree=0.2)

patterns = fuzzy_prefixspan(sequences, 0.2, max_len=2, min_len=2)                 # <temp_low -> power_high>, ...
rules = fuzzy_sequential_rules(sequences, 0.1, min_conf=0.5, max_antecedent=1, max_consequent=1)
degrees = [fuzzy_rule_degree(day, ["temp_low"], ["power_high"]) for day in sequences]
```

(full source: [`examples/example_fuzzy_sequences.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_fuzzy_sequences.py))

Output:

```text
80 days; fuzzy labels: temp_low, temp_medium, temp_high, power_low, power_medium, power_high

Patterns of two events (fuzzy support, 26 in total):
  <temp_low -> power_high>  support=0.22
  <power_high -> temp_low>  support=< 0.20 (no frecuente)

Most striking fuzzy rules (by lift):
  temp_low => power_high  support=0.22 confidence=0.63 lift=2.76
  temp_medium => power_medium  support=0.68 confidence=0.92 lift=1.25
  temp_low => temp_high  support=0.34 confidence=0.94 lift=1.19

temp_low => power_high per day: degree 0 on 45 mild days, up to 0.93 on the coldest; correlation with the cold intensity 0.86

With fuzzy sets: temp_low => power_high  support=0.22 confidence=0.63 lift=2.76
With threshold temp <  8:      cold => peak           support=0.21 confidence=1.00 lift=1.60
With threshold temp < 10:      cold => peak           support=0.39 confidence=1.00 lift=1.60
With threshold temp < 12:      cold => peak           support=0.60 confidence=0.96 lift=1.54
With threshold temp < 14:      cold => peak           support=0.62 confidence=0.82 lift=1.31
```

## Multilevel rules over a taxonomy (`FIM.taxonomy`, `ARM.generalized_rules`)

A supermarket sells 12 products grouped in categories (`Food/Dairy/milk`, `Drinks/Juice/apple_juice`...). Whoever buys bread
usually buys some dairy product, but the purchases are spread over many products and none reaches a high support on its
own; and the croissant goes with juice much more than the rest of the bread. [Generalized rules](user_guide/ARM/generalized_rules.md)
see the categories, with a lower minimum support at the deeper levels; the R-interesting ones drop the rules that only repeat,
with one product, what the category already says.

```python
from ARMxtend.ARM import generalized_rules, interesting_rules
from ARMxtend.FIM import Taxonomy

taxonomy = Taxonomy.from_paths(["Food/Dairy/milk", "Food/Bakery/croissant", "Drinks/Juice/orange_juice", ...])
rules = generalized_rules(baskets, taxonomy, {1: 0.2, 2: 0.1, 3: 0.04}, min_threshold=0.4)   # support per level
kept = interesting_rules(rules, interest=1.2)                                                  # drop the echoes
```

(full source: [`examples/example_generalized_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_generalized_rules.py))

Output:

```text
2639 baskets; taxonomy: 19 items in 3 levels (Drinks, Food)

Without a taxonomy (support 0.04, confidence 0.4, lift 1.2): 4 rules, all between specific products
With taxonomy (support 0.2 / 0.1 / 0.04 per level): 31 rules, 12 of them between categories (level 1 or 2)

R-interesting (R=1.2): 14 of 31 remain. By lift:
  Dairy, Juice => croissant  support=0.14 confidence=0.63 lift=3.26
  croissant => Dairy, Juice  support=0.14 confidence=0.71 lift=3.26
  Dairy, croissant => Juice  support=0.14 confidence=0.89 lift=2.57
  croissant => Juice  support=0.17 confidence=0.88 lift=2.55
  Juice => croissant  support=0.17 confidence=0.50 lift=2.55
  Dairy, Drinks => Bakery  support=0.27 confidence=0.79 lift=1.38

Discarded for repeating what their ancestors already say, for example:
  orange_juice => croissant  support=0.08 confidence=0.50 lift=2.57
  croissant => orange_juice  support=0.08 confidence=0.43 lift=2.57
  croissant => apple_juice  support=0.09 confidence=0.46 lift=2.54
  apple_juice => croissant  support=0.09 confidence=0.49 lift=2.54
```

## Quantitative rules with merged intervals (`ARM.quantitative_rules`)

A shop knows the age, income and sex of its customers and whether they bought a product. Income is high only between 30 and 45
years of age, and women with a high income almost always buy. [`quantitative_rules`](user_guide/ARM/quantitative_rules.md) splits
the numeric columns into equal-frequency intervals, merges adjacent ones into ranges and recovers the region; the same data
mined with five fixed fuzzy labels per column splits the region over two labels.

```python
from ARMxtend.ARM import IntervalPartition, quantitative_rules

partition = IntervalPartition(n_intervals=10, max_supp=0.4).fit(customers)
rules = quantitative_rules(customers, min_supp=0.05, partition=partition, min_threshold=0.7, max_len=3)
kept = quantitative_rules(customers, min_supp=0.05, partition=partition, min_threshold=0.7, max_len=3, interest=1.3)
```

(full source: [`examples/example_quantitative_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_quantitative_rules.py))

Output:

```text
3000 customers. With partial completeness K=1.5 and support 0.05 the paper asks for 160 intervals per attribute; here 10 and ranges of up to 40 % are enough.
Partition: 68 range items (20 base intervals + 48 ranges of several); age in 10 intervals of ~5.2 years

631 rules (support 0.05, confidence 0.7); R-interesting with R=1.3: 152

The planted region [30, 45), with the highest support with confidence >= 0.85:
  age=[28.5368,44.0395) => income=[44.7127,96.4526]  support=0.28 confidence=0.93 lift=2.33

Who buys (women, by lift):
  income=[66.5424,96.4526] & sex=F => buys=yes  support=0.09 confidence=0.89 lift=4.12
  income=[56.3001,96.4526] & sex=F => buys=yes  support=0.12 confidence=0.83 lift=3.84
  age=[33.6534,44.0395) & sex=F => buys=yes  support=0.08 confidence=0.82 lift=3.80

Fuzzy rules age => income (5 fixed labels), by confidence:
  age_low => income_high  support=0.13 confidence=0.55 lift=1.84
  age_medium => income_high  support=0.12 confidence=0.52 lift=1.64
  age_high => income_low  support=0.12 confidence=0.50 lift=1.26
The region [30, 45) straddles two labels and none reaches the confidence of the merged range.
```

## Top-k, rare and high-utility rules (`ARM.top_k_rules`, `ARM.rare_rules`, `ARM.high_utility_rules`)

A shop with 2000 baskets and the quantity bought of eight products. Three ways of looking beyond a minimum support:
[`top_k_rules`](user_guide/ARM/topk_rules.md) asks for the 5 best rules instead of guessing a threshold (0.3 gives no rules and
0.01 gives 93); [`rare_rules`](user_guide/ARM/rare_rules.md) finds the pairs that are frequent separately and rare together
(diapers and wipes, wine and cheese) without the flood of a low threshold; and
[`high_utility_rules`](user_guide/ARM/utility_rules.md) ranks by profit: wine and cheese are worth more than bread and milk
although they are bought less.

```python
from ARMxtend.ARM import high_utility_rules, rare_rules, top_k_rules
from ARMxtend.FIM import high_utility_itemsets, minimal_rare_itemsets, top_k_itemsets

top = top_k_rules(transactions, 5, min_conf=0.5)
informative = top_k_rules(transactions, 5, min_conf=0.5, min_lift=1.5)
rare = rare_rules(transactions, 0.1, min_conf=0.5, min_lift=2)
useful = high_utility_itemsets(quantities, PROFITS, 0.1)
value_rules = high_utility_rules(quantities, PROFITS, 0.1, min_conf=0.5, min_lift=1.5)
```

(full source: [`examples/example_special_rules.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_special_rules.py))

Output:

```text
2000 baskets, 8 products

The 5 rules with the highest support with confidence >= 0.5 (without fixing a minimum support):
  fruit => bread  support=0.208 confidence=0.51 lift=1.00
  eggs => bread  support=0.186 confidence=0.54 lift=1.07
  cheese => bread  support=0.158 confidence=0.50 lift=0.99
  diapers => wipes  support=0.089 confidence=0.64 lift=4.30
  wipes => diapers  support=0.089 confidence=0.60 lift=4.30
The 5 with the highest support that also have lift >= 1.5 (the popular stops covering up what associates):
  diapers => wipes  support=0.089 confidence=0.64 lift=4.30
  wipes => diapers  support=0.089 confidence=0.60 lift=4.30
  wine => cheese  support=0.081 confidence=0.69 lift=2.18
  diapers, fruit => wipes  support=0.043 confidence=0.67 lift=4.56
  fruit, wipes => diapers  support=0.043 confidence=0.64 lift=4.54
With a threshold of 0.3: 0 rules
With a threshold of 0.01: 93 rules

Minimal rare itemsets (support < 0.1, all their subsets with support >= 0.1): 28
Rare rules with confidence >= 0.5 and lift >= 2: 3. With a threshold of 0.01, which also reaches them, 93 rules come out with that confidence, 47 with that lift:
  diapers => wipes  support=0.089 confidence=0.64 lift=4.30
  wipes => diapers  support=0.089 confidence=0.60 lift=4.30
  wine => cheese  support=0.081 confidence=0.69 lift=2.18

Most frequent itemsets: {bread} 0.51; {milk} 0.45; {fruit} 0.41
High-utility itemsets (>= 10 % of the total utility, which is 11898):
  {cheese}  utility=4710  support=0.314
  {cheese, wine}  utility=3263  support=0.081
  {wine}  utility=3105  support=0.117
  {bread, cheese}  utility=2554  support=0.158
  {cheese, fruit}  utility=2278  support=0.124
High-utility rules with confidence >= 0.5 and lift >= 1.5 (without the lift, 'cheese => bread' is worth a lot only because the cheese is worth a lot):
  wine => cheese  support=0.081 confidence=0.69 lift=2.18  utility=3263.00  utility_share=0.27
  bread, wine => cheese  support=0.036 confidence=0.70 lift=2.21  utility=1542.90  utility_share=0.13
  fruit, wine => cheese  support=0.032 confidence=0.72 lift=2.29  utility=1397.50  utility_share=0.12
```

## Statistical significance of rules (`ARM.significance`)

Which of the rules that pass the thresholds are real? 300 random baskets of 12 independent items with a single planted
association (`i00 -> i01`) give 100 rules with confidence >= 0.45, more than half of them noise. The script keeps the real
ones with an exact test plus a false-discovery-rate correction, and with a permutation test; it also shows why, with time
series, the rows cannot be treated as independent. See [`ARM.significance`](user_guide/ARM/significance.md).

```python
from ARMxtend.ARM import (
    confidence_intervals, crisp_rule_miner, permutation_null, permutation_p_values, significant_rules,
)

fisher = significant_rules(rules, alpha=0.05, correction="fdr_bh", n_transactions=len(table))

miner = crisp_rule_miner(0.03, "confidence", 0.45)                       # the same miner as for the real rules
null = permutation_null(table, miner, n_permutations=50, null="permute")   # re-mine on shuffled copies
permuted = permutation_p_values(rules, null, n_transactions=len(table))

# time series: shuffle with circular shifts, which preserve the autocorrelation of each series
null = permutation_null(series, miner, n_permutations=60, null="shift")
```

(full source: [`examples/example_significance.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_significance.py))

Output:

```text
Baskets: 300 | rules with confidence >= 0.45: 100 | without i00 and i01 together (noise): 54
With Fisher's test and FDR (alpha=0.05): 22 rules, 2 of them noise
With permutation (50 shuffled copies): 5 rules, 0 of them noise; in the shuffled data 56 rules come out on average

i00 -> i01: support=0.213 confidence=0.71 [0.61, 0.79] lift=1.79 p=3.9e-13

11 pairs of independent series: Fisher (assumes independent rows) gives 6 false alarms (p < 0.01); the circular permutation test, 1 (p < 0.05)
```

## Rules in action: `ARM.anomaly`, `ARM.prediction`, `ARM.classification`

Mined rules are put to work. Rules learned from 3000 steps of normal operation of four sensors (motion and door
open at random; the light switches on after motion 99 % of the time, a chime sounds after the door opens 80 % of the time) are
applied to 2000 new steps in which the light fails 30 times; the same rules forecast each sensor; the sequential
rules of the [routine](#event-logs-sequential-patterns-and-rules-fimsequential-armsequential_rules) predict what comes next;
and class association rules classify 400 new cases into three diagnoses. See [anomalies](user_guide/ARM/anomaly_detection.md),
[prediction](user_guide/ARM/prediction.md) and [classification](user_guide/ARM/classification.md).

```python
from ARMxtend.ARM import (
    RuleClassifier, detect_anomalies, evaluate_forecast, evaluate_next_events, forecast_next, predict_next_events,
    prune_unproductive_rules, sequential_rules, significant_rules, temporal_association_rules,
)

rules = temporal_association_rules(train, 1, 0.02, min_threshold=0.75, targets=["light", "chime"],
                                   exclude_persistence=True, same_time_antecedent=False)
rules = significant_rules(rules, alpha=0.05)
rules = prune_unproductive_rules(rules[rules["consequents"].map(len) == 1], min_improvement=0.05)

flagged = detect_anomalies(test, rules, min_score=6.0)                  # rows that break a reliable rule
quality = evaluate_forecast(test, rules, threshold=0.7)                 # precision / recall of the predictions
upcoming = forecast_next(test, rules)                                   # the step after the last row

routine = sequential_rules(learned, min_supp=0.2, min_conf=0.3, max_antecedent=2, max_consequent=1)
predict_next_events([["alarm"], ["bathroom"], ["kitchen"], ["coffee"]], routine)
evaluate_next_events(unseen, routine, k=2, min_prefix=3, train=learned, target="next")

model = RuleClassifier(min_supp=0.05, min_conf=0.7, max_len=3, min_improvement=0.1).fit(cases, diagnosis)
model.score(new_cases, new_diagnosis)
```

(full source: [`examples/example_rules_in_action.py`](https://github.com/cjferba/ARMxtend/blob/master/examples/example_rules_in_action.py))

Output:

```text
Rules from normal operation (3000 instants):
  motion (t-1) => light (t)  confidence=0.99
  door (t-1) => chime (t)  confidence=0.77

New data: 2000 instants with 30 light faults. 39 instants exceed the 6 bits of surprise and contain 30 of the faults
AUC with weight 'surprise': 0.998; with weight 'count': 0.980
The most anomalous, 2024-03-02 19:00:00: score=9.4 for motion (t-1) => light (t); door (t-1) => chime (t)

Prediction of each sensor with the rules (threshold 0.7), on new data:
  chime  precision=0.80 recall=0.77 (the sensor activates 19% of the time)
  light  precision=0.94 recall=0.82 (the sensor activates 36% of the time)
Forecast for the instant after the last reading (motion: yes, door: yes): light (0.99), chime (0.77)

After alarm, bathroom, kitchen, coffee the expected next is: tv (0.77), phone (0.65)
On 100 cuts of 40 new days (with 3 or more events already occurred), the 2 predicted events include the next one 63% of the time (predicting the 2 most frequent: 49%)

CBA classifier: 12 candidate class rules, 2 selected; accuracy 0.96 on 400 new cases (majority class: 0.56)
  si cough y fever => flu  confidence=0.96 support=0.24
  si itchy y sneezing => allergy  confidence=0.71 support=0.18
  en otro caso => healthy
```

## Big Data and streaming algorithms

The Spark-based algorithms (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`, `SARE`,
`SFIM`) are exercised in `tests/test_fim_apriori_eclat.py`, `tests/test_sare_association_rules.py`
and the rest of the `tests/` suite against `tests/spark_stub.py`, a small local double of the Spark
RDD API -- see the [User Guide](USER_GUIDE_INDEX.md) for each algorithm's reference and the
`pip install -e ".[spark]"` note in the project [README](https://github.com/cjferba/ARMxtend#readme)
for running them against a real cluster.

## Estimators and synthetic data (`RuleMiner`, `datasets`)

[`make_transactions`](user_guide/datasets.md) plants 8 patterns in 2000 baskets; [`RuleMiner`](user_guide/estimators.md)
mines them with a lift and significance filter and recovers the frequent ones (the rarer ones stay below the support threshold,
as designed); the rules become 150 boolean features, and a `RuleClassifier` trained on 1000 labeled rows scores about 0.98 on the other 500.

```python
from ARMxtend.datasets import make_transactions
from ARMxtend.estimators import RuleMiner, TransactionEncoder

baskets, patterns = make_transactions(2000, 60, avg_length=7, n_patterns=8, pattern_length=3, random_state=7, return_patterns=True)
miner = RuleMiner(min_supp=0.1, min_threshold=0.8, min_lift=1.5, alpha=0.01).fit(TransactionEncoder().fit_transform(baskets))
features = miner.transform(baskets)          # one boolean column per rule
```

The full script is `examples/example_estimators.py`.
