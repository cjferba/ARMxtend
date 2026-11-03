[English](../quick-start.md) · [Español](../es/quick-start.md) · [Français](../fr/quick-start.md) · [Português](../pt/quick-start.md)

# Démarrage rapide

Tous les extraits de cette page sont exécutables tels quels (ils sont aussi couverts par la suite de tests dans
`tests/`). Voir d'abord l'[Installation](installation.md).

## 1. Règles d'association classiques

`ARM.association_rules` prend un `pandas.DataFrame` d'itemsets fréquents (colonnes `support` et
`itemsets`, comme produit par `FFIM.fpgrowth`, voir ci-dessous) et renvoie les règles dont la `metric`
atteint `min_threshold`. Outre les habituelles `confidence`/`lift`/`leverage`/`conviction`, elle prend en charge
`certainty_factor` (le CF de Shortliffe et Buchanan) qui, contrairement à la confiance, tient compte du taux de base
du conséquent et est borné dans `[-1, 1]` :

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

Notez le facteur de certitude négatif : bien que « bread -> milk » dépasse le seuil de confiance de 0.5, le lait
est déjà présent dans 80 % des transactions à lui seul ; la présence du pain *diminue* donc en réalité la
croyance dans le lait par rapport à son taux de base, ce que la confiance seule ne révèle pas.

## 2. Itemsets fréquents avec FP-Growth

`FFIM.fpgrowth` extrait les itemsets fréquents directement des transactions (sans génération de candidats) et
renvoie un `dict {clé_itemset: support}` prêt à être transmis à `association_rules` après conversion
vers le DataFrame attendu (`FIM._shared.key_to_itemset` annule l'encodage de clé `'A-B-C'`) :

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

## 3. Règles d'association floues

Les itemsets et règles flous généralisent le cas classique à des items ayant un *degré* d'appartenance dans
`[0, 1]` (p. ex. « froid » ou « faible humidité » au lieu d'un seuil strict). `FFIM.fuzzy_fpgrowth` extrait les
itemsets fréquents flous en décomposant la base floue en `num_alpha` alpha-coupes (10 est une bonne
valeur par défaut, voir [Fernandez-Basso, Ruiz et Martin-Bautista, 2021](../cite.md)) ; `FIM.FARE.fuzzy_association_rules`
en déduit ensuite le support flou (FSupp), la confiance (FConf) et le facteur de certitude (FCF), en intégrant
la mesure classique à chaque alpha-coupe pondérée par la largeur de cette coupe (voir le
[guide FARE](../user_guide/FIM/FARE.md) pour les formules exactes) :

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

`FIM.Eclat.FuzzyDECLAT` et `FIM.BD_FARE.FuzzyDAprioriTID` extraient le même type d'itemsets fréquents
flous, mais de façon distribuée sur Spark (voir la section 5).

## 4. Méta-règles d'association

Lorsque vous avez déjà extrait des règles d'association indépendamment dans *plusieurs* jeux de données (p. ex. un par
magasin, capteur ou période), `ARM.meta_rules` trouve des « règles sur les règles » : quelles règles primaires
tendent à co-occurrer dans une proportion suffisante des jeux de données, éventuellement avec des attributs
supplémentaires de chaque jeu. C'est utile pour résumer un grand nombre de règles par jeu de données et
pour repérer des motifs inter-jeux invisibles dans un jeu isolé (voir
[Ruiz et al., 2016](../cite.md)).

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

`FIM.apriori`, `FIM.Eclat` et `FIM.BD_ARE`/`FIM.BD_FARE` reprennent les algorithmes ci-dessus mais distribuent
le comptage sur un cluster Spark (`pip install -e ".[spark]"`, voir l'[Installation](installation.md)).
Ils prennent un `pyspark.SparkContext` et un `RDD` de transactions au lieu d'une liste Python :

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

`SFIM` extrait les itemsets fréquents sur une fenêtre glissante Spark Streaming (voir `SFIM.main`), et
`SARE.extractAssociationRules` extrait les règles d'association de son arbre courant d'itemsets fréquents.

## 6. Visualiser les règles

`VizARM.AREtoGraph` transforme un DataFrame de règles en graphe typé VizARE (nœuds d'items et nœuds de
règles reliés par des arêtes `antecedent`/`consequent`), selon Fernandez-Basso et al. (2026, voir
[Citer ARMxtend](../cite.md)), prêt à être exporté en JSON Graph Format, GraphML ou DOT (Gephi,
Cytoscape, Graphviz, NetworkX) :

```python
from ARMxtend.VizARM import AREtoGraph

graph = AREtoGraph.from_dataframe(rules, rule_measures=("confidence", "certainty_factor"))
jgf = graph.exportGraph(type=0)  # 0: JGF, 1: GraphML, 2: DOT, 3: the networkx.DiGraph itself
graph.to_html("rules.html")      # interactive, self-contained viewer: open it in any browser
```

## 7. Fuzzifier des attributs numériques

`preprocessing.FuzzyLib` transforme une colonne numérique en un ensemble d'étiquettes floues qui se chevauchent (une
partition triangulaire de Ruspini), prête à servir d'items dans les chaînes floues ci-dessus :

```python
import pandas as pd
from ARMxtend.preprocessing import FuzzyLib

fuzzy_lib = FuzzyLib()
fuzzy_lib.data = pd.DataFrame({"temperature": [15, 19, 22, 25, 28]})
added_columns = fuzzy_lib.Fuzzification(["temperature"], [[18, 22, 26]], [["cold", "comfort", "warm"]])
# adds temperature_cold, temperature_comfort, temperature_warm columns, degrees summing to 1 per row
```
