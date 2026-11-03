[English](../quick-start.md) · [Español](../es/quick-start.md) · [Français](../fr/quick-start.md) · [Português](../pt/quick-start.md)

# Guía rápida

Todos los fragmentos de esta página son ejecutables tal cual (también están cubiertos por la batería de tests en
`tests/`). Consulte primero la [Instalación](installation.md).

## 1. Reglas de asociación nítidas

`ARM.association_rules` recibe un `pandas.DataFrame` de itemsets frecuentes (columnas `support` e
`itemsets`, como los produce `FFIM.fpgrowth`, véase más abajo) y devuelve las reglas cuya `metric`
alcanza `min_threshold`. Además de las habituales `confidence`/`lift`/`leverage`/`conviction`, admite
`certainty_factor` (el CF de Shortliffe y Buchanan) que, a diferencia de la confianza, tiene en cuenta la tasa base
del consecuente y está acotado en `[-1, 1]`:

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

Observe el factor de certeza negativo: aunque «bread -> milk» supera el umbral de confianza de 0.5, la leche
ya está presente en el 80 % de las transacciones por sí sola, de modo que la presencia del pan *reduce* en realidad la
creencia en la leche respecto a su tasa base, algo que la confianza por sí sola no revela.

## 2. Itemsets frecuentes con FP-Growth

`FFIM.fpgrowth` extrae itemsets frecuentes directamente de las transacciones (sin generación de candidatos) y
devuelve un `dict {clave_itemset: soporte}` listo para pasar a `association_rules` tras convertirlo
al DataFrame esperado (`FIM._shared.key_to_itemset` deshace la codificación de claves `'A-B-C'`):

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

## 3. Reglas de asociación difusas

Los itemsets y reglas difusos generalizan el caso nítido a ítems con un *grado* de pertenencia en
`[0, 1]` (p. ej. «frío» o «humedad baja» en lugar de un umbral rígido). `FFIM.fuzzy_fpgrowth` extrae
itemsets frecuentes difusos descomponiendo la base de datos difusa en `num_alpha` alfa-cortes (10 es un buen
valor por defecto, véase [Fernandez-Basso, Ruiz y Martin-Bautista, 2021](../cite.md)); `FIM.FARE.fuzzy_association_rules`
deriva después el soporte difuso (FSupp), la confianza (FConf) y el factor de certeza (FCF), integrando
la medida nítida en cada alfa-corte ponderada por la anchura de ese corte (véase la
[guía de FARE](../user_guide/FIM/FARE.md) para las fórmulas exactas):

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

`FIM.Eclat.FuzzyDECLAT` y `FIM.BD_FARE.FuzzyDAprioriTID` extraen el mismo tipo de itemsets frecuentes
difusos, pero distribuidos en Spark (véase la sección 5).

## 4. Meta-reglas de asociación

Cuando ya se han extraído reglas de asociación de forma independiente en *varios* conjuntos de datos (p. ej. uno por
tienda, sensor o periodo), `ARM.meta_rules` encuentra «reglas sobre reglas»: qué reglas primarias
tienden a coincidir en una proporción suficiente de los conjuntos de datos, opcionalmente junto con atributos
adicionales de cada conjunto. Sirve tanto para resumir un gran número de reglas por conjunto como
para detectar patrones entre conjuntos que no se ven en ninguno por separado (véase
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

`FIM.apriori`, `FIM.Eclat` y `FIM.BD_ARE`/`FIM.BD_FARE` reproducen los algoritmos anteriores pero distribuyen
el recuento en un clúster Spark (`pip install -e ".[spark]"`, véase la [Instalación](installation.md)).
Reciben un `pyspark.SparkContext` y un `RDD` de transacciones en lugar de una lista de Python:

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

`SFIM` extrae itemsets frecuentes sobre una ventana deslizante de Spark Streaming (véase `SFIM.main`), y
`SARE.extractAssociationRules` extrae reglas de asociación de su árbol actual de itemsets frecuentes.

## 6. Visualización de reglas

`VizARM.AREtoGraph` convierte un DataFrame de reglas en el grafo tipado de VizARE (nodos de ítem y nodos de
regla conectados por aristas `antecedent`/`consequent`), según Fernandez-Basso et al. (2026, véase
[Cómo citar ARMxtend](../cite.md)), listo para exportar a JSON Graph Format, GraphML o DOT (Gephi,
Cytoscape, Graphviz, NetworkX):

```python
from ARMxtend.VizARM import AREtoGraph

graph = AREtoGraph.from_dataframe(rules, rule_measures=("confidence", "certainty_factor"))
jgf = graph.exportGraph(type=0)  # 0: JGF, 1: GraphML, 2: DOT, 3: the networkx.DiGraph itself
graph.to_html("rules.html")      # interactive, self-contained viewer: open it in any browser
```

## 7. Fuzzificación de atributos numéricos

`preprocessing.FuzzyLib` convierte una columna numérica en un conjunto de etiquetas difusas solapadas (una
partición triangular de Ruspini), lista para usarse como ítems en los flujos difusos anteriores:

```python
import pandas as pd
from ARMxtend.preprocessing import FuzzyLib

fuzzy_lib = FuzzyLib()
fuzzy_lib.data = pd.DataFrame({"temperature": [15, 19, 22, 25, 28]})
added_columns = fuzzy_lib.Fuzzification(["temperature"], [[18, 22, 26]], [["cold", "comfort", "warm"]])
# adds temperature_cold, temperature_comfort, temperature_warm columns, degrees summing to 1 per row
```
