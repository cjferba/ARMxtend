[English](../quick-start.md) · [Español](../es/quick-start.md) · [Français](../fr/quick-start.md) · [Português](../pt/quick-start.md)

# Início rápido

Todos os trechos desta página são executáveis como estão (também são cobertos pela suíte de testes em
`tests/`). Veja primeiro a [Instalação](installation.md).

## 1. Regras de associação clássicas

`ARM.association_rules` recebe um `pandas.DataFrame` de itemsets frequentes (colunas `support` e
`itemsets`, como produzido por `FFIM.fpgrowth`, veja abaixo) e devolve as regras cuja `metric`
atinge `min_threshold`. Além das habituais `confidence`/`lift`/`leverage`/`conviction`, suporta
`certainty_factor` (o CF de Shortliffe e Buchanan) que, ao contrário da confiança, leva em conta a taxa base
do consequente e é limitado a `[-1, 1]`:

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

Note o fator de certeza negativo: embora «bread -> milk» supere o limite de confiança de 0.5, o leite
já está presente em 80 % das transações por si só, então a presença do pão na verdade *diminui* a
crença no leite em relação à sua taxa base, algo que a confiança sozinha não revela.

## 2. Itemsets frequentes com FP-Growth

`FFIM.fpgrowth` extrai itemsets frequentes diretamente das transações (sem geração de candidatos) e
devolve um `dict {chave_itemset: suporte}` pronto para ser passado a `association_rules` depois de convertido
no DataFrame esperado (`FIM._shared.key_to_itemset` desfaz a codificação de chave `'A-B-C'`):

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

## 3. Regras de associação difusas

Os itemsets e regras difusos generalizam o caso clássico para itens com um *grau* de pertinência em
`[0, 1]` (p. ex. «frio» ou «baixa umidade» em vez de um limite rígido). `FFIM.fuzzy_fpgrowth` extrai
itemsets frequentes difusos decompondo a base difusa em `num_alpha` alfa-cortes (10 é um bom
valor padrão, veja [Fernandez-Basso, Ruiz e Martin-Bautista, 2021](../cite.md)); `FIM.FARE.fuzzy_association_rules`
deriva então o suporte difuso (FSupp), a confiança (FConf) e o fator de certeza (FCF), integrando
a medida clássica em cada alfa-corte ponderada pela largura desse corte (veja o
[guia do FARE](../user_guide/FIM/FARE.md) para as fórmulas exatas):

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

`FIM.Eclat.FuzzyDECLAT` e `FIM.BD_FARE.FuzzyDAprioriTID` extraem o mesmo tipo de itemsets frequentes
difusos, mas distribuídos no Spark (veja a seção 5).

## 4. Meta-regras de associação

Quando você já extraiu regras de associação de forma independente em *vários* conjuntos de dados (p. ex. um por
loja, sensor ou período), `ARM.meta_rules` encontra «regras sobre regras»: quais regras primárias
tendem a coocorrer numa proporção suficiente dos conjuntos de dados, opcionalmente junto com atributos
adicionais de cada conjunto. Serve tanto para resumir um grande número de regras por conjunto quanto
para detectar padrões entre conjuntos invisíveis em qualquer um isoladamente (veja
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

`FIM.apriori`, `FIM.Eclat` e `FIM.BD_ARE`/`FIM.BD_FARE` espelham os algoritmos acima, mas distribuem
a contagem por um cluster Spark (`pip install -e ".[spark]"`, veja a [Instalação](installation.md)).
Recebem um `pyspark.SparkContext` e um `RDD` de transações em vez de uma lista Python:

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

O `SFIM` extrai itemsets frequentes sobre uma janela deslizante do Spark Streaming (veja `SFIM.main`), e
`SARE.extractAssociationRules` extrai regras de associação da sua árvore atual de itemsets frequentes.

## 6. Visualizando regras

`VizARM.AREtoGraph` transforma um DataFrame de regras no grafo tipado do VizARE (nós de item e nós de
regra ligados por arestas `antecedent`/`consequent`), segundo Fernandez-Basso et al. (2026, veja
[Como citar o ARMxtend](../cite.md)), pronto para exportar para JSON Graph Format, GraphML ou DOT (Gephi,
Cytoscape, Graphviz, NetworkX):

```python
from ARMxtend.VizARM import AREtoGraph

graph = AREtoGraph.from_dataframe(rules, rule_measures=("confidence", "certainty_factor"))
jgf = graph.exportGraph(type=0)  # 0: JGF, 1: GraphML, 2: DOT, 3: the networkx.DiGraph itself
graph.to_html("rules.html")      # interactive, self-contained viewer: open it in any browser
```

## 7. Fuzzificando atributos numéricos

`preprocessing.FuzzyLib` transforma uma coluna numérica num conjunto de rótulos difusos sobrepostos (uma
partição triangular de Ruspini), pronta para ser usada como itens nos fluxos difusos acima:

```python
import pandas as pd
from ARMxtend.preprocessing import FuzzyLib

fuzzy_lib = FuzzyLib()
fuzzy_lib.data = pd.DataFrame({"temperature": [15, 19, 22, 25, 28]})
added_columns = fuzzy_lib.Fuzzification(["temperature"], [[18, 22, 26]], [["cold", "comfort", "warm"]])
# adds temperature_cold, temperature_comfort, temperature_warm columns, degrees summing to 1 per row
```
