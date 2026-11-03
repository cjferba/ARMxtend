# `ARM.association_rules`

Génère des règles d'association nettes (crisp) à partir d'un ensemble d'itemsets fréquents déjà extraits (par exemple avec
[`FFIM.fpgrowth`](../FFIM.md) ou [`FIM.apriori`](../../../user_guide/FIM/apriori.md)/[`FIM.Eclat`](../../../user_guide/FIM/Eclat.md) sur Spark).

## Signature

```python
association_rules(df, metric="confidence", min_threshold=0.8, support_only=False)
```

- **df** (`pandas.DataFrame`) : itemsets fréquents, avec les colonnes `support` (float) et `itemsets`
  (un `frozenset` d'items).
- **metric** (`str`) : l'une de `"support"`, `"confidence"`, `"lift"`, `"leverage"`, `"conviction"`,
  `"certainty_factor"` ou l'une des [mesures supplémentaires](#mesures-dinteret-supplementaires) ci-dessous.
- **min_threshold** (`float`) : valeur minimale de `metric` pour qu'une règle soit renvoyée.
- **support_only** (`bool`) : si `True`, ne calcule que `support` (utile lorsque `df` ne contient pas le support de tous
  les antécédents/conséquents nécessaires aux autres métriques).

Renvoie un `pandas.DataFrame` avec les colonnes `antecedents`, `consequents`, `antecedent support`,
`consequent support`, `support`, `confidence`, `lift`, `leverage`, `conviction`,
`certainty_factor`, plus une colonne par mesure supplémentaire.

## Mesures d'intérêt supplémentaires

Toutes sont des fonctions de `(supp(AB), supp(A), supp(B))` (de façon équivalente, du tableau à quatre cases
`<a, b, c, d> = <sAB, sA-sAB, sB-sAB, 1-sA-sB+sAB>`), définies dans `ARMxtend.ARM._measures`. Elles sont
neutres (`0`) lorsqu'elles ne sont pas définies (`0/0`), jamais `NaN`.

| Mesure (`metric=`) | Définition | Plage | Symétrique |
|---|---|---|---|
| `jaccard` | `sAB / (sA + sB - sAB)` | [0, 1] | oui |
| `cosine` | `sAB / sqrt(sA sB)` | [0, 1] | oui |
| `kulczynski` | `(Conf(A->B) + Conf(B->A)) / 2` -- robuste aux transactions nulles (Wu, Chen et Han, 2010) | [0, 1] | oui |
| `all_confidence` | `sAB / max(sA, sB)` = `min(Conf(A->B), Conf(B->A))` | [0, 1] | oui |
| `max_confidence` | `max(Conf(A->B), Conf(B->A))` | [0, 1] | oui |
| `added_value` | `Conf(A->B) - supp(B)` (Tan, Kumar et Srivastava, 2004) | [-1, 1] | non |
| `zhangs_metric` | `(sAB - sA sB) / max(sAB(1-sA), sB(sA-sAB))` (Zhang, 2000) | [-1, 1] | non |
| `yules_q` | `(ad - bc) / (ad + bc)` | [-1, 1] | oui |

Pour les questions de direction (asymétriques), utilisez `added_value`, `zhangs_metric` ou `certainty_factor` ; pour
savoir si « A et B coïncident simplement », les symétriques. Dans le cas flou ([`FIM.FARE`](../../../user_guide/FIM/FARE.md)), chacune
est évaluée à chaque alpha-coupe et agrégée avec les poids de la théorie RL, exactement comme `FConf` et `FCF`.

## Facteur de certitude

Outre la confiance, le lift et la conviction classiques, `metric="certainty_factor"` calcule le facteur de certitude de
Shortliffe et Buchanan (Delgado, Ruiz et Sánchez) :

```
CF(A -> B) = (Conf(A->B) - supp(B)) / (1 - supp(B))   if Conf(A->B) > supp(B)
           = (Conf(A->B) - supp(B)) / supp(B)          if Conf(A->B) < supp(B)
           = 0                                          if Conf(A->B) = supp(B)
```

Contrairement à la confiance, `CF` est borné dans `[-1, 1]` et tient compte du *taux de base* du conséquent : une valeur
positive signifie que l'antécédent augmente réellement la croyance dans le conséquent, une valeur négative qu'il la
*diminue* (une association négative ou indépendante : exactement le genre de règle que la confiance seule peut classer à
tort comme « intéressante »), et 0 qu'il n'y a aucun changement. Voir [Citer ARMxtend](../../../cite.md) pour la
référence.

## Exemple

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

Ici, la confiance seule (0,6 >= 0,5) signalerait « bread -> milk » comme intéressante, mais le facteur de certitude
négatif révèle que le lait est déjà présent dans 80 % de toutes les transactions : la présence du pain rend donc en
réalité le lait *moins* probable par rapport à son taux de base.
