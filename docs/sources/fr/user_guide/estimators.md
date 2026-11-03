# `estimators` -- API de style scikit-learn

Les fonctions d'ARMxtend prennent des listes de paniers et renvoient des `DataFrame`. Pour le code qui utilise déjà
scikit-learn (ou mlxtend), il existe aussi des **estimateurs** avec `fit` / `transform` / `get_params`. Ils n'ont pas
besoin de scikit-learn : celui-ci n'est utilisé, s'il est installé, que pour que `clone`, `Pipeline` et `GridSearchCV`
les acceptent.

```python
from ARMxtend.estimators import RuleMiner, TransactionEncoder
from ARMxtend.datasets import load_dataset

baskets = load_dataset("han").data                 # list of baskets
table = TransactionEncoder().fit_transform(baskets)   # boolean DataFrame, one column per item

miner = RuleMiner(min_supp=2 / 9, min_threshold=0.7, min_lift=1.2).fit(table)
miner.rules_                         # the usual rule frame (support, confidence, lift...)
miner.transform(baskets)             # boolean DataFrame: which rule antecedents each basket satisfies
miner.get_feature_names_out()        # one text per rule, 'I5 => I1'
```

## `TransactionEncoder`

Paniers <-> tableau booléen, comme dans `mlxtend.preprocessing.TransactionEncoder`. `fit` apprend `columns_` (items
triés) ; `transform` construit le tableau, en ignorant les items non vus dans `fit` (avec `handle_unknown="error"`, il
lève une erreur) ; `inverse_transform` fait le chemin inverse. Un `DataFrame` booléen (ou 0/1) est accepté en entrée et
conserve son index.

## `RuleMiner`

Un estimateur pour les trois façons de demander des règles :

| `method` | ce qu'il renvoie | paramètres principaux |
|---|---|---|
| `"fpgrowth"` (par défaut) | toutes les règles de support >= `min_supp` | `min_supp`, `metric`, `min_threshold` |
| `"top_k"` | les `k` règles de plus grand support ([`ARM.top_k_rules`](../../user_guide/ARM/topk_rules.md)) | `k`, `min_threshold` (confiance), `max_len` |
| `"rare"` | règles entre items fréquents qui sont rares ensemble ([`ARM.rare_rules`](../../user_guide/ARM/rare_rules.md)) | `max_supp`, `min_count`, `min_threshold` |

`algorithm="dense"` (uniquement avec `"fpgrowth"`) cherche les itemsets avec [`FIM.dense_itemsets`](../../user_guide/comparison.md), un
Apriori à vecteurs de bits plus rapide sur les tableaux à peu d'items ; les règles sont les mêmes.

Communs à tous : `min_lift`, `alpha` (ne garde que les règles [significatives](../../user_guide/ARM/significance.md), avec `correction` et
`test`) et `prune` (supprime les règles [redondantes](../../user_guide/ARM/redundancy.md)). Après `fit` : `rules_`, `n_transactions_` et,
pour `"fpgrowth"`, `itemsets_`. Avec `"top_k"` et `"rare"`, la mesure est la confiance et `min_supp` n'est qu'un plancher
facultatif.

`transform(X)` donne une colonne booléenne par règle, vraie lorsque le panier satisfait son **antécédent**. Ces colonnes
sont des caractéristiques pour un autre modèle dans un `Pipeline` ; savoir si le conséquent est aussi vérifié, c'est ce
qu'apprendrait un classifieur. Sans règles, le résultat a les lignes de `X` et aucune colonne.

## `RuleClassifier` et `ParamsMixin`

[`ARM.RuleClassifier`](../../user_guide/ARM/classification.md) (CBA) suit le même protocole, de sorte que
`GridSearchCV(RuleClassifier(), {"min_supp": [0.02, 0.05]}, cv=3)` fonctionne. Toute classe qui stocke sans changement
les arguments de son constructeur et hérite de `ARMxtend.base.ParamsMixin` obtient `get_params`, `set_params` et un
`repr` court.

## À savoir

- **Aucune dépendance à scikit-learn.** `RuleMiner` est un transformateur dans l'esprit mais n'hérite pas des classes de
  scikit-learn ; les tests exécutent `clone` et `GridSearchCV` uniquement si scikit-learn est installé (testé avec
  scikit-learn 1.x).
- **`transform` n'est pas une probabilité.** Il indique qu'une règle s'applique, pas qu'elle a raison ; utilisez la
  `confidence` de la règle.
- **`fit` ignore `y`** ; il est accepté pour que les pipelines puissent l'appeler.
