# `estimators` -- API al estilo scikit-learn

Las funciones de ARMxtend reciben listas de cestas y devuelven `DataFrame`s. Para el código que ya usa scikit-learn (o
mlxtend) existen también **estimadores** con `fit` / `transform` / `get_params`. No necesitan scikit-learn: solo se usa,
si está instalado, para que `clone`, `Pipeline` y `GridSearchCV` los acepten.

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

Cestas <-> tabla booleana, como en `mlxtend.preprocessing.TransactionEncoder`. `fit` aprende `columns_` (ítems
ordenados); `transform` construye la tabla, ignorando los ítems no vistos en `fit` (con `handle_unknown="error"` lanza un
error); `inverse_transform` hace el camino inverso. Se acepta como entrada un `DataFrame` booleano (o 0/1), que conserva
su índice.

## `RuleMiner`

Un estimador para las tres formas de pedir reglas:

| `method` | qué devuelve | parámetros principales |
|---|---|---|
| `"fpgrowth"` (por defecto) | todas las reglas con soporte >= `min_supp` | `min_supp`, `metric`, `min_threshold` |
| `"top_k"` | las `k` reglas con mayor soporte ([`ARM.top_k_rules`](../../user_guide/ARM/topk_rules.md)) | `k`, `min_threshold` (confianza), `max_len` |
| `"rare"` | reglas entre ítems frecuentes que son raros juntos ([`ARM.rare_rules`](../../user_guide/ARM/rare_rules.md)) | `max_supp`, `min_count`, `min_threshold` |

`algorithm="dense"` (solo con `"fpgrowth"`) busca los itemsets con [`FIM.dense_itemsets`](../../user_guide/comparison.md), un Apriori de
vectores de bits más rápido en tablas con pocos ítems; las reglas son las mismas.

Comunes a todos: `min_lift`, `alpha` (conserva solo las reglas [significativas](../../user_guide/ARM/significance.md), con `correction` y
`test`) y `prune` (elimina las reglas [redundantes](../../user_guide/ARM/redundancy.md)). Tras `fit`: `rules_`, `n_transactions_` y, para
`"fpgrowth"`, `itemsets_`. Con `"top_k"` y `"rare"` la medida es la confianza y `min_supp` es solo un umbral inferior
opcional.

`transform(X)` da una columna booleana por regla, verdadera cuando la cesta satisface su **antecedente**. Esas columnas
son características para otro modelo en un `Pipeline`; si además se cumple el consecuente es lo que aprendería un
clasificador. Sin reglas, el resultado tiene las filas de `X` y ninguna columna.

## `RuleClassifier` y `ParamsMixin`

[`ARM.RuleClassifier`](../../user_guide/ARM/classification.md) (CBA) sigue el mismo protocolo, de modo que
`GridSearchCV(RuleClassifier(), {"min_supp": [0.02, 0.05]}, cv=3)` funciona. Cualquier clase que guarde sin cambios los
argumentos de su constructor y herede de `ARMxtend.base.ParamsMixin` obtiene `get_params`, `set_params` y un `repr`
corto.

## Qué conviene saber

- **Sin dependencia de scikit-learn.** `RuleMiner` es un transformador en espíritu, pero no hereda de las clases de
  scikit-learn; las pruebas ejecutan `clone` y `GridSearchCV` solo si scikit-learn está instalado (probado con
  scikit-learn 1.x).
- **`transform` no es una probabilidad.** Indica que una regla se aplica, no que acierte; use la `confidence` de la regla.
- **`fit` ignora `y`**; se acepta para que los pipelines puedan llamarlo.
