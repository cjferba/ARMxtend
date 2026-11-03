# `estimators` -- API no estilo scikit-learn

As funções do ARMxtend recebem listas de cestas e devolvem `DataFrame`s. Para o código que já usa scikit-learn (ou
mlxtend) existem também **estimadores** com `fit` / `transform` / `get_params`. Eles não precisam do scikit-learn: ele só
é usado, se estiver instalado, para que `clone`, `Pipeline` e `GridSearchCV` os aceitem.

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

Cestas <-> tabela booleana, como em `mlxtend.preprocessing.TransactionEncoder`. `fit` aprende `columns_` (itens
ordenados); `transform` constrói a tabela, ignorando os itens não vistos em `fit` (com `handle_unknown="error"` lança um
erro); `inverse_transform` faz o caminho inverso. Um `DataFrame` booleano (ou 0/1) é aceito como entrada e mantém o seu
índice.

## `RuleMiner`

Um estimador para as três formas de pedir regras:

| `method` | o que devolve | parâmetros principais |
|---|---|---|
| `"fpgrowth"` (padrão) | todas as regras com suporte >= `min_supp` | `min_supp`, `metric`, `min_threshold` |
| `"top_k"` | as `k` regras de maior suporte ([`ARM.top_k_rules`](../../user_guide/ARM/topk_rules.md)) | `k`, `min_threshold` (confiança), `max_len` |
| `"rare"` | regras entre itens frequentes que são raros juntos ([`ARM.rare_rules`](../../user_guide/ARM/rare_rules.md)) | `max_supp`, `min_count`, `min_threshold` |

`algorithm="dense"` (apenas com `"fpgrowth"`) busca os itemsets com [`FIM.dense_itemsets`](../../user_guide/comparison.md), um Apriori de
vetores de bits mais rápido em tabelas com poucos itens; as regras são as mesmas.

Comuns a todos: `min_lift`, `alpha` (mantém apenas as regras [significativas](../../user_guide/ARM/significance.md), com `correction` e
`test`) e `prune` (remove as regras [redundantes](../../user_guide/ARM/redundancy.md)). Após `fit`: `rules_`, `n_transactions_` e, para
`"fpgrowth"`, `itemsets_`. Com `"top_k"` e `"rare"` a medida é a confiança e `min_supp` é apenas um piso opcional.

`transform(X)` dá uma coluna booleana por regra, verdadeira quando a cesta satisfaz o seu **antecedente**. Essas colunas
são características para outro modelo em um `Pipeline`; saber se o consequente também vale é o que um classificador
aprenderia. Sem regras, o resultado tem as linhas de `X` e nenhuma coluna.

## `RuleClassifier` e `ParamsMixin`

[`ARM.RuleClassifier`](../../user_guide/ARM/classification.md) (CBA) segue o mesmo protocolo, de modo que
`GridSearchCV(RuleClassifier(), {"min_supp": [0.02, 0.05]}, cv=3)` funciona. Qualquer classe que guarde sem alteração os
argumentos do seu construtor e herde de `ARMxtend.base.ParamsMixin` ganha `get_params`, `set_params` e um `repr`
curto.

## O que convém saber

- **Sem dependência do scikit-learn.** `RuleMiner` é um transformador no espírito, mas não herda das classes do
  scikit-learn; os testes executam `clone` e `GridSearchCV` apenas se o scikit-learn estiver instalado (testado com
  scikit-learn 1.x).
- **`transform` não é uma probabilidade.** Indica que uma regra se aplica, não que ela acerte; use a `confidence` da regra.
- **`fit` ignora `y`**; é aceito para que os pipelines possam chamá-lo.
