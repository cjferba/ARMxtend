# `ARM.association_rules`

Gera regras de associação nítidas (crisp) a partir de um conjunto de itemsets frequentes já extraídos (por exemplo com
[`FFIM.fpgrowth`](../FFIM.md) ou [`FIM.apriori`](../../../user_guide/FIM/apriori.md)/[`FIM.Eclat`](../../../user_guide/FIM/Eclat.md) no Spark).

## Assinatura

```python
association_rules(df, metric="confidence", min_threshold=0.8, support_only=False)
```

- **df** (`pandas.DataFrame`): itemsets frequentes, com as colunas `support` (float) e `itemsets`
  (um `frozenset` de itens).
- **metric** (`str`): uma de `"support"`, `"confidence"`, `"lift"`, `"leverage"`, `"conviction"`,
  `"certainty_factor"` ou qualquer uma das [medidas adicionais](#medidas-de-interesse-adicionais) abaixo.
- **min_threshold** (`float`): valor mínimo de `metric` para que uma regra seja devolvida.
- **support_only** (`bool`): se for `True`, calcula apenas `support` (útil quando `df` não contém o suporte de todos os
  antecedentes/consequentes necessários para as outras métricas).

Devolve um `pandas.DataFrame` com as colunas `antecedents`, `consequents`, `antecedent support`,
`consequent support`, `support`, `confidence`, `lift`, `leverage`, `conviction`,
`certainty_factor`, mais uma coluna por medida adicional.

## Medidas de interesse adicionais

Todas são funções de `(supp(AB), supp(A), supp(B))` (de forma equivalente, da tabela de quatro células
`<a, b, c, d> = <sAB, sA-sAB, sB-sAB, 1-sA-sB+sAB>`), definidas em `ARMxtend.ARM._measures`. São
neutras (`0`) quando não estão definidas (`0/0`), nunca `NaN`.

| Medida (`metric=`) | Definição | Intervalo | Simétrica |
|---|---|---|---|
| `jaccard` | `sAB / (sA + sB - sAB)` | [0, 1] | sim |
| `cosine` | `sAB / sqrt(sA sB)` | [0, 1] | sim |
| `kulczynski` | `(Conf(A->B) + Conf(B->A)) / 2` -- robusta a transações nulas (Wu, Chen e Han, 2010) | [0, 1] | sim |
| `all_confidence` | `sAB / max(sA, sB)` = `min(Conf(A->B), Conf(B->A))` | [0, 1] | sim |
| `max_confidence` | `max(Conf(A->B), Conf(B->A))` | [0, 1] | sim |
| `added_value` | `Conf(A->B) - supp(B)` (Tan, Kumar e Srivastava, 2004) | [-1, 1] | não |
| `zhangs_metric` | `(sAB - sA sB) / max(sAB(1-sA), sB(sA-sAB))` (Zhang, 2000) | [-1, 1] | não |
| `yules_q` | `(ad - bc) / (ad + bc)` | [-1, 1] | sim |

Para perguntas sobre a direção (assimétricas), use `added_value`, `zhangs_metric` ou `certainty_factor`; para saber se
"A e B apenas coincidem", as simétricas. No caso difuso ([`FIM.FARE`](../../../user_guide/FIM/FARE.md)), cada uma delas é avaliada em
cada alfa-corte e agregada com os pesos da teoria RL, exatamente como `FConf` e `FCF`.

## Fator de certeza

Além da confiança, do lift e da convicção clássicos, `metric="certainty_factor"` calcula o fator de certeza de
Shortliffe e Buchanan (Delgado, Ruiz e Sánchez):

```
CF(A -> B) = (Conf(A->B) - supp(B)) / (1 - supp(B))   if Conf(A->B) > supp(B)
           = (Conf(A->B) - supp(B)) / supp(B)          if Conf(A->B) < supp(B)
           = 0                                          if Conf(A->B) = supp(B)
```

Ao contrário da confiança, `CF` é limitado a `[-1, 1]` e leva em conta a *taxa base* do consequente: um valor positivo
significa que o antecedente aumenta de fato a crença no consequente, um valor negativo significa que a *diminui* (uma
associação negativa ou independente: exatamente o tipo de regra que a confiança sozinha pode classificar
enganosamente como "interessante") e 0 significa que não há mudança. Veja [Como citar o ARMxtend](../../../cite.md) para a
referência.

## Exemplo

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

Aqui a confiança sozinha (0,6 >= 0,5) marcaria "bread -> milk" como interessante, mas o fator de certeza negativo revela
que o leite já está presente em 80 % de todas as transações, de modo que a presença do pão torna o leite na verdade
*menos* provável em relação à sua taxa base.
