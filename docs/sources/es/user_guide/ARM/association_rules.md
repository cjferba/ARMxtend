# `ARM.association_rules`

Genera reglas de asociación nítidas (crisp) a partir de un conjunto de itemsets frecuentes ya extraídos (por ejemplo con
[`FFIM.fpgrowth`](../FFIM.md) o [`FIM.apriori`](../../../user_guide/FIM/apriori.md)/[`FIM.Eclat`](../../../user_guide/FIM/Eclat.md) en Spark).

## Firma

```python
association_rules(df, metric="confidence", min_threshold=0.8, support_only=False)
```

- **df** (`pandas.DataFrame`): itemsets frecuentes, con las columnas `support` (float) e `itemsets`
  (un `frozenset` de ítems).
- **metric** (`str`): una de `"support"`, `"confidence"`, `"lift"`, `"leverage"`, `"conviction"`,
  `"certainty_factor"` o cualquiera de las [medidas adicionales](#medidas-de-interes-adicionales) de abajo.
- **min_threshold** (`float`): valor mínimo de `metric` para que se devuelva una regla.
- **support_only** (`bool`): si es `True`, solo calcula `support` (útil cuando `df` no contiene el soporte de todos los
  antecedentes/consecuentes necesarios para las demás métricas).

Devuelve un `pandas.DataFrame` con las columnas `antecedents`, `consequents`, `antecedent support`,
`consequent support`, `support`, `confidence`, `lift`, `leverage`, `conviction`,
`certainty_factor`, más una columna por cada medida adicional.

## Medidas de interés adicionales

Todas son funciones de `(supp(AB), supp(A), supp(B))` (de forma equivalente, de la tabla de cuatro celdas
`<a, b, c, d> = <sAB, sA-sAB, sB-sAB, 1-sA-sB+sAB>`), definidas en `ARMxtend.ARM._measures`. Son
neutras (`0`) cuando no están definidas (`0/0`), nunca `NaN`.

| Medida (`metric=`) | Definición | Rango | Simétrica |
|---|---|---|---|
| `jaccard` | `sAB / (sA + sB - sAB)` | [0, 1] | sí |
| `cosine` | `sAB / sqrt(sA sB)` | [0, 1] | sí |
| `kulczynski` | `(Conf(A->B) + Conf(B->A)) / 2` -- robusta a las transacciones nulas (Wu, Chen y Han, 2010) | [0, 1] | sí |
| `all_confidence` | `sAB / max(sA, sB)` = `min(Conf(A->B), Conf(B->A))` | [0, 1] | sí |
| `max_confidence` | `max(Conf(A->B), Conf(B->A))` | [0, 1] | sí |
| `added_value` | `Conf(A->B) - supp(B)` (Tan, Kumar y Srivastava, 2004) | [-1, 1] | no |
| `zhangs_metric` | `(sAB - sA sB) / max(sAB(1-sA), sB(sA-sAB))` (Zhang, 2000) | [-1, 1] | no |
| `yules_q` | `(ad - bc) / (ad + bc)` | [-1, 1] | sí |

Para preguntas sobre la dirección (asimétricas) use `added_value`, `zhangs_metric` o `certainty_factor`; para saber
si "A y B simplemente coinciden", las simétricas. En el caso difuso ([`FIM.FARE`](../../../user_guide/FIM/FARE.md)) cada una de ellas se
evalúa en cada alfa-corte y se agrega con los pesos de la teoría RL, exactamente igual que `FConf` y `FCF`.

## Factor de certeza

Además de la confianza, el lift y la convicción clásicos, `metric="certainty_factor"` calcula el factor de certeza de
Shortliffe y Buchanan (Delgado, Ruiz y Sánchez):

```
CF(A -> B) = (Conf(A->B) - supp(B)) / (1 - supp(B))   if Conf(A->B) > supp(B)
           = (Conf(A->B) - supp(B)) / supp(B)          if Conf(A->B) < supp(B)
           = 0                                          if Conf(A->B) = supp(B)
```

A diferencia de la confianza, `CF` está acotado en `[-1, 1]` y tiene en cuenta la *tasa base* del consecuente: un valor
positivo significa que el antecedente aumenta realmente la creencia en el consecuente, un valor negativo significa que
la *disminuye* (una asociación negativa o independiente: justo el tipo de regla que la confianza sola puede presentar
erróneamente como "interesante") y 0 significa que no hay cambio. Véase [Cómo citar ARMxtend](../../../cite.md) para la
referencia.

## Ejemplo

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

Aquí la confianza por sí sola (0,6 >= 0,5) marcaría "bread -> milk" como interesante, pero el factor de certeza
negativo revela que la leche ya está presente en el 80 % de todas las transacciones, de modo que la presencia del pan
hace que la leche sea en realidad *menos* probable respecto a su tasa base.
