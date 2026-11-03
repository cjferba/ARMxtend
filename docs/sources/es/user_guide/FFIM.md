# `FFIM` -- FP-Growth (nítido y difuso)

Extracción de itemsets frecuentes en una sola máquina mediante FP-Growth (Han, Pei y Yin, 2000): construye un árbol de
patrones frecuentes y lo explora recursivamente sin generar candidatos, lo que suele ser más rápido que los enfoques
tipo Apriori/Eclat en una sola máquina.

## Nítido: `fpgrowth`

```python
fpgrowth(transactions, min_supp)
```

- **transactions** (`Sequence[Iterable[str]]`): una transacción por elemento.
- **min_supp** (`float`): soporte relativo mínimo, en `(0, 1]`.

Devuelve un `dict {clave_itemset: soporte}`, por ejemplo `{"bread": 0.8, "bread-milk": 0.6, ...}` -- convierta una
clave de nuevo en un `frozenset` con `FIM._shared.key_to_itemset` antes de pasarla a
[`ARM.association_rules`](ARM/association_rules.md).

```python
from ARMxtend.FFIM import fpgrowth

transactions = [
    ["bread", "milk"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "cola"],
]
fpgrowth(transactions, min_supp=0.5)
# {'beer': 0.6, 'beer-diapers': 0.6, 'milk': 0.8, 'diapers-milk': 0.6,
#  'bread-milk': 0.6, 'diapers': 0.8, 'bread-diapers': 0.6, 'bread': 0.8}
```

## Difuso: `fuzzy_fpgrowth`

```python
fuzzy_fpgrowth(transactions, min_supp, num_alpha)
```

- **transactions** (`Sequence[Iterable[Tuple[str, float]]]`): cada transacción es una lista de pares
  `(ítem, grado de pertenencia en [0, 1])`.
- **num_alpha** (`int`): número de alfa-cortes usados para descomponer la base de datos difusa (10 es un valor por
  defecto razonable, véase [Cómo citar ARMxtend](../../cite.md)).

Como el FP-tree fusiona los caminos de transacciones que comparten *los mismos* ítems, no puede acomodar directamente
grados de pertenencia reales arbitrarios (dos transacciones con ítems idénticos pero grados distintos dejarían de ser
"el mismo camino"). `fuzzy_fpgrowth` lo resuelve binarizando la base de datos difusa de forma independiente en cada uno
de los `num_alpha` alfa-cortes (véase `FIM._shared.alpha_cuts`) para descubrir itemsets candidatos, y recalculando
después el soporte exacto de cada candidato en cada alfa-corte. Devuelve `dict {clave_itemset: numpy.ndarray(num_alpha)}`
-- la lista de bits de soporte relativo del itemset en cada alfa-corte, lista para pasarla a
[`FIM.FARE.fuzzy_association_rules`](../../user_guide/FIM/FARE.md) (exactamente la misma convención de salida que
[`FIM.Eclat.FuzzyDECLAT`](../../user_guide/FIM/Eclat.md) y [`FIM.BD_FARE.FuzzyDAprioriTID`](../../user_guide/FIM/BD_FARE.md), de modo que los tres son
intercambiables como entrada de `FARE`).

Un itemset se considera frecuente si su soporte difuso **agregado** FSupp (una suma ponderada de su soporte en cada
alfa-corte, véase [`FIM.FARE`](../../user_guide/FIM/FARE.md)) alcanza `min_supp` -- no basta con que sea frecuente de forma aislada en un
único alfa-corte.

```python
from ARMxtend.FFIM import fuzzy_fpgrowth

fuzzy_transactions = [
    [("cold", 1.0), ("low_humidity", 0.8)],
    [("cold", 0.9), ("low_humidity", 0.6)],
    [("warm", 0.7), ("low_humidity", 0.9)],
    [("cold", 0.6), ("low_humidity", 0.7)],
    [("warm", 1.0), ("low_humidity", 0.2)],
]
fuzzy_fpgrowth(fuzzy_transactions, min_supp=0.3, num_alpha=10)
```

## Más rápido: `FIM.dense_fuzzy_itemsets`

`fuzzy_fpgrowth` ejecuta FP-Growth una vez por alfa-corte y después vuelve a recorrer las transacciones para cada
candidato. Cuando el número de ítems difusos es moderado (de decenas a unos cientos) existe un extractor exacto mucho más
rápido con la **misma entrada y salida**:

```python
from ARMxtend.FIM import dense_fuzzy_itemsets

dense_fuzzy_itemsets(fuzzy_transactions, min_supp=0.3, num_alpha=10)    # same dict as fuzzy_fpgrowth
```

Mantiene un vector de bits empaquetado por ítem *y* alfa-corte, y cuenta los candidatos de cada nivel de Apriori con un
AND bit a bit y un recuento de bits. El soporte agregado FSupp es antimonótono (es una suma ponderada no negativa de
soportes que lo son), de modo que la poda sobre él es exacta. Medido con `benchmarks/fuzzy_dense.py` (10 alfa-cortes,
`min_supp=0.03`, datos sintéticos, mejor de 2):

| transacciones | ítems | itemsets | `fuzzy_fpgrowth` | `dense_fuzzy_itemsets` |
|---:|---:|---:|---:|---:|
| 2 000 | 30 | 353 | 2.317 s | 0.009 s |
| 2 000 | 200 | 135 | 0.524 s | 0.012 s |
| 5 000 | 30 | 354 | 5.735 s | 0.020 s |
| 5 000 | 200 | 138 | 1.338 s | 0.023 s |

Se comprueba que la salida es idéntica en cada ejecución. Con miles de ítems muy dispersos conserve `fuzzy_fpgrowth`.
Desde la línea de comandos: `armxtend fuzzy-rules data.csv --algorithm dense`.
