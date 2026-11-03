# `FFIM` -- FP-Growth (nítido e difuso)

Extração de itemsets frequentes em uma única máquina por meio do FP-Growth (Han, Pei e Yin, 2000): constrói uma árvore de
padrões frequentes e a percorre recursivamente sem gerar candidatos, o que costuma ser mais rápido que as abordagens do
tipo Apriori/Eclat em uma única máquina.

## Nítido: `fpgrowth`

```python
fpgrowth(transactions, min_supp)
```

- **transactions** (`Sequence[Iterable[str]]`): uma transação por elemento.
- **min_supp** (`float`): suporte relativo mínimo, em `(0, 1]`.

Devolve um `dict {chave_itemset: suporte}`, por exemplo `{"bread": 0.8, "bread-milk": 0.6, ...}` -- converta uma
chave de volta em `frozenset` com `FIM._shared.key_to_itemset` antes de passá-la a
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

- **transactions** (`Sequence[Iterable[Tuple[str, float]]]`): cada transação é uma lista de pares
  `(item, grau de pertinência em [0, 1])`.
- **num_alpha** (`int`): número de alfa-cortes usados para decompor a base de dados difusa (10 é um valor padrão
  razoável, veja [Como citar o ARMxtend](../../cite.md)).

Como a FP-tree funde os caminhos de transações que compartilham *os mesmos* itens, ela não consegue acomodar
diretamente graus de pertinência reais arbitrários (duas transações com itens idênticos mas graus diferentes deixariam
de ser "o mesmo caminho"). `fuzzy_fpgrowth` contorna isso binarizando a base difusa de forma independente em cada um dos
`num_alpha` alfa-cortes (veja `FIM._shared.alpha_cuts`) para descobrir itemsets candidatos e, depois, recalculando o
suporte exato de cada candidato em cada alfa-corte. Devolve `dict {chave_itemset: numpy.ndarray(num_alpha)}` -- a lista
de bits de suporte relativo do itemset em cada alfa-corte, pronta para ser passada a
[`FIM.FARE.fuzzy_association_rules`](../../user_guide/FIM/FARE.md) (exatamente a mesma convenção de saída de
[`FIM.Eclat.FuzzyDECLAT`](../../user_guide/FIM/Eclat.md) e [`FIM.BD_FARE.FuzzyDAprioriTID`](../../user_guide/FIM/BD_FARE.md), de modo que os três são
intercambiáveis como entrada do `FARE`).

Um itemset é considerado frequente se o seu suporte difuso **agregado** FSupp (uma soma ponderada do seu suporte em cada
alfa-corte, veja [`FIM.FARE`](../../user_guide/FIM/FARE.md)) atinge `min_supp` -- e não apenas se for frequente isoladamente em um único
alfa-corte.

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

## Mais rápido: `FIM.dense_fuzzy_itemsets`

`fuzzy_fpgrowth` executa o FP-Growth uma vez por alfa-corte e depois percorre novamente as transações para cada
candidato. Quando o número de itens difusos é moderado (de dezenas a algumas centenas), existe um extrator exato muito
mais rápido com a **mesma entrada e saída**:

```python
from ARMxtend.FIM import dense_fuzzy_itemsets

dense_fuzzy_itemsets(fuzzy_transactions, min_supp=0.3, num_alpha=10)    # same dict as fuzzy_fpgrowth
```

Mantém um vetor de bits compactado por item *e* alfa-corte, e conta os candidatos de cada nível do Apriori com um E
bit a bit e uma contagem de bits. O suporte agregado FSupp é antimonótono (é uma soma ponderada não negativa de suportes
que o são), de modo que a poda sobre ele é exata. Medido com `benchmarks/fuzzy_dense.py` (10 alfa-cortes,
`min_supp=0.03`, dados sintéticos, melhor de 2):

| transações | itens | itemsets | `fuzzy_fpgrowth` | `dense_fuzzy_itemsets` |
|---:|---:|---:|---:|---:|
| 2 000 | 30 | 353 | 2.317 s | 0.009 s |
| 2 000 | 200 | 135 | 0.524 s | 0.012 s |
| 5 000 | 30 | 354 | 5.735 s | 0.020 s |
| 5 000 | 200 | 138 | 1.338 s | 0.023 s |

Verifica-se que a saída é idêntica em cada execução. Com milhares de itens muito esparsos, mantenha `fuzzy_fpgrowth`.
Pela linha de comando: `armxtend fuzzy-rules data.csv --algorithm dense`.
