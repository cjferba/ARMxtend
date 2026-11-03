# `FFIM` -- FP-Growth (net et flou)

Extraction d'itemsets fréquents sur une seule machine via FP-Growth (Han, Pei et Yin, 2000) : construit un arbre de
motifs fréquents et l'explore récursivement sans générer de candidats, ce qui est généralement plus rapide que les
approches de type Apriori/Eclat sur une seule machine.

## Net : `fpgrowth`

```python
fpgrowth(transactions, min_supp)
```

- **transactions** (`Sequence[Iterable[str]]`) : une transaction par élément.
- **min_supp** (`float`) : support relatif minimal, dans `(0, 1]`.

Renvoie un `dict {clé_itemset: support}`, par exemple `{"bread": 0.8, "bread-milk": 0.6, ...}` -- reconvertissez une
clé en `frozenset` avec `FIM._shared.key_to_itemset` avant de la passer à
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

## Flou : `fuzzy_fpgrowth`

```python
fuzzy_fpgrowth(transactions, min_supp, num_alpha)
```

- **transactions** (`Sequence[Iterable[Tuple[str, float]]]`) : chaque transaction est une liste de paires
  `(item, degré d'appartenance dans [0, 1])`.
- **num_alpha** (`int`) : nombre d'alpha-coupes utilisées pour décomposer la base de données floue (10 est une valeur
  par défaut raisonnable, voir [Citer ARMxtend](../../cite.md)).

Comme le FP-tree fusionne les chemins de transactions qui partagent *les mêmes* items, il ne peut pas accueillir
directement des degrés d'appartenance réels arbitraires (deux transactions aux items identiques mais aux degrés
différents cesseraient d'être « le même chemin »). `fuzzy_fpgrowth` contourne le problème en binarisant la base floue
indépendamment à chacune des `num_alpha` alpha-coupes (voir `FIM._shared.alpha_cuts`) pour découvrir les itemsets
candidats, puis en recalculant le support exact de chaque candidat à chaque alpha-coupe. Il renvoie
`dict {clé_itemset: numpy.ndarray(num_alpha)}` -- la liste de bits de support relatif de l'itemset à chaque alpha-coupe,
prête à être passée à [`FIM.FARE.fuzzy_association_rules`](../../user_guide/FIM/FARE.md) (exactement la même convention de sortie que
[`FIM.Eclat.FuzzyDECLAT`](../../user_guide/FIM/Eclat.md) et [`FIM.BD_FARE.FuzzyDAprioriTID`](../../user_guide/FIM/BD_FARE.md), de sorte que les trois sont
interchangeables comme entrée de `FARE`).

Un itemset est considéré comme fréquent si son support flou **agrégé** FSupp (une somme pondérée de son support à chaque
alpha-coupe, voir [`FIM.FARE`](../../user_guide/FIM/FARE.md)) atteint `min_supp` -- et non simplement s'il est fréquent isolément à une
seule alpha-coupe.

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

## Plus rapide : `FIM.dense_fuzzy_itemsets`

`fuzzy_fpgrowth` exécute FP-Growth une fois par alpha-coupe puis reparcourt les transactions pour chaque candidat.
Lorsque le nombre d'items flous est modéré (de quelques dizaines à quelques centaines), il existe un extracteur exact
bien plus rapide avec les **mêmes entrée et sortie** :

```python
from ARMxtend.FIM import dense_fuzzy_itemsets

dense_fuzzy_itemsets(fuzzy_transactions, min_supp=0.3, num_alpha=10)    # same dict as fuzzy_fpgrowth
```

Il garde un vecteur de bits compact par item *et* alpha-coupe, et compte les candidats de chaque niveau d'Apriori avec
un ET bit à bit et un comptage de bits. Le support agrégé FSupp est anti-monotone (c'est une somme pondérée non négative
de supports qui le sont), de sorte que l'élagage sur ce support est exact. Mesuré avec `benchmarks/fuzzy_dense.py`
(10 alpha-coupes, `min_supp=0.03`, données synthétiques, meilleur de 2) :

| transactions | items | itemsets | `fuzzy_fpgrowth` | `dense_fuzzy_itemsets` |
|---:|---:|---:|---:|---:|
| 2 000 | 30 | 353 | 2.317 s | 0.009 s |
| 2 000 | 200 | 135 | 0.524 s | 0.012 s |
| 5 000 | 30 | 354 | 5.735 s | 0.020 s |
| 5 000 | 200 | 138 | 1.338 s | 0.023 s |

La sortie est vérifiée comme identique à chaque exécution. Avec des milliers d'items très épars, conservez
`fuzzy_fpgrowth`. En ligne de commande : `armxtend fuzzy-rules data.csv --algorithm dense`.
