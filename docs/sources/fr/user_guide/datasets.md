# `datasets` -- données synthétiques et petits jeux de données d'exemple

Pour essayer un algorithme, ou le mesurer, il faut des données dont on connaît la réponse. `ARMxtend.datasets` propose
des générateurs qui **plantent des motifs** (pour vérifier qu'ils ressortent) et quelques jeux de données minuscules
inclus dans le paquet ; rien n'est téléchargé.

```python
from ARMxtend.datasets import make_transactions, make_sequences, make_quantities, make_labeled_items, load_dataset

baskets, patterns = make_transactions(2000, n_items=60, n_patterns=8, pattern_length=3, random_state=7, return_patterns=True)
sequences = make_sequences(500, random_state=1)
quantities, profits = make_quantities(1000, random_state=1)     # for high_utility_rules
items, labels = make_labeled_items(1500, noise=0.05, random_state=3)   # for RuleClassifier
weather = load_dataset("weather")                             # weather.data, weather.target
```

## Générateurs

Tous prennent `random_state` (un entier ou un `numpy.random.Generator`) et donnent les mêmes données pour la même graine.

- **`make_transactions`** : dans le style du générateur IBM Quest (Agrawal et Srikant, 1994, mais ce n'est pas un portage).
  Chaque transaction est un mélange de motifs plantés, chaque item étant perdu avec la probabilité `corruption`, et des
  items isolés sont tirés avec une popularité de type Zipf. Les poids des motifs sont exponentiels : quelques-uns sont
  fréquents et certains rares. Avec `return_patterns=True`, vous obtenez aussi les motifs.
- **`make_sequences`** : séquences d'itemsets ; avec la probabilité 1/2, un motif planté est intercalé dans l'ordre,
  chacun de ses items dans son propre élément.
- **`make_quantities`** : paniers de `make_transactions` avec des quantités dans `1..max_quantity` et des profits
  unitaires log-normaux.
- **`make_labeled_items`** : un tableau booléen et une classe décidée par `n_rules` règles à deux conditions (la première
  qui s'applique) ; `noise` modifie au hasard cette fraction des classes, de sorte que `1 - noise` environ est la
  meilleure exactitude que l'on puisse attendre.

## Jeux d'exemple : `load_dataset(name)`

| nom | contenu |
|---|---|
| `"han"` | 9 transactions sur 5 items (l'exemple AllElectronics de Han et Kamber) |
| `"weather"` | 14 jours, 4 attributs catégoriels comme items (`outlook=sunny`) et la classe `play` (le tableau du tennis de Quinlan) |
| `"sales"` | quantités de 7 items dans 5 ventes et leurs profits unitaires (l'exemple de [haute utilité](../../user_guide/ARM/utility_rules.md)) |
| `"routines"` | 6 jours d'événements d'une maison, sous forme de séquences |

Le résultat est un `Dataset` (un dictionnaire dont les clés se lisent aussi comme des attributs) avec `data`,
`description` et, selon le cas, `target`, `profits` ou `table`. Chaque appel donne des copies fraîches.
`list_datasets()` liste les noms.

Ligne de commande : [`armxtend generate`](../../user_guide/cli.md#generate) écrit des données synthétiques dans les formats que lisent les
autres commandes.

## À savoir

- **Les données synthétiques sont plus faciles que les données réelles.** Les motifs plantés se séparent nettement du
  bruit, ce que les données réelles font rarement ; utilisez les générateurs pour vérifier et chronométrer des
  algorithmes, pas pour comparer leur qualité.
- **Certains motifs plantés ne seront pas trouvés** pour un `min_supp` donné : les poids exponentiels en rendent certains
  rares (c'est voulu). Comparez avec `return_patterns`, pas avec le nombre de motifs.
- **Les petits jeux de données ont été saisis à partir des tableaux habituels des manuels.** Vérifiez-les avec les sources
  si vous publiez des chiffres obtenus avec eux.

## Références

- R. Agrawal, R. Srikant (1994). *Fast algorithms for mining association rules.* VLDB 1994, 487-499 (le générateur de
  données synthétiques utilisé pour évaluer les extracteurs d'itemsets).
- J. Han, M. Kamber (2006). *Data Mining: Concepts and Techniques*, 2e éd. (l'exemple à 9 transactions).
- J. R. Quinlan (1986). *Induction of decision trees.* Machine Learning 1, 81-106 (le tableau météo).
