[English](../installation.md) · [Español](../es/installation.md) · [Français](../fr/installation.md) · [Português](../pt/installation.md)

# Installation d'ARMxtend

---

ARMxtend est en cours de préparation pour sa première publication sur PyPI (voir le [journal des modifications](https://github.com/cjferba/ARMxtend/blob/master/CHANGELOG.md)).
En attendant, installez-le depuis le [dépôt GitHub](https://github.com/cjferba/ARMxtend) :

```bash
pip install git+https://github.com/cjferba/ARMxtend.git
# or, to work on it, from a local checkout:
git clone https://github.com/cjferba/ARMxtend.git
cd ARMxtend
pip install -e .
```

Une fois publié, `pip install armxtend` suffira. Python 3.9 ou supérieur est requis (testé sur 3.9-3.12).
L'installation fournit aussi la [commande `armxtend`](../user_guide/cli.md).

Cela installe le cœur de la bibliothèque, pour une seule machine : `ARM` (règles d'association classiques et
méta-règles), `FIM.FARE` (règles d'association floues), `FFIM` (FP-Growth classique et flou), `VizARM`
et `preprocessing`. Ses seules dépendances sont `numpy`, `pandas` et `networkx`.

### Extra Big Data / streaming

Les algorithmes Big Data et streaming (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`,
`SARE`, `SFIM`) s'exécutent sur [Apache Spark](https://spark.apache.org/) et nécessitent en plus
`pyspark`. Installez-le avec l'extra `spark` :

```bash
pip install -e ".[spark]"
```

`pyspark` lui-même nécessite un environnement Java (JDK 17 ou 21 pour PySpark 4) installé et présent dans le `PATH` ; voir le
[guide d'installation de PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
si `pyspark` ne parvient pas à démarrer un `SparkContext`.

### Exécution des tests

```bash
pip install -e ".[spark,dev]"
pytest
```

Les tests qui exécutent les algorithmes sur un vrai Spark local (y compris le driver de streaming FIMoTS) nécessitent un JDK ; ils sont
ignorés s'il n'y en a pas.

Les tests des algorithmes basés sur Spark s'exécutent contre `tests/spark_stub.py`, une doublure locale de
la petite partie de l'API RDD/broadcast de Spark utilisée par ces algorithmes ; ils exercent donc exactement le même
code de production sans nécessiter de cluster réel ni de JVM.
