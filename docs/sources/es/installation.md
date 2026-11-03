[English](../installation.md) · [Español](../es/installation.md) · [Français](../fr/installation.md) · [Português](../pt/installation.md)

# Instalación de ARMxtend

---

ARMxtend se está preparando para su primera publicación en PyPI (véase el [registro de cambios](https://github.com/cjferba/ARMxtend/blob/master/CHANGELOG.md)).
Hasta entonces, instálelo desde el [repositorio de GitHub](https://github.com/cjferba/ARMxtend):

```bash
pip install git+https://github.com/cjferba/ARMxtend.git
# or, to work on it, from a local checkout:
git clone https://github.com/cjferba/ARMxtend.git
cd ARMxtend
pip install -e .
```

Una vez publicado, bastará con `pip install armxtend`. Requiere Python 3.9 o superior (probado en 3.9-3.12).
La instalación incluye también el [comando `armxtend`](../user_guide/cli.md).

Esto instala el núcleo de la biblioteca, para una sola máquina: `ARM` (reglas de asociación nítidas y
meta-reglas), `FIM.FARE` (reglas de asociación difusas), `FFIM` (FP-Growth nítido y difuso), `VizARM`
y `preprocessing`. Sus únicas dependencias son `numpy`, `pandas` y `networkx`.

### Extra para Big Data y streaming

Los algoritmos de Big Data y streaming (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`,
`SARE`, `SFIM`) se ejecutan sobre [Apache Spark](https://spark.apache.org/) y requieren además
`pyspark`. Instálelo con el extra `spark`:

```bash
pip install -e ".[spark]"
```

El propio `pyspark` necesita un entorno de ejecución de Java (JDK 17 o 21 para PySpark 4) instalado y en el `PATH`; consulte la
[guía de instalación de PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
si `pyspark` no consigue iniciar un `SparkContext`.

### Ejecución de los tests

```bash
pip install -e ".[spark,dev]"
pytest
```

Los tests que ejecutan los algoritmos sobre un Spark local real (incluido el driver de streaming de FIMoTS) necesitan un JDK; se
omiten cuando no se encuentra ninguno.

Los tests de los algoritmos basados en Spark se ejecutan contra `tests/spark_stub.py`, un doble local de
la pequeña parte de la API RDD/broadcast de Spark que usan estos algoritmos, de modo que ejercitan exactamente el mismo
código de producción sin necesitar un clúster real ni una JVM.
