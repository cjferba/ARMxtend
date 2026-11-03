[English](../installation.md) · [Español](../es/installation.md) · [Français](../fr/installation.md) · [Português](../pt/installation.md)

# Instalação do ARMxtend

---

O ARMxtend está sendo preparado para sua primeira publicação no PyPI (veja o [registro de alterações](https://github.com/cjferba/ARMxtend/blob/master/CHANGELOG.md)).
Até lá, instale-o a partir do [repositório do GitHub](https://github.com/cjferba/ARMxtend):

```bash
pip install git+https://github.com/cjferba/ARMxtend.git
# or, to work on it, from a local checkout:
git clone https://github.com/cjferba/ARMxtend.git
cd ARMxtend
pip install -e .
```

Depois de publicado, bastará `pip install armxtend`. Requer Python 3.9 ou superior (testado em 3.9-3.12).
A instalação também fornece o [comando `armxtend`](../user_guide/cli.md).

Isto instala o núcleo da biblioteca, para uma única máquina: `ARM` (regras de associação clássicas e
meta-regras), `FIM.FARE` (regras de associação difusas), `FFIM` (FP-Growth clássico e difuso), `VizARM`
e `preprocessing`. Suas únicas dependências são `numpy`, `pandas` e `networkx`.

### Extra de Big Data / streaming

Os algoritmos de Big Data e streaming (`FIM.apriori`, `FIM.Eclat`, `FIM.BD_ARE`, `FIM.BD_FARE`,
`SARE`, `SFIM`) são executados no [Apache Spark](https://spark.apache.org/) e exigem adicionalmente
o `pyspark`. Instale-o com o extra `spark`:

```bash
pip install -e ".[spark]"
```

O próprio `pyspark` requer um ambiente de execução Java (JDK 17 ou 21 para o PySpark 4) instalado e no `PATH`; veja o
[guia de instalação do PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
se o `pyspark` não conseguir iniciar um `SparkContext`.

### Executando os testes

```bash
pip install -e ".[spark,dev]"
pytest
```

Os testes que executam os algoritmos num Spark local real (incluindo o driver de streaming do FIMoTS) precisam de um JDK; são
ignorados quando nenhum é encontrado.

Os testes dos algoritmos baseados em Spark são executados contra `tests/spark_stub.py`, um dublê local da
pequena parte da API RDD/broadcast do Spark usada por esses algoritmos, de modo que exercitam exatamente o mesmo
código de produção sem exigir um cluster real nem uma JVM.
