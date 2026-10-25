# ARMxtend

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
[![CI](https://github.com/cjferba/ARMxtend/actions/workflows/ci.yml/badge.svg)](https://github.com/cjferba/ARMxtend/actions/workflows/ci.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

**ARMxtend (association rule mining extensions) is a Python library of useful tools for the association rule mining.**

<br>

Carlos Fernandez-Basso  2015-2026

<br>

## Links

- **Documentation:** [https://cjferba.github.io/ARMxtend](https://cjferba.github.io/ARMxtend)
- **Examples:** [https://cjferba.github.io/ARMxtend/examples](https://cjferba.github.io/ARMxtend/examples/) -- runnable, CI-tested scripts for every module (see [`examples/`](examples/))
- **Applications:** [https://cjferba.github.io/ARMxtend/applications](https://cjferba.github.io/ARMxtend/applications/) -- published studies (healthcare, energy efficiency of buildings, law enforcement) that use these methods
- **Citing ARMxtend:** [https://cjferba.github.io/ARMxtend/cite](https://cjferba.github.io/ARMxtend/cite/) -- the paper behind each module, with BibTeX
- PyPI: not published yet (the release workflow and `pyproject.toml` are ready; see [CHANGELOG](CHANGELOG.md))


## Installing ARMxtend

```bash
pip install git+https://github.com/cjferba/ARMxtend.git            # nucleo: ARM, FIM.FARE, FFIM, VizARM, preprocessing
pip install "ARMxtend[spark] @ git+https://github.com/cjferba/ARMxtend.git"   # + pyspark (Big Data / streaming)
```

Desde un clon del repositorio: `pip install -e .` o `pip install -e ".[spark,dev]"` (para desarrollar).
Requiere Python >= 3.9. Instalado el paquete, tambien hay una linea de comandos (`armxtend rules datos.csv ...`,
ver la [guia](https://cjferba.github.io/ARMxtend/user_guide/cli/)).

Ejecutar los tests y las comprobaciones del CI:

```bash
pip install -e ".[spark,dev]"
pytest --cov=ARMxtend     # tests + cobertura (el CI exige >= 80 %)
ruff check . && mypy      # lint y tipos
```

Los algoritmos de Big Data se prueban de dos formas. La mayoria de tests corren contra `tests/spark_stub.py`, un
doble local de la API de Spark que no necesita cluster ni JVM. Ademas, `tests/test_spark_integration.py`,
`tests/test_fimots_streaming.py` y `tests/test_fimots_drivers.py` ejecutan los algoritmos (incluido FIMoTS tras
cada deslizamiento de ventana, contra un oraculo de fuerza bruta) sobre un Spark real en modo local; requieren un
JDK >= 17 (`JAVA_HOME`) y se omiten automaticamente si no lo hay. Sin `pyspark` se omiten tambien los tests de
Spark y el nucleo sigue funcionando.

## Features

### Mineria de itemsets frecuentes (FIM) y reglas de asociacion (ARE)

|                | Crisp                                             | Difuso (alpha-cortes)                      |
|----------------|----------------------------------------------------|---------------------------------------------|
| Secuencial     | `ARM.association_rules`, `FFIM.fpgrowth`            | `FIM.FARE`, `FFIM.fuzzy_fpgrowth`            |
| Big Data (Spark) | `FIM.apriori.DApriori`/`DAprioriTID`, `FIM.Eclat.DECLAT`, `FIM.BD_ARE` | `FIM.BD_FARE.FuzzyDAprioriTID`, `FIM.Eclat.FuzzyDECLAT` |
| Streaming (Spark) | `SFIM` (FIMoTS) + `SARE.extractAssociationRules` | -                                             |

### Otros tipos de reglas y herramientas

* `ARM.negative_rules`: reglas con **items ausentes** (`A -> ~B`), crisp y difusas (negacion como complemento por niveles, RL-theory)
* `ARM.exception_rules`: reglas de **excepcion** y **anomalas** (cuando falla una regla fuerte)
* `ARM.gradual_rules`: reglas **graduales** ("cuanto mas A, mas B") sobre columnas numericas
* `FIM.condensed` (itemsets **cerrados/maximales**) y `ARM.redundancy` (**poda de reglas redundantes**)
* `ARM.association_rules` / `FIM.FARE`: ademas de soporte, confianza, lift, leverage, conviccion y factor de certeza, Jaccard, cosine, Kulczynski, all/max-confidence, added value, Zhang y Yule's Q (en el caso difuso, agregadas por alpha-cortes)

### Series temporales, sensores y secuencias

* `preprocessing.timeseries`: de columnas numericas ordenadas en el tiempo a **items** (simbolos por cuantiles/uniforme/SAX, eventos, calendario, retardos, ventanas)
* `ARM.temporal_rules`: reglas **temporales con retardo** ("puerta abierta hace 1 h y es de noche -> calefaccion encendida"), crisp y difusas, y perfil de una regla por hora/dia
* `ARM.rule_drift`: **deriva de reglas** -- cuales son estables, emergentes, evanescentes o intermitentes en ventanas sucesivas
* `FIM.sequential` y `ARM.sequential_rules`: **patrones secuenciales** (PrefixSpan, tambien en Spark) y **reglas secuenciales** ("si X, despues Y")

### Meta-reglas de asociacion

* `ARM.meta_rules`: mineria de reglas de asociacion sobre reglas ya extraidas de **multiples** datasets
  (crisp `crisp_meta_association_rules` y difusa `fuzzy_meta_association_rules`, ponderada por el
  soporte/factor de certeza de cada regla primaria)

### Preprocesado y visualizacion

* `preprocessing.FuzzyLib`: fuzzificacion de atributos numericos con conjuntos triangulares (Ruspini), trapezoidales o gaussianos, a mano o **automatica** (uniforme, cuantiles o k-means), y conversion a transacciones difusas
* `VizARM.AREtoGraph`: transformacion de reglas de asociacion al grafo tipado item/rule VizARE, exportable a JGF/GraphML/DOT


## Getting Started

Check out the [Quick Start](https://cjferba.github.io/ARMxtend/quick-start/) documentation to get started.


## About

Lead Developer: [Carlos Fernandez Basso](https://github.com/cjferba)


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). See [LICENSE](LICENSE) for details.
