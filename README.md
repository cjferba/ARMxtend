# ARMxtend

![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)

**ARMxtend (association rule mining extensions) is a Python library of useful tools for the association rule mining.**

<br>

Carlos Fernandez-Basso  2015-2020

<br>

## Links

- **Documentation:** [http://cjferba.github.io/armxtend](http://cjferba.github.io/ARMxtend)
- PyPI: 


## Installing ARMxtend

```bash
pip install -e .            # nucleo: ARM, FIM.FARE, FFIM, VizARM, preprocessing
pip install -e ".[spark]"   # + pyspark, necesario para los algoritmos de Big Data/streaming
```

Ejecutar los tests: `pip install pytest && pytest tests/ ARMxtend/FIM/BD_ARE/TestFunctions.py`
(los tests de los algoritmos de Spark corren contra `tests/spark_stub.py`, un doble local
de la API de Spark, por lo que no requieren un cluster ni una JVM instalados).

## Features

### Mineria de itemsets frecuentes (FIM) y reglas de asociacion (ARE)

|                | Crisp                                             | Difuso (alpha-cortes)                      |
|----------------|----------------------------------------------------|---------------------------------------------|
| Secuencial     | `ARM.association_rules`, `FFIM.fpgrowth`            | `FIM.FARE`, `FFIM.fuzzy_fpgrowth`            |
| Big Data (Spark) | `FIM.apriori.DApriori`/`DAprioriTID`, `FIM.Eclat.DECLAT`, `FIM.BD_ARE` | `FIM.BD_FARE.FuzzyDAprioriTID`, `FIM.Eclat.FuzzyDECLAT` |
| Streaming (Spark) | `SFIM` (FIMoTS) + `SARE.extractAssociationRules` | -                                             |

### Preprocesado y visualizacion

* `preprocessing.FuzzyLib`: fuzzificacion de atributos numericos (particion triangular de Ruspini)
* `VizARM.AREtoGraph`: exportacion de reglas de asociacion a un grafo (GraphML/DOT)


## Getting Started

Check out the [Quick Start](http://cjferba.github.io/armxtend) documentation to get started.


## About Omniboard

Lead Developer: [Carlos Fernandez Basso](https://github.com/cjferba)


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). See [LICENSE](LICENSE) for details.
