[English](../index.md) · [Español](../es/index.md) · [Français](../fr/index.md) · [Português](../pt/index.md)

## Bienvenido a la documentación de ARMxtend

**ARMxtend (association rule mining extensions) es una biblioteca de Python para la extracción de reglas de
asociación, tanto nítidas (crisp) como difusas, en una sola máquina y en Apache Spark para escenarios de Big Data y
streaming, además de meta-reglas de asociación para resumir las reglas halladas en varios conjuntos de datos.**

[![DOI](https://img.shields.io/badge/KNOSYS-2018.09.026-green)](https://doi.org/10.1016/j.knosys.2018.09.026)

<hr>

## Enlaces

- **Documentación:** [http://cjferba.github.io/armxtend](http://cjferba.github.io/ARMxtend)
- Repositorio del código fuente: [https://github.com/cjferba/armxtend](https://github.com/cjferba/ARMxtend)

<hr>

## Características

|                    | Nítido (crisp)                                          | Difuso (alfa-cortes)                                       |
|--------------------|----------------------------------------------------------|------------------------------------------------------------|
| Una sola máquina   | `ARM.association_rules`, `FFIM.fpgrowth`                  | `FIM.FARE`, `FFIM.fuzzy_fpgrowth`                           |
| Big Data (Spark)   | `FIM.apriori` (DApriori/DAprioriTID), `FIM.Eclat.DECLAT`, `FIM.BD_ARE` | `FIM.BD_FARE.FuzzyDAprioriTID`, `FIM.Eclat.FuzzyDECLAT`     |
| Streaming (Spark)  | `SFIM` (FIMoTS) + `SARE.extractAssociationRules`          | -                                                            |
| Meta-reglas (reglas sobre reglas, entre varios conjuntos de datos) | `ARM.meta_rules.crisp_meta_association_rules` | `ARM.meta_rules.fuzzy_meta_association_rules`               |

Además, `preprocessing.FuzzyLib` para fuzzificar atributos numéricos y `VizARM.AREtoGraph` para transformar reglas
en el grafo tipado ítem/regla de VizARE (JGF/GraphML/DOT).

Consulte la [Guía rápida](quick-start.md) para ver ejemplos ejecutables de cada uno de ellos y la
[Guía de usuario](../USER_GUIDE_INDEX.md) (en inglés) para la referencia completa de cada módulo.

<hr>

## Cómo citar

Si utiliza ARMxtend en una publicación científica, considere citar la investigación en la que se basa:

[![DOI](https://img.shields.io/badge/KNOSYS-2018.09.026-green)](https://doi.org/10.1016/j.knosys.2018.09.026)

```
@article{fernandez2019finding,
  title={Finding tendencies in streaming data using big data frequent itemset mining},
  author={Fernandez-Basso, Carlos and Francisco-Agra, Abel J and Martin-Bautista, Maria J and Ruiz, M Dolores},
  journal={Knowledge-Based Systems},
  volume={163},
  pages={666--674},
  year={2019},
  publisher={Elsevier}
}
```

Consulte [Cómo citar ARMxtend](../cite.md) para las referencias de cada módulo.
