[English](../index.md) · [Español](../es/index.md) · [Français](../fr/index.md) · [Português](../pt/index.md)

## Bem-vindo à documentação do ARMxtend

**O ARMxtend (association rule mining extensions) é uma biblioteca Python para mineração de regras de
associação, clássicas (crisp) e difusas, numa única máquina e no Apache Spark para cenários de Big Data e
streaming, além de meta-regras de associação para resumir as regras encontradas em vários conjuntos de dados.**

[![DOI](https://img.shields.io/badge/KNOSYS-2018.09.026-green)](https://doi.org/10.1016/j.knosys.2018.09.026)

<hr>

## Links

- **Documentação:** [http://cjferba.github.io/armxtend](http://cjferba.github.io/ARMxtend)
- Repositório do código-fonte: [https://github.com/cjferba/armxtend](https://github.com/cjferba/ARMxtend)

<hr>

## Funcionalidades

|                    | Clássico (crisp)                                        | Difuso (alfa-cortes)                                       |
|--------------------|----------------------------------------------------------|------------------------------------------------------------|
| Uma única máquina  | `ARM.association_rules`, `FFIM.fpgrowth`                  | `FIM.FARE`, `FFIM.fuzzy_fpgrowth`                           |
| Big Data (Spark)   | `FIM.apriori` (DApriori/DAprioriTID), `FIM.Eclat.DECLAT`, `FIM.BD_ARE` | `FIM.BD_FARE.FuzzyDAprioriTID`, `FIM.Eclat.FuzzyDECLAT`     |
| Streaming (Spark)  | `SFIM` (FIMoTS) + `SARE.extractAssociationRules`          | -                                                            |
| Meta-regras (regras sobre regras, entre vários conjuntos de dados) | `ARM.meta_rules.crisp_meta_association_rules` | `ARM.meta_rules.fuzzy_meta_association_rules`               |

Além disso, `preprocessing.FuzzyLib` para fuzzificar atributos numéricos e `VizARM.AREtoGraph` para transformar regras
no grafo tipado item/regra do VizARE (JGF/GraphML/DOT).

Veja o [Início rápido](quick-start.md) para exemplos executáveis de cada um deles e o
[Guia do usuário](../USER_GUIDE_INDEX.md) (em inglês) para a referência completa de cada módulo.

<hr>

## Como citar

Se você usa o ARMxtend numa publicação científica, considere citar a pesquisa em que se baseia:

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

Veja [Como citar o ARMxtend](../cite.md) para as referências de cada módulo.
