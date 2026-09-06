
## Welcome to ARMxtend's documentation!

**ARMxtend (association rule mining extensions) is a Python library for mining association rules,
covering the crisp and fuzzy cases, on a single machine as well as on Apache Spark for Big Data and
streaming scenarios, plus meta-association rules for summarizing rules found across multiple datasets.**


[![DOI](https://img.shields.io/badge/KNOSYS-2018.09.026-green)](https://doi.org/10.1016/j.knosys.2018.09.026)
![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)


<hr>

## Links

- **Documentation:** [http://cjferba.github.io/armxtend](http://cjferba.github.io/ARMxtend)
- Source code repository: [https://github.com/cjferba/armxtend](https://github.com/cjferba/ARMxtend)

<hr>

## Features

|                    | Crisp                                                   | Fuzzy (alpha-cuts)                                       |
|--------------------|----------------------------------------------------------|------------------------------------------------------------|
| Single machine     | `ARM.association_rules`, `FFIM.fpgrowth`                  | `FIM.FARE`, `FFIM.fuzzy_fpgrowth`                           |
| Big Data (Spark)   | `FIM.apriori` (DApriori/DAprioriTID), `FIM.Eclat.DECLAT`, `FIM.BD_ARE` | `FIM.BD_FARE.FuzzyDAprioriTID`, `FIM.Eclat.FuzzyDECLAT`     |
| Streaming (Spark)  | `SFIM` (FIMoTS) + `SARE.extractAssociationRules`          | -                                                            |
| Meta-rules (rules about rules, across multiple datasets) | `ARM.meta_rules.crisp_meta_association_rules` | `ARM.meta_rules.fuzzy_meta_association_rules`               |

Plus `preprocessing.FuzzyLib` for fuzzifying numeric attributes and `VizARM.AREtoGraph` for
transforming rules into the VizARE typed item/rule graph (JGF/GraphML/DOT).

See the [Quick Start](quick-start.md) for runnable examples of every one of these, and the
[User Guide](USER_GUIDE_INDEX.md) for the full reference of each module.

<hr>

## Citing

If you use ARMxtend as part of your workflow in a scientific publication, please consider citing the
underlying research:

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

See [Citing ARMxtend](cite.md) for the references behind each individual module (crisp/fuzzy Big Data
algorithms, fuzzy association rules, meta-association rules).
