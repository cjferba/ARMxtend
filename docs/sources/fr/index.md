[English](../index.md) · [Español](../es/index.md) · [Français](../fr/index.md) · [Português](../pt/index.md)

## Bienvenue dans la documentation d'ARMxtend

**ARMxtend (association rule mining extensions) est une bibliothèque Python pour l'extraction de règles
d'association, classiques (crisp) et floues, sur une seule machine comme sur Apache Spark pour le Big Data et le
streaming, ainsi que de méta-règles d'association pour résumer les règles trouvées dans plusieurs jeux de données.**

[![DOI](https://img.shields.io/badge/KNOSYS-2018.09.026-green)](https://doi.org/10.1016/j.knosys.2018.09.026)

<hr>

## Liens

- **Documentation :** [http://cjferba.github.io/armxtend](http://cjferba.github.io/ARMxtend)
- Dépôt du code source : [https://github.com/cjferba/armxtend](https://github.com/cjferba/ARMxtend)

<hr>

## Fonctionnalités

|                    | Classique (crisp)                                       | Flou (alpha-coupes)                                        |
|--------------------|----------------------------------------------------------|------------------------------------------------------------|
| Une seule machine  | `ARM.association_rules`, `FFIM.fpgrowth`                  | `FIM.FARE`, `FFIM.fuzzy_fpgrowth`                           |
| Big Data (Spark)   | `FIM.apriori` (DApriori/DAprioriTID), `FIM.Eclat.DECLAT`, `FIM.BD_ARE` | `FIM.BD_FARE.FuzzyDAprioriTID`, `FIM.Eclat.FuzzyDECLAT`     |
| Streaming (Spark)  | `SFIM` (FIMoTS) + `SARE.extractAssociationRules`          | -                                                            |
| Méta-règles (règles sur les règles, entre plusieurs jeux de données) | `ARM.meta_rules.crisp_meta_association_rules` | `ARM.meta_rules.fuzzy_meta_association_rules`               |

S'y ajoutent `preprocessing.FuzzyLib` pour fuzzifier des attributs numériques et `VizARM.AREtoGraph` pour transformer
les règles en graphe typé item/règle VizARE (JGF/GraphML/DOT).

Consultez le [Démarrage rapide](quick-start.md) pour des exemples exécutables de chacun d'eux et le
[Guide de l'utilisateur](../USER_GUIDE_INDEX.md) (en anglais) pour la référence complète de chaque module.

<hr>

## Citer

Si vous utilisez ARMxtend dans une publication scientifique, pensez à citer la recherche sous-jacente :

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

Voir [Citer ARMxtend](../cite.md) pour les références de chaque module.
