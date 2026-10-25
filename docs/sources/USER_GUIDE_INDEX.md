# User Guide Index

## Crisp association rule mining (single machine)
- [ARM.association_rules](user_guide/ARM/association_rules.md) (including the additional interest measures)
- [FFIM.fpgrowth](user_guide/FFIM.md)
- [FIM.condensed (closed and maximal itemsets)](user_guide/FIM/condensed.md)
- [ARM.redundancy (pruning redundant rules)](user_guide/ARM/redundancy.md)

## Other kinds of rules (single machine)
- [ARM.negative_rules (rules with absent items, crisp and fuzzy)](user_guide/ARM/negative_rules.md)
- [ARM.exception_rules (exception and anomalous rules)](user_guide/ARM/exception_rules.md)
- [ARM.gradual_rules ("the more A, the more B")](user_guide/ARM/gradual_rules.md)

## Time series, sensors and sequences (single machine)
- [preprocessing.timeseries (symbols, events, calendar items, lags)](user_guide/preprocessing/timeseries.md)
- [ARM.temporal_rules (rules with a delay, crisp and fuzzy; rule profiles by hour/day)](user_guide/ARM/temporal_rules.md)
- [ARM.rule_drift (emerging, stable and vanishing rules over time windows)](user_guide/ARM/rule_drift.md)
- [FIM.sequential (frequent sequential patterns, PrefixSpan; Spark version)](user_guide/FIM/sequential.md)
- [ARM.sequential_rules ("if X, then later Y")](user_guide/ARM/sequential_rules.md)

## Fuzzy association rule mining (single machine)
- [FIM.FARE](user_guide/FIM/FARE.md)
- [FFIM.fuzzy_fpgrowth](user_guide/FFIM.md)

## Crisp frequent itemsets and association rules (Big Data, Spark)
- [FIM.apriori (DApriori / DAprioriTID)](user_guide/FIM/apriori.md)
- [FIM.Eclat.DECLAT](user_guide/FIM/Eclat.md)
- [FIM.BD_ARE](user_guide/FIM/BD_ARE.md)

## Fuzzy frequent itemsets (Big Data, Spark)
- [FIM.BD_FARE.FuzzyDAprioriTID](user_guide/FIM/BD_FARE.md)
- [FIM.Eclat.FuzzyDECLAT](user_guide/FIM/Eclat.md)

## Streaming frequent itemsets and association rules (Spark)
- [SFIM / SARE (FIMoTS)](user_guide/SARE_SFIM.md)

## Scalability
- [Spark benchmarks: data, cores, memory](user_guide/scalability.md)

## Meta-association rules
- [ARM.meta_rules](user_guide/ARM/meta_rules.md)

## Command line
- [armxtend (rules, fuzzy-rules, gradual-rules, temporal-rules, rule-drift, sequences, sequential-rules)](user_guide/cli.md)

## Visualization
- [VizARM.AREtoGraph](user_guide/VizARM.md)

## Preprocessing
- [preprocessing.FuzzyLib](user_guide/preprocessing/FuzzyLib.md)
