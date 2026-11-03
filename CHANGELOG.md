# Changelog

All notable changes to ARMxtend. The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project uses [semantic versioning](https://semver.org/).

## [0.1.0] - unreleased

First version prepared for publication on PyPI (until then: `pip install git+https://github.com/cjferba/ARMxtend.git`).

### Added
- **Fuzzy significance with FConf:** `fuzzy_rule_significance(..., statistic="fconf")` and `armxtend fuzzy-rules --statistic fconf` test the confidence aggregated by alpha-cuts.
- **Rule-level utility:** `high_utility_rules` adds `utility_confidence` and `consequent_utility_share` (and `min_utility_conf`).
- **Sequential rule significance:** `ARM.sequential_rule_significance` / `significant_sequential_rules` (permutation of the element order, with multiple-testing correction).
- **Documentation in several languages:** installation, quick start and home page in Spanish, French and Portuguese (`docs/sources/{es,fr,pt}`).
- **Code in English:** docstrings, comments, error messages, CLI text, tests, examples and benchmarks.
- **Rules:** rules with absent items, crisp and fuzzy (`ARM.negative_rules`); exception and anomalous rules
  (`ARM.exception_rules`); gradual rules (`ARM.gradual_rules`); closed and maximal itemsets (`FIM.condensed`);
  redundant-rule pruning (`ARM.redundancy`); meta-association rules (`ARM.meta_rules`).
- **Taxonomies and multilevel rules:** item hierarchies with several parents (`FIM.taxonomy.Taxonomy`), extended and
  rolled-up transactions, generalized itemsets that never pair an item with its ancestor and accept a minimum support
  per level, generalized rules and Srikant-Agrawal R-interesting rules (`ARM.generalized_rules`); `armxtend generalized-rules`.
- **Quantitative rules:** equal-frequency intervals merged into ranges up to a maximum support, rules over numeric and
  categorical columns with R-interesting pruning, reusable `IntervalPartition` and the partial-completeness interval count
  (`ARM.quantitative_rules`); `armxtend quantitative-rules`.
- **Top-k, rare and high-utility rules:** the `k` most frequent itemsets and rules with no support threshold, by an exact
  best-first search (`FIM.topk`, `ARM.topk_rules`); minimal rare itemsets and rules among items that are frequent apart and
  rare together (`FIM.rare`, `ARM.rare_rules`); high-utility itemsets (two-phase algorithm over quantities and unit profits)
  and rules with `utility` and `utility_share` (`FIM.utility`, `ARM.utility_rules`); `armxtend top-k-rules | rare-rules |
  high-utility-rules`.
- **Estimators and data:** `TransactionEncoder` and `RuleMiner` (fpgrowth, top-k, rare) with `fit` / `transform` / `get_params`,
  `RuleClassifier` on the same protocol (`ARMxtend.estimators`, `ARMxtend.base`); synthetic generators with planted patterns and
  four packaged example data sets (`ARMxtend.datasets`); `armxtend generate`.
- **Project:** `CITATION.cff`, an executed quick-start notebook (`examples/notebooks/quickstart.ipynb`) and a comparison with
  mlxtend (`benchmarks/compare_mlxtend.py`: identical itemsets and rules). `FFIM.fpgrowth` (equal transactions grouped, weighted conditional
  paths) and `ARM.association_rules` (measure computed once, vectorized) are 5x and 3x faster, on par with mlxtend FP-Growth.
- **Dense fuzzy miner:** `FIM.dense_fuzzy_itemsets` (level-wise Apriori over one bit vector per item and alpha-cut, pruning on the
  anti-monotone FSupp) returns exactly what `FFIM.fuzzy_fpgrowth` returns, 45x-280x faster in `benchmarks/fuzzy_dense.py`;
  `armxtend fuzzy-rules --algorithm dense`.
- **Fuzzy significance:** per-rule permutation test for fuzzy rules over scalar-cardinality confidence, corrected for multiple
  tests (`ARM.fuzzy_rule_significance`, `ARM.fuzzy_significant_rules`); `fuzzy_association_rules` annotates `n_candidates`;
  `armxtend fuzzy-rules --significance`.
- **Dense miner:** `FIM.dense_itemsets`, level-wise Apriori over packed bit vectors (about twice as fast as mlxtend's Apriori on
  a boolean table with few items), also as `RuleMiner(algorithm="dense")`.
- **Spark:** `DECLAT` / `FuzzyDECLAT` keep numpy TID-lists and broadcast them once (same results; `FuzzyDECLAT` 19.6 s to
  0.8 s and 1.1 GB to 23 MB of driver memory at 100 000 transactions); input from DataFrames and Parquet
  (`FIM.spark_io`); `prefixspan_spark` accepts DataFrames and no longer collects an RDD; sequence benchmark.
- **Time series, sensors and sequences:** `preprocessing.timeseries` (symbolic discretization -- quantile,
  uniform, SAX-like --, event and calendar items, lagged items, window items); lagged (inter-transaction) rules,
  crisp and fuzzy, and per-group rule profiles (`ARM.temporal_rules`); rule drift over time windows with
  stable / emerging / vanishing / intermittent status (`ARM.rule_drift`); PrefixSpan sequential patterns, with
  a Spark MLlib version (`FIM.sequential`); sequential rules (`ARM.sequential_rules`); fuzzy sequential patterns and
  rules, whose events carry a degree of membership, with min / product t-norms (`FIM.fuzzy_sequential`,
  `ARM.fuzzy_sequential_rules`).
- **Statistical significance** (`ARM.significance`): per-rule Fisher / chi-squared p-values, Bonferroni / Holm /
  Benjamini-Hochberg / Benjamini-Yekutieli corrections over the family of candidate rules, Wilson and bootstrap
  intervals, and minP / maxT permutation tests with circular-shift nulls for time series; `--alpha`,
  `--correction`, `--test`, `--permutations` in the command line. `association_rules` and
  `temporal_association_rules` record `n_candidates` (and `n_transactions`) in `rules.attrs`.
- **Rules in action:** anomaly detection by rule violation, with surprise-weighted scores and explanations, also for
  lagged rules (`ARM.anomaly`); forecasting of the next time step with lagged rules and of the pending events of a
  sequence with sequential rules, with evaluation against real data and a popularity baseline (`ARM.prediction`);
  class association rules and a CBA classifier with `fit` / `predict` / `explain` / `score`
  (`ARM.classification`); pruning of unproductive rules (`ARM.redundancy.prune_unproductive_rules`); and the commands
  `armxtend anomalies | forecast | next-events | classify`.
- **Measures:** Jaccard, cosine, Kulczynski, all/max confidence, added value, Zhang's metric and Yule's Q in
  `ARM.association_rules` and, aggregated level by level (RL-theory), in `FIM.FARE`.
- **Fuzzification:** trapezoidal and Gaussian sets, automatic partitions (uniform, quantile, k-means) and
  `GetFuzzyTransactions` in `preprocessing.FuzzyLib`.
- **VizARM:** the VizARE typed item/rule graph (Fernandez-Basso et al., 2026), JSON Graph Format export, rule
  summaries, a self-contained interactive HTML viewer (`AREtoGraph.to_html`) and Neo4j/MongoDB exports
  (`to_cypher`, `to_documents`).
- **Command line:** `armxtend rules | generalized-rules | quantitative-rules | top-k-rules | rare-rules | high-utility-rules | generate | fuzzy-rules | gradual-rules | temporal-rules | rule-drift | sequences |
  sequential-rules | fuzzy-sequences | fuzzy-sequential-rules | anomalies | forecast | next-events | classify` (also `python -m ARMxtend`).
- **Tooling:** `pyproject.toml` packaging, type hints and `py.typed`, ruff, mypy, coverage, a release workflow with
  PyPI trusted publishing, runnable examples tested in CI, and a benchmark of the Spark algorithms
  (`benchmarks/spark_scalability.py`).
- **Tests on a real Spark:** the Big Data algorithms, the FIMoTS algorithm after every window slide (against a
  brute-force oracle) and the `SARE.main`/`SFIM.main` drivers in TEXTFILE and STREAMING mode.

### Fixed
- **FIMoTS (streaming), incremental maintenance** returned wrong results as soon as the data changed:
  itemsets that became frequent were not added to their already-expanded left siblings (frequent itemsets went
  missing); items first seen after the initial window were never tracked; tree lookups matched node keys with
  `startswith` (confusing `milk` with `milkshake`) and pruning removed only direct children; two sibling
  itemsets becoming frequent in the same window were registered twice and counted double.
- `SARE.main`/`SFIM.main` in TEXTFILE mode failed when `filesDir` did not end in `/`.
- Certainty factor: when the antecedent has no support (e.g. at a high alpha-cut of a fuzzy rule) the value is 0,
  as in Eq. (7) of Ruiz et al. (2016), not -1; the fuzzy FCF was biased downwards.
- `ARM.association_rules` no longer fails on an empty DataFrame.
- `DApriori` read its broadcast variable on the driver, shipping the candidates inside every task.

### Changed
- `SFIM` re-exports the FIMoTS implementation of `SARE` instead of keeping an identical copy that could diverge.
- Minimum Python version is now 3.9 (the tested range is 3.9-3.12); the license is CC BY-NC 4.0.
- `setup.py` was replaced by `pyproject.toml`; the distribution name is now `ARMxtend` (it was the unedited
  placeholder `Cjferba`).
