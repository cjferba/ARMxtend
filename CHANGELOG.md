# Changelog

All notable changes to ARMxtend. The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project uses [semantic versioning](https://semver.org/).

## [0.1.0] - unreleased

First version prepared for publication on PyPI (until then: `pip install git+https://github.com/cjferba/ARMxtend.git`).

### Added
- **Rules:** rules with absent items, crisp and fuzzy (`ARM.negative_rules`); exception and anomalous rules
  (`ARM.exception_rules`); gradual rules (`ARM.gradual_rules`); closed and maximal itemsets (`FIM.condensed`);
  redundant-rule pruning (`ARM.redundancy`); meta-association rules (`ARM.meta_rules`).
- **Time series, sensors and sequences:** `preprocessing.timeseries` (symbolic discretization -- quantile,
  uniform, SAX-like --, event and calendar items, lagged items, window items); lagged (inter-transaction) rules,
  crisp and fuzzy, and per-group rule profiles (`ARM.temporal_rules`); rule drift over time windows with
  stable / emerging / vanishing / intermittent status (`ARM.rule_drift`); PrefixSpan sequential patterns, with
  a Spark MLlib version (`FIM.sequential`); sequential rules (`ARM.sequential_rules`).
- **Measures:** Jaccard, cosine, Kulczynski, all/max confidence, added value, Zhang's metric and Yule's Q in
  `ARM.association_rules` and, aggregated level by level (RL-theory), in `FIM.FARE`.
- **Fuzzification:** trapezoidal and Gaussian sets, automatic partitions (uniform, quantile, k-means) and
  `GetFuzzyTransactions` in `preprocessing.FuzzyLib`.
- **VizARM:** the VizARE typed item/rule graph (Fernandez-Basso et al., 2026), JSON Graph Format export, rule
  summaries, a self-contained interactive HTML viewer (`AREtoGraph.to_html`) and Neo4j/MongoDB exports
  (`to_cypher`, `to_documents`).
- **Command line:** `armxtend rules | fuzzy-rules | gradual-rules | temporal-rules | rule-drift | sequences |
  sequential-rules` (also `python -m ARMxtend`).
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
