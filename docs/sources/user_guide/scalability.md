# Scalability of the Spark algorithms

This page reports how the Big Data algorithms (`FIM.apriori.DApriori`, `DAprioriTID`, `FIM.Eclat.DECLAT`,
`FIM.BD_FARE.FuzzyDAprioriTID`, `FIM.Eclat.FuzzyDECLAT`) behave on a **real Spark** as the data, the number of cores
and the number of items grow. The measurements come from
[`benchmarks/spark_scalability.py`](https://github.com/cjferba/ARMxtend/blob/master/benchmarks/spark_scalability.py),
which you can re-run on your own machine or cluster.

> **What these numbers are, and are not.** Everything below was measured on **one machine** in Spark's `local[n]`
> mode. It shows how the work is spread over cores and how the cost grows with the data. It says nothing about
> network cost, shuffles between nodes, or HDFS/S3 reads in a real cluster: ARMxtend has not been measured on a
> cluster yet. The data are synthetic and each number is the best of two runs on a machine that was doing nothing
> else; expect run-to-run variation of up to about 30 %.

## Set-up

| | |
|---|---|
| Machine | Apple M2 Pro, 10 cores (6 performance + 4 efficiency), 16 GB RAM, macOS |
| Software | Python 3.12, PySpark 4.2.0, OpenJDK 21, `spark.driver.memory=6g` |
| Data | synthetic baskets of about 10 items drawn with Zipf-like popularity, written as 8 text files and read with `sc.textFile` |
| Mining | `min_supp = 0.05`, 100 distinct items unless stated otherwise (about 250 frequent itemsets); fuzzy runs use 5 alpha-cuts |

```bash
export JAVA_HOME=...   # a JDK 17 or 21
python benchmarks/spark_scalability.py all --output results.json
```

## Growth with the number of transactions (8 cores)

Seconds per run:

| Transactions | `DApriori` | `DAprioriTID` | `DECLAT` | `FuzzyDAprioriTID` | `FuzzyDECLAT` |
|---:|---:|---:|---:|---:|---:|
| 25 000 | 1.08 | 1.28 | 0.49 | 2.1 | 0.44 |
| 50 000 | 1.37 | 1.62 | 0.47 | 3.9 | 0.61 |
| 100 000 | 2.20 | 2.66 | 0.67 | 7.3 | 0.81 |
| 200 000 | 3.57 | 4.70 | 0.73 | 14.2 | 1.14 |
| 400 000 | 6.20 | 8.70 | 1.17 | 27.5 | 2.18 |

Fitting `t ~ N^a` gives `a = 0.64` (`DApriori`), `0.71` (`DAprioriTID`), `0.32` (`DECLAT`), `0.93` (`FuzzyDAprioriTID`) and
`0.55` (`FuzzyDECLAT`). Exponents below 1 mostly reflect the fixed cost of a few tenths of a second per Spark job; at these
sizes the Eclat variants are dominated by it. `FuzzyDAprioriTID` is the only one close to linear.

> **Before and after.** Earlier versions of `DECLAT` and `FuzzyDECLAT` kept one Python dictionary and one vector per TID and
> counted candidates with Python loops: `FuzzyDECLAT` took 19.6 s at 100 000 transactions (exponent 1.20, no speed-up with
> cores) and 1.3 GB of driver memory. They now store each item's TID-list as numpy arrays (sorted TIDs and, for the fuzzy
> case, the degrees), intersect them with vectorised searches and broadcast them once instead of once per level. The result
> is bit-identical (checked against the previous implementation and FP-Growth); `FuzzyDECLAT` takes 0.81 s at 100 000 and
> `DECLAT` 0.67 s (1.07 s before).

## Speed-up with the number of cores (200 000 transactions)

Seconds, with speed-up `S = T1 / Tn` and efficiency `E = S / n`:

| Cores | `DApriori` | `DAprioriTID` | `DECLAT` | `FuzzyDECLAT` |
|---:|---:|---:|---:|---:|
| 1 | 14.5 | 18.2 | 2.0 | 3.1 |
| 2 | 7.6 (S 1.91, E 95 %) | 9.5 (S 1.91, E 96 %) | 1.1 (S 1.72, E 86 %) | 1.7 (S 1.78, E 89 %) |
| 4 | 4.2 (S 3.48, E 87 %) | 5.2 (S 3.52, E 88 %) | 0.8 (S 2.54, E 63 %) | 1.0 (S 2.94, E 74 %) |
| 8 | 3.5 (S 4.18, E 52 %) | 4.3 (S 4.27, E 53 %) | 0.8 (S 2.36, E 29 %) | 1.1 (S 2.93, E 37 %) |

* `DApriori` and `DAprioriTID` scale almost ideally up to 4 cores. At 8 cores eight Python workers and the driver compete
  for 10 cores, four of them slow efficiency cores, so take the 8-core figure as a property of this machine.
* `DECLAT` and `FuzzyDECLAT` are now the fastest, but stop improving after about 4 cores: building and collecting the
  TID-lists is a fixed, mostly serial part.

## Memory of the driver

`DECLAT` and `FuzzyDECLAT` build the TID-lists in the executors, **collect them in the driver and broadcast them**. That is
the design of the algorithm (the DECLAT paper already notes it is the most memory-hungry) and it still bounds the data they
can handle by the memory of the driver. What changed is the size of the representation: 4 bytes per item occurrence
(8 more in the fuzzy case) instead of Python objects. Growth of the driver's resident memory at 100 000 transactions, in a
fresh process:

| Algorithm | Before | Now |
|---|---:|---:|
| `DECLAT` | +63 MB | +8 MB |
| `FuzzyDECLAT` | +1 083 MB | +23 MB |

The vertical database is of the order of the data themselves (about 12 bytes per occurrence in the fuzzy case), so the
limit is the memory of the driver, not an overhead of 100 times the data. Beyond that, use `DApriori` / `DAprioriTID`.

## Sequences (PrefixSpan)

`FIM.sequential.prefixspan` (pure Python, one machine) against `prefixspan_spark` (MLlib) on synthetic sequences of about 6
elements over 60 items, `min_supp = 0.05` (about 105 patterns). Seconds, 8 cores:

| Sequences | `prefixspan` | Spark, list | Spark, RDD | Spark, DataFrame |
|---:|---:|---:|---:|---:|
| 10 000 | 0.23 | 0.71 | 0.57 | 0.67 |
| 40 000 | 1.00 | 1.16 | 1.04 | 1.11 |
| 160 000 | 4.54 | 4.04 | 2.95 | 3.45 |
| 640 000 | 22.4 (+1 167 MB) | 18.6 | 10.3 | 11.1 |

On one machine the local implementation wins below about 100 000 sequences and Spark starts to pay off above that, mainly
with an RDD or DataFrame input, which `prefixspan_spark` no longer collects in the driver. The local version needs about
1.8 MB of memory per 1 000 sequences; Spark is the way out when the sequences do not fit in one machine. As with everything
here, this is `local[n]`, not a cluster.

## Number of items and caching of the input

* **Distinct items** (100 000 transactions, 50 to 400 items): the time *decreases* (for `DApriori`, 2.3 s with 50
  items down to 1.0 s with 400). The cost is driven by the number of frequent itemsets (466 with 50 items, 108 with
  400 at this `min_supp`), not by the number of distinct items.
* **Caching the input RDD** (`rdd.cache()` before the call): no measurable difference in local mode (for example
  `DApriori`, 1.64 s without and 1.74 s with caching). Reading a local text file is cheap, so this experiment cannot
  tell whether it helps when the input comes from HDFS or S3. The algorithms make several passes over the RDD, so
  caching a slow-to-read input is the usual Spark advice, but that benefit has not been measured here and the
  algorithms do not cache anything themselves.

## Recommendations

* For big **crisp** data use `DApriori` or `DAprioriTID`: they are linear in the data and the only ones whose speed-up
  was close to ideal. `DECLAT` is faster, as long as all the TID-lists fit in the driver.
* For fuzzy data `FuzzyDECLAT` is now much faster than `FuzzyDAprioriTID` (2.2 s against 27.5 s at 400 000 transactions) with
  the same driver-memory condition; `FuzzyDAprioriTID` does not need to hold the data in the driver.
* For sequences use the local `prefixspan` unless they do not fit in memory.
* A low `min_supp` multiplies the number of frequent itemsets and hence the work, much more than extra items do.
* Give the driver enough memory (`spark.driver.memory`) when you use the Eclat variants.

For the streaming algorithm (FIMoTS) the relevant guarantee is correctness rather than speed: after every window
slide its result is checked against a brute-force count on a real Spark (see
[SFIM / SARE](SARE_SFIM.md)). Its throughput has not been benchmarked.
