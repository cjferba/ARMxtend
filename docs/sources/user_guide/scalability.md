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
| 25 000 | 0.89 | 1.05 | 0.49 | 1.8 | 3.8 |
| 50 000 | 1.19 | 1.43 | 0.70 | 3.2 | 9.6 |
| 100 000 | 1.84 | 2.39 | 1.15 | 6.6 | 20.2 |
| 200 000 | 3.21 | 3.64 | 1.74 | - | - |
| 400 000 | 5.84 | 7.25 | 3.52 | - | - |

Fitting `t ~ N^a` gives `a = 0.69` (`DApriori`, `DAprioriTID`), `0.70` (`DECLAT`), `0.93` (`FuzzyDAprioriTID`) and
**`1.20` (`FuzzyDECLAT`)**. The crisp exponents are below 1 because each Spark job has a fixed cost of a few
tenths of a second; the last doubling (200 000 to 400 000) multiplies the time by 1.8 to 2.0, so the crisp
algorithms are **linear in the number of transactions** once the data are big enough to hide that fixed cost.
`FuzzyDAprioriTID` is also close to linear. `FuzzyDECLAT` grows faster than the data (see below), and the fuzzy
algorithms were only run up to 100 000 transactions.

## Speed-up with the number of cores (200 000 transactions)

Seconds, with speed-up `S = T1 / Tn` and efficiency `E = S / n`:

| Cores | `DApriori` | `DAprioriTID` | `DECLAT` | `FuzzyDECLAT` |
|---:|---:|---:|---:|---:|
| 1 | 14.4 | 17.9 | 3.5 | 61.1 |
| 2 | 7.4 (S 1.95, E 98 %) | 9.2 (S 1.94, E 97 %) | 2.3 (S 1.53, E 77 %) | 37.2 (S 1.64, E 82 %) |
| 4 | 4.0 (S 3.61, E 90 %) | 5.0 (S 3.57, E 89 %) | 1.7 (S 2.00, E 50 %) | 37.5 (S 1.63, E 41 %) |
| 8 | 3.4 (S 4.20, E 52 %) | 4.2 (S 4.29, E 54 %) | 1.8 (S 1.97, E 25 %) | 44.4 (S 1.38, E 17 %) |

* `DApriori` and `DAprioriTID` scale almost ideally up to 4 cores. At 8 cores the efficiency drops to about 50 %:
  eight Python workers and the driver compete for 10 cores, four of which are slow efficiency cores, and the fixed
  per-job cost is now a large share of a 3-second run. Take the 8-core figure as a property of this machine.
* `DECLAT` is the fastest algorithm on one core but stops improving after 2 to 4 cores, because the TID-lists are
  collected and processed in the driver (see below).
* `FuzzyDECLAT` **does not scale with more cores** and is even slower with 8 than with 2.
  `FuzzyDAprioriTID` was not part of this experiment.

## Memory of the driver

`DECLAT` and `FuzzyDECLAT` build the TID-lists with `groupByKey().collectAsMap()`, so **all of them end up in the
driver process**, and then broadcast them. This is the algorithm's design (the DECLAT paper already notes it is the
most memory-hungry), and it sets the size of the data they can handle by the memory of the driver, not by the size
of the cluster. Peak resident memory of the Python driver process at 400 000 transactions, each algorithm measured
in a fresh process:

| Algorithm | Driver memory before / peak |
|---|---|
| `DApriori` | 120 MB / 121 MB |
| `DECLAT` | 120 MB / **402 MB** (about 0.7 KB per transaction) |

`FuzzyDECLAT` needs much more because every TID carries a vector with one value per alpha-cut: the peak resident
memory of the driver process was 411, 708 and 1 303 MB at 25 000, 50 000 and 100 000 transactions (roughly 12 MB per
1 000 transactions with 5 alpha-cuts; these peaks come from the general benchmark run, where the process had
already executed the smaller sizes, so they are upper bounds). That vector is also shipped to every worker in a broadcast and its supports are computed with Python loops, which
explains the super-linear time and the lack of speed-up.

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
  was close to ideal. `DECLAT` is faster for moderate data, as long as all the TID-lists fit in the driver.
* For big **fuzzy** data use `FuzzyDAprioriTID`. Avoid `FuzzyDECLAT` beyond a few tens of thousands of transactions.
* A low `min_supp` multiplies the number of frequent itemsets and hence the work, much more than extra items do.
* Give the driver enough memory (`spark.driver.memory`) when you use the Eclat variants.

For the streaming algorithm (FIMoTS) the relevant guarantee is correctness rather than speed: after every window
slide its result is checked against a brute-force count on a real Spark (see
[SFIM / SARE](SARE_SFIM.md)). Its throughput has not been benchmarked.
