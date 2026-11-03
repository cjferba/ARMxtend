# Comparison with mlxtend

`benchmarks/compare_mlxtend.py` mines the same synthetic baskets ([`make_transactions`](datasets.md), 60 items, average length
7) with ARMxtend and with [mlxtend](https://rasbt.github.io/mlxtend/), checks that the **itemsets and rules are identical**,
and times them (best of 3, one machine, `min_supp=0.05`, `min_conf=0.7`; seconds):

| transactions | itemsets | rules | ARMxtend FP-Growth (lists) | ARMxtend dense (table) | mlxtend FP-Growth (table) | mlxtend Apriori (table) | ARMxtend rules | mlxtend rules |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 000 | 537 | 716 | 0.015 | 0.003 | 0.016 | 0.004 | 0.004 | 0.004 |
| 10 000 | 508 | 655 | 0.052 | 0.005 | 0.054 | 0.008 | 0.004 | 0.003 |
| 50 000 | 515 | 679 | 0.251 | 0.015 | 0.246 | 0.033 | 0.004 | 0.003 |

Inputs differ as in each library's own use: ARMxtend's FP-Growth takes lists of baskets and the dense miner and mlxtend take a
boolean table (with a list as input the dense miner needs 0.1 s at 50 000 transactions, almost all of it building the
bit vectors). In every case all of them return the same itemsets (same supports) and the same rules (same confidences).
(Before the optimization of version 0.1.0, ARMxtend's FP-Growth took 0.058, 0.262 and 1.368 s and its rules 0.014, 0.013 and 0.013 s.)

## How to read it

- **FP-Growth and rules are on par with mlxtend** on this data. FP-Growth groups equal transactions and keeps each conditional
  path once with its weight; the rule measures are computed once, vectorized, over all candidate rules.
- **The dense miner (`FIM.dense_itemsets`) is about twice as fast as mlxtend's Apriori** on a boolean table with few items:
  level-wise Apriori over packed bit vectors, all the candidates of a level counted at once with numpy. It is the one to use
  with tens or hundreds of items and many transactions; with thousands of sparse items `FFIM.fpgrowth` is better. Both are
  exact and give the same result.
- **The reasons to use ARMxtend are what mlxtend does not have**: fuzzy, gradual, sequential, temporal, quantitative, rare,
  high-utility and top-k rules, significance testing, anomaly detection and forecasting with rules, and the Spark versions.
- **It says nothing about quality or about Spark.** It compares speed on one kind of synthetic data (60 items, many short
  baskets). With thousands of items or very low supports the ranking may change. The Spark scalability measurements are in
  [scalability](scalability.md).
- **SPMF is not included.** It is a Java program distributed as a `.jar` that ARMxtend does not ship; a comparison needs
  you to download it.

Run it yourself:

```bash
pip install mlxtend
python benchmarks/compare_mlxtend.py --sizes 2000 10000 50000 --items 60 --min-supp 0.05
```
