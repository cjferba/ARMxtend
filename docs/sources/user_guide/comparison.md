# Comparison with mlxtend

`benchmarks/compare_mlxtend.py` mines the same synthetic baskets ([`make_transactions`](datasets.md), 60 items, average length
7) with ARMxtend and with [mlxtend](https://rasbt.github.io/mlxtend/), checks that the **itemsets and rules are identical**,
and times them (best of 3, one machine, `min_supp=0.05`, `min_conf=0.7`; seconds):

| transactions | itemsets | rules | ARMxtend FP-Growth | mlxtend FP-Growth | mlxtend Apriori | ARMxtend rules | mlxtend rules |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 000 | 537 | 716 | 0.015 | 0.017 | 0.004 | 0.004 | 0.004 |
| 10 000 | 508 | 655 | 0.052 | 0.057 | 0.010 | 0.004 | 0.003 |
| 50 000 | 515 | 679 | 0.266 | 0.259 | 0.035 | 0.004 | 0.003 |

In every case both libraries return the same itemsets (same supports) and the same rules (same confidences). (Before the
optimization of version 0.1.0, ARMxtend's FP-Growth took 0.058, 0.262 and 1.368 s and its rules 0.014, 0.013 and 0.013 s.)

## How to read it

- **FP-Growth and rules are on par with mlxtend** on this data. FP-Growth groups equal transactions and keeps each conditional
  path once with its weight; the rule measures are computed once, vectorized, over all candidate rules.
- **mlxtend's Apriori is still 7 to 8 times faster** here: it works on a dense boolean matrix with numpy, which suits a table
  of 60 items and short baskets. ARMxtend has no dense miner; with many more items that advantage disappears.
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
