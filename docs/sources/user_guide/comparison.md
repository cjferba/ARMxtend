# Comparison with mlxtend

`benchmarks/compare_mlxtend.py` mines the same synthetic baskets ([`make_transactions`](datasets.md), 60 items, average length
7) with ARMxtend and with [mlxtend](https://rasbt.github.io/mlxtend/), checks that the **itemsets and rules are identical**,
and times them (best of 3, one machine, `min_supp=0.05`, `min_conf=0.7`; seconds):

| transactions | itemsets | rules | ARMxtend FP-Growth | mlxtend FP-Growth | mlxtend Apriori | ARMxtend rules | mlxtend rules |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 000 | 537 | 716 | 0.058 | 0.017 | 0.004 | 0.014 | 0.004 |
| 10 000 | 508 | 655 | 0.262 | 0.058 | 0.009 | 0.013 | 0.003 |
| 50 000 | 515 | 679 | 1.368 | 0.266 | 0.037 | 0.013 | 0.003 |

In every case both libraries return the same itemsets (same supports) and the same rules (same confidences).

## How to read it

- **For plain frequent itemsets and rules, mlxtend is faster.** ARMxtend's FP-Growth is pure Python and about 4 to 5 times
  slower than mlxtend's, and mlxtend's Apriori (dense numpy bit operations) is far ahead on this narrow, dense table. ARMxtend's
  rule generation is about 4 times slower too. If that is all you need and it is on one machine, use mlxtend.
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
