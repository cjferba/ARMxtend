# -*- coding: utf-8 -*-
"""
Compares the fuzzy itemset miners of ARMxtend on one machine: checks that ``FFIM.fuzzy_fpgrowth`` and
``FIM.dense_fuzzy_itemsets`` return the SAME itemsets and bit-lists, and times both.

    python benchmarks/fuzzy_dense.py --sizes 2000 5000 --items 30 200 --min-supp 0.03 --alpha 10

Each time is the best of `--repeat` runs. Synthetic data: baskets from ``datasets.make_transactions`` with a
Beta(2, 2) membership degree for every item.
"""
import argparse
import time

import numpy as np

from ARMxtend.datasets import make_transactions
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM import dense_fuzzy_itemsets


def best(function, repeat):
    times, result = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return min(times), result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000])
    parser.add_argument("--items", type=int, nargs="+", default=[30, 200])
    parser.add_argument("--min-supp", type=float, default=0.03)
    parser.add_argument("--alpha", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print("{:>8} {:>6} {:>9} {:>10} {:>10} {:>8} {:>6}".format(
        "trans.", "items", "itemsets", "fpgrowth", "dense", "speedup", "same"))
    for size in args.sizes:
        for items in args.items:
            rng = np.random.default_rng(args.seed)
            baskets = make_transactions(size, items, avg_length=8, random_state=args.seed)
            transactions = [[(item, round(float(rng.beta(2, 2)), 3)) for item in basket] for basket in baskets]
            t_fp, mined_fp = best(lambda: fuzzy_fpgrowth(transactions, args.min_supp, args.alpha), args.repeat)
            t_dense, mined_dense = best(lambda: dense_fuzzy_itemsets(transactions, args.min_supp, args.alpha),
                                        args.repeat)
            same = set(mined_fp) == set(mined_dense) and all(np.allclose(mined_fp[k], mined_dense[k])
                                                              for k in mined_fp)
            print("{:>8} {:>6} {:>9} {:>9.3f}s {:>9.3f}s {:>7.0f}x {:>6}".format(
                size, items, len(mined_fp), t_fp, t_dense, t_fp / max(t_dense, 1e-9), "yes" if same else "NO"))


if __name__ == "__main__":
    main()
