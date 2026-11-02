# -*- coding: utf-8 -*-
"""
Compara ARMxtend con mlxtend (una sola maquina): comprueba que dan los MISMOS itemsets y reglas y mide el tiempo.

    python benchmarks/compare_mlxtend.py --sizes 2000 10000 50000 --items 60 --min-supp 0.05

Necesita ``pip install mlxtend``. Cada medida es la mejor de `--repeat` ejecuciones. No incluye SPMF (necesita Java y
su .jar, que no se distribuyen con ARMxtend).
"""
import argparse
import time

import pandas as pd

from ARMxtend.ARM import association_rules
from ARMxtend.datasets import make_transactions
from ARMxtend.estimators import TransactionEncoder
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM._shared import key_to_itemset


def best(function, repeat):
    times, result = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return min(times), result


def frame(frequent):
    return pd.DataFrame({"support": list(frequent.values()), "itemsets": [key_to_itemset(key) for key in frequent]})


def rule_keys(rules):
    return {(frozenset(a), frozenset(c)): round(conf, 9) for a, c, conf in
            zip(rules["antecedents"], rules["consequents"], rules["confidence"])}


def main(argv=None):
    from mlxtend.frequent_patterns import apriori, fpgrowth as ml_fpgrowth
    from mlxtend.frequent_patterns import association_rules as ml_rules

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000, 50000])
    parser.add_argument("--items", type=int, default=60)
    parser.add_argument("--min-supp", type=float, default=0.05)
    parser.add_argument("--min-conf", type=float, default=0.7)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = []
    for size in args.sizes:
        baskets = make_transactions(size, args.items, avg_length=7, random_state=args.seed)
        table = TransactionEncoder().fit_transform(baskets)
        t_arm, mine_arm = best(lambda: frame(fpgrowth(baskets, args.min_supp)), args.repeat)
        t_fp, mine_fp = best(lambda: ml_fpgrowth(table, min_support=args.min_supp, use_colnames=True), args.repeat)
        t_ap, mine_ap = best(lambda: apriori(table, min_support=args.min_supp, use_colnames=True), args.repeat)
        same = ({row["itemsets"]: round(row["support"], 9) for _, row in mine_arm.iterrows()}
                == {frozenset(row["itemsets"]): round(row["support"], 9) for _, row in mine_fp.iterrows()}
                == {frozenset(row["itemsets"]): round(row["support"], 9) for _, row in mine_ap.iterrows()})
        r_arm, rules_arm = best(lambda: association_rules(mine_arm, "confidence", args.min_conf), args.repeat)
        r_ml, rules_ml = best(lambda: ml_rules(mine_fp, metric="confidence", min_threshold=args.min_conf), args.repeat)
        same_rules = rule_keys(rules_arm) == rule_keys(rules_ml)
        rows.append({"transactions": size, "itemsets": len(mine_arm), "rules": len(rules_arm),
                     "ARMxtend fpgrowth (s)": round(t_arm, 3), "mlxtend fpgrowth (s)": round(t_fp, 3),
                     "mlxtend apriori (s)": round(t_ap, 3), "ARMxtend rules (s)": round(r_arm, 3),
                     "mlxtend rules (s)": round(r_ml, 3), "same itemsets": same, "same rules": same_rules})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    main()
