# -*- coding: utf-8 -*-
"""Tests of the dense fuzzy itemset miner (FIM.dense_fuzzy_itemsets)."""
import itertools

import numpy as np
import pytest

from ARMxtend.cli import main
from ARMxtend.datasets import make_transactions
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM import dense_fuzzy_itemsets
from ARMxtend.FIM._shared import alpha_levels, alpha_weights


def fuzzy_baskets(count, items, seed):
    rng = np.random.default_rng(seed)
    baskets = make_transactions(count, items, avg_length=5, random_state=seed)
    return [[(item, float(np.round(rng.beta(2, 2), 3))) for item in basket] for basket in baskets]


def brute_force(transactions, min_supp, num_alpha):
    """FSupp straight from the definition: min t-norm inside a transaction, mean over the alpha-cuts."""
    levels, weights = alpha_levels(num_alpha), alpha_weights(alpha_levels(num_alpha))
    degrees = [dict(transaction) for transaction in transactions]
    items = sorted({item for transaction in degrees for item in transaction})
    found = {}
    for size in range(1, 4):
        for itemset in itertools.combinations(items, size):
            vector = np.zeros(num_alpha)
            for transaction in degrees:
                if all(item in transaction for item in itemset):
                    vector += (min(transaction[item] for item in itemset) >= levels)
            vector /= len(transactions)
            if vector @ weights >= min_supp - 1e-12:
                found["-".join(itemset)] = vector
    return found


def same(left, right):
    return set(left) == set(right) and all(np.allclose(left[key], right[key]) for key in left)


@pytest.mark.parametrize("seed,num_alpha,min_supp", [(0, 4, 0.1), (1, 10, 0.05), (2, 2, 0.03)])
def test_matches_fuzzy_fpgrowth_and_the_definition(seed, num_alpha, min_supp):
    transactions = fuzzy_baskets(150, 10, seed)
    dense = dense_fuzzy_itemsets(transactions, min_supp, num_alpha)
    assert dense and same(dense, fuzzy_fpgrowth(transactions, min_supp, num_alpha))
    brute = brute_force(transactions, min_supp, num_alpha)
    assert same({k: v for k, v in dense.items() if k.count("-") < 3}, brute)


def test_edge_cases_and_validation():
    assert dense_fuzzy_itemsets([], 0.5, 4) == {} and dense_fuzzy_itemsets([[], []], 0.5, 4) == {}
    assert dense_fuzzy_itemsets([[("a", 0.0)], [("a", 0.0)]], 0.5, 4) == {}
    only = dense_fuzzy_itemsets([[("a", 1.0), ("b", 0.5)], [("a", 1.0)]], 0.25, 2)
    assert set(only) == {"a", "b", "a-b"} and np.allclose(only["a-b"], [0.0, 0.5])
    with pytest.raises(ValueError):
        dense_fuzzy_itemsets([[("a", 0.5)]], 0.0, 4)
    with pytest.raises(ValueError):
        dense_fuzzy_itemsets([[("a", 1.5)]], 0.5, 4)
    with pytest.raises(ValueError):
        dense_fuzzy_itemsets([[("a", 0.5)]], 0.5, 0)


def test_cli_dense_algorithm_gives_the_same_rules(tmp_path, capsys):
    rng = np.random.default_rng(3)
    path = tmp_path / "numeric.csv"
    base = rng.normal(size=300)
    path.write_text("a,b,c\n" + "\n".join("{:.4f},{:.4f},{:.4f}".format(x, x + rng.normal(scale=0.3), rng.normal())
                                          for x in base))
    outputs = []
    for algorithm in ("fpgrowth", "dense"):
        assert main(["fuzzy-rules", str(path), "--min-supp", "0.1", "--min-threshold", "0.6", "--format", "csv",
                     "--algorithm", algorithm]) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] and outputs[0].count("\n") > 1
