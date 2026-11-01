# -*- coding: utf-8 -*-
"""Tests de FIM.taxonomy y ARM.generalized_rules: taxonomias, itemsets generalizados y reglas R-interesantes."""
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import pytest

from ARMxtend.ARM import association_rules, generalized_rules, interesting_rules, prune_redundant_rules, significant_rules
from ARMxtend.FFIM import fpgrowth
from ARMxtend.FIM import Taxonomy, extend_transactions, generalize_transactions, generalized_itemsets

SHOP = Taxonomy.from_paths(["Clothes/Outerwear/Jacket", "Clothes/Outerwear/Ski Pants", "Clothes/Shirts",
                            "Footwear/Shoes", "Footwear/Hiking Boots"])
# El ejemplo de Srikant y Agrawal (1995)
SHOP_BASKETS = [["Shirts"], ["Jacket", "Hiking Boots"], ["Ski Pants", "Hiking Boots"], ["Shoes"], ["Shoes"], ["Jacket"]]


def random_taxonomy(seed, dag=False):
    """18 items: 12 hojas, 4 categorias y 2 raices (con `dag`, algunas hojas y categorias tienen un segundo padre)."""
    rng = np.random.RandomState(seed)
    pairs = [("m{}".format(m), "r{}".format(m % 2)) for m in range(4)]
    pairs += [("l{}".format(leaf), "m{}".format(rng.randint(4))) for leaf in range(12)]
    if dag:
        pairs += [("l{}".format(leaf), "m{}".format(rng.randint(4))) for leaf in rng.choice(12, 4, replace=False)]
        pairs += [("m0", "r1"), ("l1", "r0")]
    return Taxonomy(pairs)


def random_baskets(seed, rows=60):
    rng = np.random.RandomState(seed)
    return [["l{}".format(i) for i in np.flatnonzero(rng.rand(12) < rng.uniform(0.1, 0.45))] or ["l0"] for _ in range(rows)]


# ----------------------------------------------------------------------------- taxonomia

def test_taxonomy_structure_and_queries():
    assert SHOP.items == ["Clothes", "Footwear", "Hiking Boots", "Jacket", "Outerwear", "Shirts", "Shoes", "Ski Pants"]
    assert len(SHOP) == 8
    assert SHOP.roots() == ["Clothes", "Footwear"] and SHOP.leaves() == ["Hiking Boots", "Jacket", "Shirts", "Shoes", "Ski Pants"]
    assert SHOP.ancestors("Jacket") == {"Outerwear", "Clothes"} and SHOP.parents("Jacket") == {"Outerwear"}
    assert SHOP.descendants("Clothes") == {"Outerwear", "Shirts", "Jacket", "Ski Pants"} and SHOP.children("Clothes") == {"Outerwear", "Shirts"}
    assert [SHOP.level(i) for i in ("Clothes", "Outerwear", "Jacket")] == [1, 2, 3] and SHOP.depth == 3
    assert SHOP.is_ancestor("Clothes", "Jacket") and not SHOP.is_ancestor("Jacket", "Clothes") and not SHOP.is_ancestor("Shirts", "Shirts")
    assert "Jacket" in SHOP and "Socks" not in SHOP
    assert SHOP.level("Socks") == 1 and SHOP.ancestors("Socks") == frozenset() and SHOP.at_level("Socks", 2) == {"Socks"}
    assert "3 levels" in repr(SHOP)

    pairs = [(row.child, row.parent) for row in SHOP.to_frame().itertuples()]
    for built in (Taxonomy(pairs), Taxonomy(SHOP.to_frame()), Taxonomy({c: p for c, p in pairs})):
        assert built.to_frame().equals(SHOP.to_frame())
    assert Taxonomy({"a": ["b", "c"]}).parents("a") == {"b", "c"}


def test_taxonomy_levels_and_at_level_with_several_parents():
    tax = Taxonomy([("B", "A"), ("C", "B"), ("X", "C"), ("X", "D")])              # X: camino largo y camino corto
    assert [tax.level(i) for i in "ABCDX"] == [1, 2, 3, 1, 4]                      # el mayor camino
    assert tax.ancestors("X") == {"A", "B", "C", "D"}
    assert tax.at_level("X", 4) == {"X"} and tax.at_level("X", 3) == {"C", "D"}
    assert tax.at_level("X", 2) == {"B", "D"} and tax.at_level("X", 1) == {"A", "D"}
    assert tax.at_level("B", 5) == {"B"}


def test_taxonomy_validation():
    with pytest.raises(ValueError, match="ciclo"):
        Taxonomy([("a", "b"), ("b", "c"), ("c", "a")])
    with pytest.raises(ValueError, match="propio padre"):
        Taxonomy([("a", "a")])
    with pytest.raises(ValueError):
        Taxonomy([("a-b", "c")])
    with pytest.raises(ValueError, match="dos columnas"):
        Taxonomy(pd.DataFrame({"child": ["a"]}))


def test_extend_and_generalize_transactions():
    assert extend_transactions(SHOP_BASKETS[:2], SHOP) == [{"Shirts", "Clothes"}, {"Jacket", "Outerwear", "Clothes", "Hiking Boots", "Footwear"}]
    assert extend_transactions([["Socks"]], SHOP) == [{"Socks"}]
    assert generalize_transactions(SHOP_BASKETS[:3], SHOP, 2) == [{"Shirts"}, {"Outerwear", "Hiking Boots"}, {"Outerwear", "Hiking Boots"}]
    assert generalize_transactions(SHOP_BASKETS[:2], SHOP, 1) == [{"Clothes"}, {"Clothes", "Footwear"}]
    assert generalize_transactions(SHOP_BASKETS[:2], SHOP, 3) == [{"Shirts"}, {"Jacket", "Hiking Boots"}]
    with pytest.raises(ValueError):
        generalize_transactions([["a"]], SHOP, 0)


# ------------------------------------------------------------------- itemsets generalizados

def brute_itemsets(baskets, taxonomy, min_supp, max_len):
    """Enumeracion exhaustiva: sobre las transacciones extendidas, sin itemsets con un item y un antecesor suyo."""
    extended = extend_transactions(baskets, taxonomy)
    vocabulary = sorted(set().union(*extended))
    levels = sorted({taxonomy.level(i) for i in vocabulary})
    if isinstance(min_supp, dict):
        listed = sorted(min_supp)
        thresholds = {level: min_supp[max([k for k in listed if k <= level] or [listed[0]])] for level in levels}
    else:
        thresholds = {level: min_supp for level in levels}
    result = {}
    for size in range(1, (max_len or len(vocabulary)) + 1):
        for itemset in combinations(vocabulary, size):
            if any(taxonomy.is_ancestor(a, b) for a in itemset for b in itemset):
                continue
            support = sum(set(itemset) <= basket for basket in extended) / len(extended)
            if support >= max(thresholds[taxonomy.level(i)] for i in itemset) - 1e-9:
                result[frozenset(itemset)] = support
        if size >= 3 and not any(len(k) == size for k in result):
            break
    return result


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dag", [False, True])
@pytest.mark.parametrize("min_supp", [0.25, {1: 0.5, 2: 0.3, 3: 0.12}])
def test_generalized_itemsets_equal_the_brute_force_enumeration(seed, dag, min_supp):
    taxonomy, baskets = random_taxonomy(seed, dag), random_baskets(seed)
    found = generalized_itemsets(baskets, taxonomy, min_supp, max_len=3)
    expected = brute_itemsets(baskets, taxonomy, min_supp, 3)
    assert len(found) == len(found["itemsets"].unique()) > 3
    assert dict(zip(found["itemsets"], found["support"])) == pytest.approx(expected)
    assert found["support"].is_monotonic_decreasing and found.attrs["n_transactions"] == len(baskets)


def test_generalized_itemsets_without_taxonomy_are_the_usual_frequent_itemsets():
    baskets = random_baskets(11)
    found = generalized_itemsets(baskets, Taxonomy([]), 0.15)
    usual = fpgrowth(baskets, 0.15)
    assert {tuple(sorted(s)): v for s, v in zip(found["itemsets"], found["support"])} == pytest.approx(
        {tuple(sorted(k.split("-"))): v for k, v in usual.items()})


def test_never_an_item_with_its_ancestor_and_hand_checked_supports():
    found = generalized_itemsets(SHOP_BASKETS, SHOP, 1 / 6)
    support = dict(zip(found["itemsets"], found["support"]))
    assert not any(SHOP.is_ancestor(a, b) for itemset in support for a in itemset for b in itemset)
    assert support[frozenset({"Outerwear", "Footwear"})] == pytest.approx(2 / 6)
    assert support[frozenset({"Clothes", "Hiking Boots"})] == pytest.approx(2 / 6)
    assert support[frozenset({"Outerwear"})] == pytest.approx(3 / 6)
    assert frozenset({"Jacket", "Outerwear"}) not in support and frozenset({"Hiking Boots", "Footwear"}) not in support


def test_support_per_level_uses_the_threshold_of_the_most_general_item():
    found = generalized_itemsets(SHOP_BASKETS, SHOP, {1: 0.6, 2: 0.3})
    # nivel 1 (Clothes, Footwear: 4/6) necesita 0.6; los niveles 2 y 3 (Outerwear 3/6, Hiking Boots, Shoes, Jacket: 2/6), 0.3;
    # {Outerwear, Footwear} y {Clothes, Hiking Boots} tienen 2/6 pero heredan el 0.6 de su item mas general
    assert set(found["itemsets"]) == {frozenset({"Clothes"}), frozenset({"Footwear"}), frozenset({"Outerwear"}),
                                      frozenset({"Hiking Boots"}), frozenset({"Shoes"}), frozenset({"Jacket"}),
                                      frozenset({"Outerwear", "Hiking Boots"})}


def test_generalized_itemsets_edge_cases_and_errors():
    empty = generalized_itemsets([], SHOP, 0.5)
    assert len(empty) == 0 and empty.attrs["n_transactions"] == 0
    assert len(generalized_itemsets([["Jacket"]], SHOP, 0.5, max_len=1)) == 3        # Jacket, Outerwear, Clothes
    with pytest.raises(ValueError):
        generalized_itemsets(SHOP_BASKETS, SHOP, 0)
    with pytest.raises(ValueError):
        generalized_itemsets(SHOP_BASKETS, SHOP, {1: 1.5})
    with pytest.raises(ValueError):
        generalized_itemsets(SHOP_BASKETS, SHOP, {})
    with pytest.raises(ValueError):
        generalized_itemsets(SHOP_BASKETS, SHOP, 0.3, max_len=0)


# ----------------------------------------------------------------------------------- reglas

def by_rule(frame, column):
    return {(tuple(sorted(a)), tuple(sorted(c))): value for a, c, value in zip(frame["antecedents"], frame["consequents"], frame[column])}


def test_the_running_example_of_srikant_and_agrawal():
    rules = generalized_rules(SHOP_BASKETS, SHOP, 0.3, min_threshold=0.6)
    boots, outer = ("Hiking Boots",), ("Outerwear",)
    assert by_rule(rules, "support") == pytest.approx({(outer, boots): 1 / 3, (outer, ("Footwear",)): 1 / 3,
                                                       (boots, outer): 1 / 3, (boots, ("Clothes",)): 1 / 3})
    assert by_rule(rules, "confidence") == pytest.approx({(outer, boots): 2 / 3, (outer, ("Footwear",)): 2 / 3,
                                                          (boots, outer): 1.0, (boots, ("Clothes",)): 1.0})
    assert set(by_rule(rules, "level").values()) == {2}
    assert rules.attrs["n_transactions"] == 6 and rules.attrs["n_candidates"] >= len(rules) and rules.attrs["taxonomy"] is SHOP

    kept = interesting_rules(rules, interest=1.1)
    ratio = by_rule(kept, "interest")
    assert len(kept) == 4
    assert ratio[(outer, boots)] == pytest.approx(2.0)                                  # soporte 1/3 frente al esperado 1/6
    assert ratio[(boots, outer)] == pytest.approx(4 / 3)                                # confianza 1 frente a la esperada 3/4
    assert np.isnan(ratio[(boots, ("Clothes",))]) and np.isnan(ratio[(outer, ("Footwear",))])
    assert set(by_rule(interesting_rules(rules, interest=1.5), "support")) == set(ratio) - {(boots, outer)}


def test_rules_equal_association_rules_on_the_brute_force_itemsets():
    taxonomy, baskets = random_taxonomy(2, dag=True), random_baskets(2)
    rules = generalized_rules(baskets, taxonomy, 0.2, min_threshold=0.5, max_len=3)
    frequent = brute_itemsets(baskets, taxonomy, 0.2, 3)
    expected = association_rules(pd.DataFrame({"itemsets": list(frequent), "support": list(frequent.values())}), min_threshold=0.5)
    keyed = lambda frame, column: {(a, c): v for a, c, v in zip(frame["antecedents"], frame["consequents"], frame[column])}   # noqa: E731
    assert len(rules) > 10
    for column in ("support", "confidence"):
        assert keyed(rules, column) == pytest.approx(keyed(expected, column))
    assert not any(taxonomy.is_ancestor(a, b) for x, y in zip(rules["antecedents"], rules["consequents"]) for a in x | y for b in x | y)
    assert all(level == max(taxonomy.level(i) for i in x | y) for x, y, level in zip(rules["antecedents"], rules["consequents"], rules["level"]))


def brute_interesting(rules, baskets, taxonomy, interest):
    """R-interesantes por definicion: antecesoras por biyeccion de items, soportes contados sobre las transacciones."""
    extended = extend_transactions(baskets, taxonomy)

    def sup(items):
        return sum(set(items) <= basket for basket in extended) / len(extended)

    sets = [(frozenset(a), frozenset(c)) for a, c in zip(rules["antecedents"], rules["consequents"])]

    def replaces(ancestor_side, side):             # `ancestor_side` sale de `side` sustituyendo items por si mismos o por antecesores
        return len(ancestor_side) == len(side) and any(
            all(g == i or taxonomy.is_ancestor(g, i) for g, i in zip(perm, sorted(side))) for perm in permutations(sorted(ancestor_side)))

    def is_ancestor_rule(general, specific):
        return general != specific and replaces(general[0], specific[0]) and replaces(general[1], specific[1])

    def pair_up(general_side, specific_side):       # emparejamiento item a item (el primero valido)
        for perm in permutations(sorted(general_side)):
            if all(g == i or taxonomy.is_ancestor(g, i) for g, i in zip(perm, sorted(specific_side))):
                return list(zip(perm, sorted(specific_side)))
        raise AssertionError

    memo = {}

    def is_interesting(rule):
        if rule not in memo:
            ancestors = [g for g in sets if is_ancestor_rule(g, rule) and is_interesting(g)]
            close = [g for g in ancestors if not any(is_ancestor_rule(g, other) for other in ancestors if other != g)]
            ratios = []
            for g in close:
                pairs_x, pairs_y = pair_up(g[0], rule[0]), pair_up(g[1], rule[1])
                scale = np.prod([sup([i]) / sup([a]) for a, i in pairs_x + pairs_y])
                scale_y = np.prod([sup([i]) / sup([a]) for a, i in pairs_y])
                actual_sup, actual_conf = sup(rule[0] | rule[1]), sup(rule[0] | rule[1]) / sup(rule[0])
                ratios.append(max(actual_sup / (sup(g[0] | g[1]) * scale), actual_conf / (sup(g[0] | g[1]) / sup(g[0]) * scale_y)))
            memo[rule] = (not ratios or min(ratios) >= interest - 1e-9, min(ratios) if ratios else np.nan)
        return memo[rule][0]

    return {rule: memo[rule][1] for rule in sets if is_interesting(rule)}


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("dag", [False, True])
@pytest.mark.parametrize("interest", [1.0, 1.3])
def test_interesting_rules_equal_the_brute_force_definition(seed, dag, interest):
    taxonomy, baskets = random_taxonomy(seed, dag), random_baskets(seed, rows=80)
    rules = generalized_rules(baskets, taxonomy, 0.15, min_threshold=0.45, max_len=3)
    kept = interesting_rules(rules, interest=interest)
    expected = brute_interesting(rules, baskets, taxonomy, interest)
    found = {(a, c): r for a, c, r in zip(kept["antecedents"], kept["consequents"], kept["interest"])}
    assert 0 < len(found) < len(rules) and set(found) == set(expected)
    for rule, ratio in expected.items():
        assert found[rule] == pytest.approx(ratio, nan_ok=True)


def test_generalized_rules_with_interest_downstream_and_errors():
    taxonomy, baskets = random_taxonomy(1), random_baskets(1, rows=200)
    rules = generalized_rules(baskets, taxonomy, 0.08, min_threshold=0.4, max_len=2)
    pruned = generalized_rules(baskets, taxonomy, 0.08, min_threshold=0.4, max_len=2, interest=1.2)
    assert pruned.drop(columns="interest").equals(interesting_rules(rules, interest=1.2).drop(columns="interest")) and len(pruned) < len(rules)
    assert pruned.attrs["n_transactions"] == 200 and pruned.attrs["taxonomy"] is taxonomy
    assert len(significant_rules(rules, 0.5, n_transactions=200)) <= len(rules)         # usa attrs['n_candidates']
    assert len(prune_redundant_rules(rules)) <= len(rules)

    nothing = generalized_rules(baskets, taxonomy, 0.9, interest=1.2)
    assert len(nothing) == 0 and "level" in nothing.columns
    with pytest.raises(ValueError, match="taxonomia"):
        interesting_rules(pd.DataFrame(rules.values, columns=rules.columns))           # sin attrs
    with pytest.raises(ValueError, match="interest"):
        interesting_rules(rules, interest=0.5)


def test_categories_reveal_rules_that_no_single_product_reaches():
    """Cada categoria se reparte entre 4 productos: ningun producto alcanza el soporte, la categoria si."""
    rng = np.random.RandomState(0)
    taxonomy = Taxonomy([("a{}".format(i), "A") for i in range(4)] + [("b{}".format(i), "B") for i in range(4)])
    baskets = []
    for _ in range(1000):
        if rng.rand() < 0.5:                                                  # compra de A, y casi siempre tambien de B
            basket = ["a{}".format(rng.randint(4))]
            if rng.rand() < 0.9:
                basket.append("b{}".format(rng.randint(4)))
        else:
            basket = ["b{}".format(rng.randint(4))] if rng.rand() < 0.3 else ["x"]
        baskets.append(basket)
    assert set(fpgrowth(baskets, 0.2)) == {"x"}                               # de los productos, solo 'x' es frecuente
    rules = generalized_rules(baskets, taxonomy, 0.2, min_threshold=0.8)
    assert set(by_rule(rules, "support")) == {(("A",), ("B",))}              # B => A tiene confianza 0.75
    assert by_rule(rules, "confidence")[(("A",), ("B",))] == pytest.approx(0.9, abs=0.05)
