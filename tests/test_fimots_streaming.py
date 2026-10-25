# -*- coding: utf-8 -*-
"""
Verificacion del algoritmo FIMoTS (SARE/SFIM) sobre Spark real: tras CADA
desplazamiento de la ventana, los itemsets frecuentes que mantiene el algoritmo
incremental deben coincidir exactamente con los calculados por fuerza bruta
sobre las transacciones de la ventana.

Los escenarios incluyen "deriva de concepto" (las frecuencias de los items
cambian a mitad del flujo, lo que hace que itemsets infrecuentes pasen a ser
frecuentes y viceversa) e items cuyo nombre es prefijo de otro ('milk' y
'milkshake').
"""
import itertools
import random

import pytest

pytest.importorskip("pyspark")

from ARMxtend.SARE.DecimalFraction import DecimalFraction  # noqa: E402
from ARMxtend.SARE.FIMoTS_Algorithm import FIMoTS_Main, FIMoTS_TreeInitialization  # noqa: E402
from ARMxtend.SARE.FIMoTS_Node import FIMoTS_Node  # noqa: E402
from ARMxtend.SARE.FIMoTS_Structure import FIMoTS_Structure  # noqa: E402
from ARMxtend.SARE.FIMoTS_Tree import FIMoTS_Tree  # noqa: E402
from ARMxtend.SARE.WindowData import WindowData  # noqa: E402

LETTERS = (list("abcde"), [0.7, 0.6, 0.4, 0.25, 0.1])
PREFIX_NAMES = (["milk", "milkshake", "bread", "breadcrumb", "egg"], [0.7, 0.6, 0.5, 0.4, 0.3])


def brute_force_frequent(lines, min_support, max_len=4):
    sets = [set(line.split()) for line in lines]
    universe = sorted(set().union(*sets))
    frequent = set()
    for size in range(1, max_len + 1):
        for combo in itertools.combinations(universe, size):
            if sum(1 for s in sets if set(combo) <= s) / len(sets) >= min_support - 1e-12:
                frequent.add("_".join(combo))
    return frequent


def check_batches(sc, batches, intervals, numerator=3, denominator=10):
    """
    Alimenta FIMoTS con una lista de lotes (cada uno, una lista de lineas de transaccion) y devuelve los
    desajustes con la fuerza bruta tras cada desplazamiento: [(paso, faltan, sobran)].
    """
    size = len(batches[0])
    minSupport = DecimalFraction(numerator, denominator)
    window, structure, mismatches = WindowData([], 0), FIMoTS_Structure(), []

    for step, lines in enumerate(batches):
        deleted = window.deleteTransacInterval() if len(window.transactionIntervals) == intervals else 0
        rdd = sc.parallelize(lines, 2)
        window.addTransacInterval([rdd, rdd.count()])
        if len(window.transactionIntervals) < intervals:
            continue

        if step == intervals - 1:
            FIMoTS_TreeInitialization(structure, window, minSupport, sc)
        else:
            FIMoTS_Main(deleted, size, structure, window, minSupport, sc)

        current = [line for batch in batches[max(0, step - intervals + 1):step + 1] for line in batch]
        expected = brute_force_frequent(current, numerator / denominator)
        found = set(structure.frequentItemsetsBounds.getItemsetsKeys())
        if found != expected:
            mismatches.append((step, sorted(expected - found), sorted(found - expected)))
    return mismatches


def run_stream(sc, seed, items, probs, intervals=3, size=25, steps=9, numerator=3, denominator=10):
    """Flujo aleatorio con deriva de concepto a mitad del recorrido (ver `check_batches`)."""
    rng = random.Random(seed)
    batches = []
    for step in range(steps):
        drift = step >= steps // 2           # a mitad del flujo cambian las frecuencias
        lines = []
        for _ in range(size):
            transaction = {item for item, p in zip(items, probs)
                           if rng.random() < ((1 - p) * 0.5 + 0.2 if drift else p)}
            lines.append(" ".join(sorted(transaction)) if transaction else "zz")
        batches.append(lines)
    return check_batches(sc, batches, intervals, numerator, denominator)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("scenario", [LETTERS, PREFIX_NAMES], ids=["letters", "prefix-names"])
def test_incremental_fimots_matches_brute_force_after_every_slide(real_sc, scenario, seed):
    items, probs = scenario
    assert run_stream(real_sc, seed, items, probs) == []


def test_window_sizes_and_support_thresholds_other_than_the_default(real_sc):
    assert run_stream(real_sc, 11, *LETTERS, intervals=2, size=30, steps=8, numerator=1, denominator=4) == []
    assert run_stream(real_sc, 12, *LETTERS, intervals=4, size=15, steps=10, numerator=2, denominator=5) == []


# --- el arbol de itemsets ------------------------------------------------------------------

def _tree(*keys):
    nodes = {"-1": FIMoTS_Node([], -1.0, None, None, 0, 0)}
    for key in keys:
        parent = "_".join(key.split("_")[:-1]) or "-1"
        nodes[key] = FIMoTS_Node(key.split("_"), 0.5, None, parent, 0, 0)
    return FIMoTS_Tree(nodes)


def test_siblings_are_matched_by_exact_parent_not_by_text_prefix():
    tree = _tree("milk", "milkshake", "milk_a", "milk_b", "milkshake_a", "milkshake_z")
    assert sorted(tree.getRightSiblings("milk_a")) == ["milk_b"]
    assert sorted(tree.getLeftSiblings("milk_b")) == ["milk_a"]
    assert sorted(tree.getLeftSiblings("milkshake_z")) == ["milkshake_a"]       # no 'milk_a' ni 'milk_b'
    assert sorted(tree.getRightSiblings("milk")) == ["milkshake"]
    assert sorted(tree.getLeftSiblings("milkshake")) == ["milk"]


def test_first_level_siblings_ignore_the_root_node():
    tree = _tree("!", "a", "b")
    assert sorted(tree.getRightSiblings("!")) == ["a", "b"]
    assert sorted(tree.getLeftSiblings("a")) == ["!"]
    assert "-1" not in tree.getLeftSiblings("a")


def test_delete_child_nodes_removes_every_descendant_but_not_look_alikes():
    tree = _tree("a", "a_b", "a_b_c", "a_b_c_d", "a_c", "ab", "ab_c")
    deleted = tree.deleteChildNodes("a_b")
    assert sorted(deleted) == ["a_b_c", "a_b_c_d"]
    assert sorted(k for k in tree.nodeMap if k != "-1") == ["a", "a_b", "a_c", "ab", "ab_c"]
    assert sorted(tree.deleteChildNodes("a")) == ["a_b", "a_c"]
    assert "ab_c" in tree.nodeMap


# --- escenarios deterministas ---------------------------------------------------------------

def test_an_item_that_first_appears_after_the_initial_window_is_tracked(real_sc):
    # 'c' no existe en la ventana inicial; despues aparece junto a 'a' y se vuelve frecuente
    early = ["a b"] * 10 + ["a"] * 5 + ["b"] * 5
    late = ["a b c"] * 8 + ["a c"] * 6 + ["b"] * 6
    assert check_batches(real_sc, [early, early, late, late, late], intervals=2) == []


def test_two_sibling_itemsets_that_become_frequent_in_the_same_window(real_sc):
    # 'c' y 'd' son raros al principio y pasan a ser frecuentes a la vez (y siempre juntos)
    rare = ["a"] * 17 + ["c", "d", "a b"]
    common = ["a c d"] * 8 + ["a"] * 6 + ["c d"] * 4 + ["b"] * 2
    assert check_batches(real_sc, [rare, rare, common, common, common, rare, rare], intervals=2) == []


def test_an_itemset_that_stops_being_frequent_is_removed_with_all_its_descendants(real_sc):
    frequent = ["a b c"] * 15 + ["a b"] * 3 + ["d"] * 2
    gone = ["d"] * 18 + ["a"] * 2
    assert check_batches(real_sc, [frequent, frequent, gone, gone, gone], intervals=2) == []


def test_sfim_reuses_the_single_fimots_implementation_of_sare():
    # SFIM y SARE llegaron a tener copias identicas que divergieron al corregir solo una
    from ARMxtend.SFIM.FIMoTS_Algorithm import FIMoTS_Main as sfimMain
    from ARMxtend.SFIM.FIMoTS_Tree import FIMoTS_Tree as sfimTree
    from ARMxtend.SFIM.WindowData import WindowData as sfimWindow

    assert sfimMain is FIMoTS_Main and sfimTree is FIMoTS_Tree and sfimWindow is WindowData
