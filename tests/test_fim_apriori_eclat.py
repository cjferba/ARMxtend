# -*- coding: utf-8 -*-
"""
Regression tests for the frequent itemset mining algorithms in
Big Data (ARMxtend.FIM.apriori, ARMxtend.FIM.Eclat) and for association rules
(ARMxtend.FIM.BD_ARE), using the example dataset (Table 2) of
Fernandez-Basso, Ruiz & Martin-Bautista (2024), "New Spark solutions for
distributed frequent itemset and association rule mining algorithms",
Cluster Computing, 27, 1217-1234.

They run against `spark_stub.FakeSparkContext`, a local stand-in for the
Spark API used by these algorithms (map/flatMap/reduceByKey/broadcast),
so they run without needing a real JVM/cluster, executing
exactly the same production code that would run on an
authentic SparkContext.
"""
import numpy as np
import pytest

from spark_stub import FakeSparkContext

from ARMxtend.FIM.apriori import DApriori, DAprioriTID
from ARMxtend.FIM.Eclat import DECLAT, FuzzyDECLAT
from ARMxtend.FIM.BD_ARE import association_rules_bd
from ARMxtend.FIM.BD_FARE import FuzzyDAprioriTID

TABLE2_TRANSACTIONS = [
    "A,B,C", "B,D", "A,C,D", "B,C", "A,C",
    "B,D", "A,B,C", "B,C,D", "A,B,D", "B,C,D",
]

EXPECTED_FREQ_ITEMSETS = {
    "A": 0.5, "B": 0.8, "C": 0.7, "D": 0.6,
    "A-B": 0.3, "A-C": 0.4, "B-C": 0.5, "B-D": 0.5, "C-D": 0.3,
}


@pytest.fixture
def sc():
    return FakeSparkContext()


@pytest.mark.parametrize("algorithm", [DApriori, DAprioriTID, DECLAT])
def test_crisp_fim_algorithms_agree_with_hand_computed_itemsets(sc, algorithm):
    result = algorithm.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    assert result == pytest.approx(EXPECTED_FREQ_ITEMSETS)


def test_association_rules_bd_matches_sequential_generate_rules(sc):
    from ARMxtend.FIM.BD_ARE import generate_rules

    freqItemsets = DApriori.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    sequentialRules = generate_rules(freqItemsets, min_conf=0.7)
    distributedRules = association_rules_bd(sc, freqItemsets, min_conf=0.7)

    assert distributedRules == pytest.approx(sequentialRules)
    assert set(distributedRules.keys()) == {("A", "C"), ("C", "B"), ("D", "B")}


def test_fuzzy_declat_with_full_membership_matches_crisp_declat(sc):
    fuzzyTransactions = [[(item, 1.0) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    crispResult = DECLAT.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    fuzzyResult = FuzzyDECLAT.run(sc, sc.parallelize(fuzzyTransactions), min_supp=0.3, num_alpha=3)

    assert set(fuzzyResult.keys()) == set(crispResult.keys())
    for key, crispSupport in crispResult.items():
        # With full membership (degree 1.0), the itemset satisfies ALL the
        # alpha-cuts (including the most demanding one, alpha=1, via >=), so its
        # support equals the crisp one in each of them
        assert fuzzyResult[key] == pytest.approx(np.full(3, crispSupport))


def test_fuzzy_declat_partial_membership_is_non_decreasing_by_looseness(sc):
    # With partial membership degree (0.6) and alpha levels=[1, 0.67, 0.33],
    # only the loosest level (0.33) is exceeded (>=): the support must be 0 at
    # the strict levels and match the crisp one only at the loosest.
    partialTransactions = [[(item, 0.6) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    crispResult = DECLAT.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    # min_supp slightly below 0.3/3 to avoid the exact edge
    # (0.3 is not exactly representable in binary floating point)
    fuzzyResult = FuzzyDECLAT.run(sc, sc.parallelize(partialTransactions), min_supp=0.09, num_alpha=3)

    for key, crispSupport in crispResult.items():
        assert fuzzyResult[key] == pytest.approx([0.0, 0.0, crispSupport])
        # Support is non-decreasing when relaxing the alpha-cut requirement
        assert np.all(np.diff(fuzzyResult[key]) >= -1e-9)


def test_fuzzy_daprioritid_with_full_membership_matches_crisp_dapriori(sc):
    fuzzyTransactions = [[(item, 1.0) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]

    crispResult = DApriori.run(sc, sc.parallelize(TABLE2_TRANSACTIONS), min_supp=0.3)
    fuzzyResult = FuzzyDAprioriTID.run(sc, sc.parallelize(fuzzyTransactions), min_supp=0.3, num_alpha=3)

    assert set(fuzzyResult.keys()) == set(crispResult.keys())
    for key, crispSupport in crispResult.items():
        assert np.all(fuzzyResult[key] == pytest.approx(crispSupport))


# ------------------------------------------- compact TID-lists (numpy arrays) of DECLAT and FuzzyDECLAT

def _random_lines(seed, rows=300, items=9):
    rng = np.random.RandomState(seed)
    probability = rng.uniform(0.4, 0.85, items)
    names = ["i{}".format(k) for k in range(items)]
    return [",".join(name for name, hit in zip(names, rng.rand(items) < probability) if hit) for _ in range(rows)]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("min_supp", [0.05, 0.15])
def test_declat_matches_fpgrowth_on_random_data(sc, seed, min_supp):
    from ARMxtend.FFIM import fpgrowth

    lines = _random_lines(seed)
    result = DECLAT.run(sc, sc.parallelize(lines), min_supp)
    expected = fpgrowth([line.split(",") if line else [] for line in lines], min_supp)
    assert set(result) == set(expected) and result == pytest.approx(expected)
    assert max(len(key.split("-")) for key in result) >= 4                        # recorre varios niveles


@pytest.mark.parametrize("seed", range(4))
def test_fuzzy_declat_matches_fuzzy_fpgrowth_on_random_data(sc, seed):
    from ARMxtend.FFIM import fuzzy_fpgrowth

    rng = np.random.RandomState(seed)
    grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.33, 0.5, 0.97])               # incluye los niveles alpha exactos
    transactions = [[("i{}".format(k), float(rng.choice(grid))) for k in range(7) if rng.rand() < 0.8] for _ in range(250)]
    result = FuzzyDECLAT.run(sc, sc.parallelize(transactions), min_supp=0.05, num_alpha=5)
    expected = fuzzy_fpgrowth(transactions, min_supp=0.05, num_alpha=5)
    assert set(result) == set(expected)
    for key, vector in result.items():
        assert vector == pytest.approx(expected[key], abs=1e-12)
    assert max(len(key.split("-")) for key in result) >= 3


def test_declat_and_fuzzy_declat_handle_empty_input_blank_lines_and_repeated_items(sc):
    assert DECLAT.run(sc, sc.parallelize([]), 0.1) == {} and FuzzyDECLAT.run(sc, sc.parallelize([]), 0.1, 3) == {}
    lines = ["a,b", "a,,b", "", "a,a,b", "  a , b "]                               # vacias, repetidos y espacios
    assert DECLAT.run(sc, sc.parallelize(lines), 0.5) == pytest.approx({"a": 0.8, "b": 0.8, "a-b": 0.8})
    repeated = [[("a", 0.4), ("a", 0.9), ("b", 1.0)], [("a", 1.0), ("b", 0.2)]]      # a repeated item counts with its highest degree
    result = FuzzyDECLAT.run(sc, sc.parallelize(repeated), 0.1, 5)
    assert result["a"] == pytest.approx([0.5, 1.0, 1.0, 1.0, 1.0]) and result["a-b"] == pytest.approx([0.0, 0.5, 0.5, 0.5, 1.0])


def test_fuzzy_declat_rejects_degrees_outside_the_unit_interval(sc):
    for bad in (1.2, -0.1, float("nan")):
        with pytest.raises(ValueError, match="membership degree"):
            FuzzyDECLAT.run(sc, sc.parallelize([[("a", 0.5), ("b", bad)]]), 0.1, 3)


def test_the_vertical_database_is_broadcast_once_and_released(sc):
    class Spy(FakeSparkContext):
        def __init__(self):
            super(Spy, self).__init__()
            self.broadcasts = []

        def broadcast(self, value):
            handle = super(Spy, self).broadcast(value)
            self.broadcasts.append(handle)
            handle.unpersisted = False
            original = handle.unpersist
            handle.unpersist = lambda: (original(), setattr(handle, "unpersisted", True))[0]
            return handle

    spy = Spy()
    DECLAT.run(spy, spy.parallelize(TABLE2_TRANSACTIONS), 0.3)                         # tres niveles de candidatos
    assert len(spy.broadcasts) == 1 and spy.broadcasts[0].unpersisted
    spy = Spy()
    FuzzyDECLAT.run(spy, spy.parallelize([[(item, 1.0) for item in line.split(",")] for line in TABLE2_TRANSACTIONS]), 0.3, 3)
    assert len(spy.broadcasts) == 1 and spy.broadcasts[0].unpersisted


def test_intersect_positions_and_dtype_edge_cases():
    from ARMxtend.FIM.Eclat import _intersect_positions, _tid_dtype

    first, second = np.array([1, 4, 7, 9]), np.array([0, 4, 9, 12, 30])
    mine, theirs = _intersect_positions(first, second)
    assert first[mine].tolist() == second[theirs].tolist() == [4, 9]
    mine, theirs = _intersect_positions(np.array([40]), second)                         # greater than all those of `second`
    assert len(mine) == len(theirs) == 0
    mine, theirs = _intersect_positions(np.array([30]), second)                         # the last of `second`
    assert mine.tolist() == [0] and theirs.tolist() == [4]
    for empty in (_intersect_positions(np.array([], dtype=int), second), _intersect_positions(first, np.array([], dtype=int))):
        assert len(empty[0]) == len(empty[1]) == 0
    assert _tid_dtype(10 ** 6) == np.int32 and _tid_dtype(2 ** 31) == np.int64
