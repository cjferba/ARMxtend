# -*- coding: utf-8 -*-
"""Tests de preprocessing.membership y de la fuzzificacion trapezoidal/gaussiana/automatica de FuzzyLib."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules
from ARMxtend.preprocessing import FuzzyLib
from ARMxtend.preprocessing.membership import (
    automatic_partition, automatic_peaks, default_labels, gaussian_membership,
    trapezoidal_membership, trapezoidal_partition, triangular_partition,
)


# --- funciones de pertenencia ----------------------------------------------

def test_trapezoid_hand_computed_values():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8.0])
    degrees = trapezoidal_membership(x, 1, 3, 5, 7)
    assert degrees.tolist() == pytest.approx([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])


def test_trapezoid_vertical_sides_and_infinite_shoulders():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    # lado izquierdo vertical en b = 1 (a == b): 0 antes, 1 desde b
    assert trapezoidal_membership(x, 1, 1, 2, 3).tolist() == pytest.approx([0, 1, 1, 0])
    # hombros infinitos
    assert trapezoidal_membership(x, -np.inf, -np.inf, 1, 3).tolist() == pytest.approx([1, 1, 0.5, 0])
    assert trapezoidal_membership(x, 1, 3, np.inf, np.inf).tolist() == pytest.approx([0, 0, 0.5, 1])


def test_trapezoid_rejects_unordered_vertices():
    with pytest.raises(ValueError):
        trapezoidal_membership([1.0], 3, 2, 4, 5)


def test_gaussian_membership():
    assert gaussian_membership(5.0, 5.0, 2.0) == pytest.approx(1.0)
    assert gaussian_membership(7.0, 5.0, 2.0) == pytest.approx(np.exp(-0.5))
    assert gaussian_membership([3.0, 7.0], 5.0, 2.0)[0] == pytest.approx(gaussian_membership(7.0, 5.0, 2.0))
    with pytest.raises(ValueError):
        gaussian_membership(1.0, 0.0, 0.0)


def test_triangular_peaks_must_be_strictly_increasing():
    with pytest.raises(ValueError):
        triangular_partition([1.0], [1, 1, 2])


# --- particiones automaticas ------------------------------------------------

def test_automatic_peaks_methods():
    x = np.arange(0, 101, dtype=float)
    assert automatic_peaks(x, 3, "uniform").tolist() == pytest.approx([0, 50, 100])
    assert automatic_peaks(x, 5, "quantile").tolist() == pytest.approx([0, 25, 50, 75, 100])
    # datos en tres grupos claros: k-means encuentra el centro de cada uno
    clusters = np.r_[np.full(40, 1.0), np.full(40, 5.0), np.full(40, 9.0)] + np.tile([-0.1, 0.0, 0.1, 0.0], 30)
    assert automatic_peaks(clusters, 3, "kmeans").tolist() == pytest.approx([1.0, 5.0, 9.0], abs=0.05)
    # los cuantiles, en cambio, no reflejan los grupos: sus extremos son min y max
    assert automatic_peaks(clusters, 3, "quantile")[0] == pytest.approx(clusters.min())


def test_quantile_partition_balances_the_data():
    rng = np.random.RandomState(0)
    skewed = rng.exponential(scale=2.0, size=2000)
    memberships, _ = automatic_partition(skewed, 3, "quantile")
    # con picos en min/mediana/max, cada etiqueta "domina" a un numero similar de datos
    # que con 'uniform', donde casi todos caen en la etiqueta baja
    uniformMemberships, _ = automatic_partition(skewed, 3, "uniform")
    quantileShare = np.bincount(memberships.argmax(axis=1), minlength=3) / len(skewed)
    uniformShare = np.bincount(uniformMemberships.argmax(axis=1), minlength=3) / len(skewed)
    assert quantileShare.min() > uniformShare.min()
    assert quantileShare[0] < uniformShare[0]
    assert uniformShare[0] > 0.8                      # la cola larga aplasta a 'uniform'


@pytest.mark.parametrize("method", ["uniform", "quantile", "kmeans"])
@pytest.mark.parametrize("membership", ["triangular", "trapezoidal"])
def test_ruspini_partitions_sum_to_one(method, membership):
    rng = np.random.RandomState(1)
    values = rng.normal(size=500)
    memberships, params = automatic_partition(values, 4, method, membership)
    assert memberships.shape == (500, 4) and len(params) == 4
    assert np.allclose(memberships.sum(axis=1), 1.0)
    assert memberships.min() >= 0.0 and memberships.max() <= 1.0


def test_gaussian_neighbours_cross_at_one_half():
    _, params = automatic_partition(np.linspace(0, 10, 50), 3, "uniform", "gaussian")
    (c0, s0), (c1, s1), _ = params
    midpoint = (c0 + c1) / 2
    assert gaussian_membership(midpoint, c0, s0) == pytest.approx(0.5)
    assert gaussian_membership(midpoint, c1, s1) == pytest.approx(0.5)


def test_trapezoidal_partition_has_cores_of_full_membership():
    values = np.linspace(0, 12, 121)
    memberships, params = automatic_partition(values, 3, "uniform", "trapezoidal")
    peaks = [0.0, 6.0, 12.0]
    for column, peak in enumerate(peaks):
        assert memberships[np.argmin(np.abs(values - peak)), column] == pytest.approx(1.0)
    # el nucleo del conjunto central ocupa un cuarto de hueco a cada lado del pico
    assert memberships[np.argmin(np.abs(values - 7.4)), 1] == pytest.approx(1.0)


def test_automatic_partition_errors():
    with pytest.raises(ValueError, match="variabilidad"):
        automatic_peaks([1.0, 1.0, 1.0, 1.0], 3, "quantile")
    with pytest.raises(ValueError):
        automatic_peaks([1.0, 2.0], 1, "quantile")
    with pytest.raises(ValueError):
        automatic_peaks([1.0, 2.0], 3, "median")
    with pytest.raises(ValueError):
        automatic_partition([1.0, 2.0, 3.0], 3, "uniform", "sigmoid")
    with pytest.raises(ValueError):
        automatic_peaks([np.nan, np.nan], 3)


def test_default_labels():
    assert default_labels(3) == ["low", "medium", "high"]
    assert default_labels(7)[:2] == ["L1", "L2"]


# --- FuzzyLib ---------------------------------------------------------------

def _lib(**columns):
    lib = FuzzyLib()
    lib.data = pd.DataFrame(columns)
    return lib


def test_fuzzification_trapezoidal_and_gaussian_through_fuzzylib():
    lib = _lib(t=[10.0, 20.0, 30.0])
    added = lib.Fuzzification(["t"], [[(-np.inf, -np.inf, 10, 20), (10, 20, np.inf, np.inf)]],
                              [["cold", "hot"]], membership="trapezoidal")
    assert added == ["t_cold", "t_hot"]
    assert lib.data["t_cold"].tolist() == pytest.approx([1, 0, 0])
    assert lib.data["t_hot"].tolist() == pytest.approx([0, 1, 1])

    lib = _lib(t=[10.0, 20.0, 30.0])
    lib.Fuzzification(["t"], [[(20.0, 5.0)]], [["warm"]], membership="gaussian")
    assert lib.data["t_warm"].tolist() == pytest.approx(
        [np.exp(-0.5 * 4), 1.0, np.exp(-0.5 * 4)])


def test_fuzzification_rejects_unknown_membership():
    with pytest.raises(ValueError):
        _lib(t=[1.0, 2.0]).Fuzzification(["t"], [[1, 2]], [["a", "b"]], membership="sigmoid")


def test_auto_fuzzification_adds_columns_and_records_partitions():
    rng = np.random.RandomState(2)
    lib = _lib(temp=rng.normal(20, 5, 200), hum=rng.uniform(0, 100, 200), name=["x"] * 200)
    added = lib.AutoFuzzification(n_labels=3, method="quantile")

    assert added == ["temp_low", "temp_medium", "temp_high", "hum_low", "hum_medium", "hum_high"]
    assert np.allclose(lib.data[["temp_low", "temp_medium", "temp_high"]].sum(axis=1), 1.0)
    assert lib.partitions["temp"]["membership"] == "triangular"
    assert len(lib.partitions["temp"]["params"]) == 3
    assert "name_low" not in lib.data.columns                 # solo atributos numericos


def test_auto_fuzzification_params_are_reusable_with_fuzzification():
    rng = np.random.RandomState(3)
    first = _lib(x=rng.normal(size=300))
    first.AutoFuzzification(["x"], n_labels=3, method="kmeans", membership="trapezoidal")
    params = first.partitions["x"]["params"]

    second = _lib(x=first.data["x"].to_numpy())
    second.Fuzzification(["x"], [params], [["low", "medium", "high"]], membership="trapezoidal")
    assert np.allclose(second.data[["x_low", "x_medium", "x_high"]].to_numpy(),
                       first.data[["x_low", "x_medium", "x_high"]].to_numpy())


def test_auto_fuzzification_custom_labels_and_errors():
    lib = _lib(x=np.arange(50.0))
    assert lib.AutoFuzzification(["x"], n_labels=2, FuzzyLabel=["no", "si"]) == ["x_no", "x_si"]
    with pytest.raises(ValueError):
        lib.AutoFuzzification(["x"], n_labels=3, FuzzyLabel=["a", "b"])
    with pytest.raises(ValueError, match="Atributo 'c'"):
        _lib(c=[5.0] * 10).AutoFuzzification(["c"], n_labels=3)


def test_get_fuzzy_transactions_feed_the_fuzzy_miners():
    rng = np.random.RandomState(4)
    x = rng.uniform(0, 10, 200)
    lib = _lib(x=x, y=x * 2 + rng.normal(0, 0.5, 200))          # y crece con x
    lib.AutoFuzzification(n_labels=3, method="quantile")

    transactions = lib.GetFuzzyTransactions()
    assert len(transactions) == 200
    assert all(degree > 0 for tx in transactions for _, degree in tx)
    strong = lib.GetFuzzyTransactions(columns=["x_low", "x_high"], min_degree=0.5)
    assert all(item in ("x_low", "x_high") and degree > 0.5 for tx in strong for item, degree in tx)
    assert sum(len(tx) for tx in strong) < sum(len(tx) for tx in lib.GetFuzzyTransactions(["x_low", "x_high"]))

    itemsets = fuzzy_fpgrowth(transactions, min_supp=0.1, num_alpha=5)
    rules = fuzzy_association_rules(itemsets, 5, metric="confidence", min_threshold=0.7)
    # la dependencia monotona entre x e y aparece como regla entre etiquetas bajas
    lowRules = rules[(rules["antecedents"] == frozenset(["x_low"])) & (rules["consequents"] == frozenset(["y_low"]))]
    assert len(lowRules) == 1
