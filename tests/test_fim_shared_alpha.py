# -*- coding: utf-8 -*-
"""
Tests de regresion para las utilidades de alpha-cortes de
ARMxtend.FIM._shared (alpha_levels, alpha_weights, alpha_cuts,
weighted_alpha_aggregate) y ARMxtend.ARM._measures.certainty_factor,
verificados contra los ejemplos numericos textuales y la Tabla 6 de:

    Delgado, M., Ruiz, M.D., Sanchez, D., Serrano, J.M. (2011). "A formal
    model for mining fuzzy rules using the RL representation theory".
    Information Sciences, 181, 5194-5213.

    Ruiz, M.D., et al. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223 (Seccion 4.4, Tabla 6).

    Fernandez-Basso, C., Ruiz, M.D., Martin-Bautista, M.J. (2021). "Spark
    solutions for discovering fuzzy association rules in Big Data". Int. J.
    of Approximate Reasoning, 137, 94-112 (Seccion 3.1.1).
"""
import numpy as np
import pytest

from ARMxtend.ARM._measures import certainty_factor
from ARMxtend.FIM._shared import alpha_cuts, alpha_levels, alpha_weights, weighted_alpha_aggregate


def test_alpha_levels_four_cuts_matches_paper_example():
    assert list(alpha_levels(4)) == [1.0, 0.75, 0.5, 0.25]


def test_alpha_cuts_matches_fuzzytoarray_textual_example():
    # "if the itemset X is satisfied with degree 0.25 [...] and the set of
    # alpha-cuts is {1, 0.75, 0.5, 0.25}, then the associated bit-list of X
    # [...] will be [0, 0, 0, 1]" (Fernandez-Basso et al. 2021, Sec. 3.1.1)
    assert list(alpha_cuts(0.25, 4)) == [0, 0, 0, 1]


def test_alpha_cuts_boundary_values():
    assert list(alpha_cuts(1.0, 4)) == [1, 1, 1, 1]
    assert list(alpha_cuts(0.0, 4)) == [0, 0, 0, 0]
    # en el limite (>=), un valor igual al nivel alpha lo satisface
    assert list(alpha_cuts(0.5, 4)) == [0, 0, 1, 1]


def test_alpha_cuts_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        alpha_cuts(1.5, 4)
    with pytest.raises(ValueError):
        alpha_cuts(-0.1, 4)


def test_alpha_weights_equidistant_levels_are_uniform():
    weights = alpha_weights(alpha_levels(5))
    assert weights == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])
    assert sum(weights) == pytest.approx(1.0)


def test_weighted_alpha_aggregate_matches_paper_fsupp():
    # Tabla 6: a_i para A={i1,i3} en cada uno de los 5 niveles, N=6
    a = np.array([0, 1, 1, 3, 3])
    weights = alpha_weights(alpha_levels(5))
    fsupp = weighted_alpha_aggregate(a / 6, weights)
    assert fsupp == pytest.approx(0.2667, abs=1e-3)


def test_certainty_factor_matches_paper_table6_per_level_values():
    # Tabla 6 (a, b, c, d) por nivel para A={i1,i3} -> B={i4}, N=6
    a = np.array([0, 1, 1, 3, 3])
    b = np.array([1, 1, 1, 1, 1])
    c = np.array([0, 0, 1, 0, 0])
    N = 6

    supportAB = a / N
    supportA = (a + b) / N
    supportB = (a + c) / N

    cfPerLevel = certainty_factor(supportAB, supportA, supportB)
    assert cfPerLevel == pytest.approx([0.0, 0.4, 0.25, 0.5, 0.5])

    weights = alpha_weights(alpha_levels(5))
    assert weighted_alpha_aggregate(cfPerLevel, weights) == pytest.approx(0.33, abs=1e-9)


def test_certainty_factor_scalar_matches_definition():
    # Delgado et al. 2011, Definicion 1: signo del CF segun creencia
    increases = certainty_factor(support_AB=0.4, support_A=0.5, support_B=0.2)
    assert increases > 0  # Conf=0.8 > supp(B)=0.2 -> la creencia aumenta

    decreases = certainty_factor(support_AB=0.05, support_A=0.5, support_B=0.5)
    assert decreases < 0  # Conf=0.1 < supp(B)=0.5 -> la creencia disminuye

    unchanged = certainty_factor(support_AB=0.25, support_A=0.5, support_B=0.5)
    assert unchanged == pytest.approx(0.0)  # Conf=0.5 = supp(B) -> sin cambio


def test_certainty_factor_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        supportA = rng.uniform(0.01, 1.0)
        supportB = rng.uniform(0.0, 1.0)
        supportAB = rng.uniform(0.0, min(supportA, supportB) if supportB > 0 else supportA)
        cf = certainty_factor(supportAB, supportA, supportB)
        assert -1.0 - 1e-9 <= cf <= 1.0 + 1e-9
