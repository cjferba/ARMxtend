# -*- coding: utf-8 -*-
"""Tests de regresion para ARMxtend.FIM.FARE (reglas de asociacion difusas)."""
import numpy as np
import pandas as pd
import pytest

from ARMxtend.FIM.FARE import fuzzy_association_rules, fuzzy_association_rules_all_alphas


@pytest.fixture
def fuzzy_itemsets_df():
    return pd.DataFrame({
        "itemsets": [frozenset(["A"]), frozenset(["B"]), frozenset(["A", "B"])],
        "support": [np.array([0.5, 0.4, 0.2]), np.array([0.8, 0.6, 0.3]), np.array([0.3, 0.2, 0.1])],
    })


def test_fuzzy_association_rules_at_alpha_zero(fuzzy_itemsets_df):
    rules = fuzzy_association_rules(fuzzy_itemsets_df, alpha_level=0, min_threshold=0.5)
    assert len(rules) == 1
    row = rules.iloc[0]
    assert row["antecedents"] == frozenset(["A"])
    assert row["consequents"] == frozenset(["B"])
    assert row["support"] == pytest.approx(0.3)
    assert row["confidence"] == pytest.approx(0.6)


def test_fuzzy_association_rules_all_alphas_adds_alpha_column(fuzzy_itemsets_df):
    rules = fuzzy_association_rules_all_alphas(fuzzy_itemsets_df, num_alpha=3, min_threshold=0.3)
    assert set(rules["alpha"]) == {0, 1, 2}
    assert {"A", "B"} <= set().union(*rules["antecedents"], *rules["consequents"])


def test_fuzzy_association_rules_confidence_decreases_with_stricter_alpha(fuzzy_itemsets_df):
    rules = fuzzy_association_rules_all_alphas(fuzzy_itemsets_df, num_alpha=3, min_threshold=0.0)
    a_to_b = rules[(rules["antecedents"] == frozenset(["A"])) & (rules["consequents"] == frozenset(["B"]))]
    confidences = a_to_b.sort_values("alpha")["confidence"].to_numpy()
    assert np.allclose(confidences, [0.3 / 0.5, 0.2 / 0.4, 0.1 / 0.2])
