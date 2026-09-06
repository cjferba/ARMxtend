# -*- coding: utf-8 -*-
"""
Tests de regresion para ARMxtend.ARM.meta_rules (meta-reglas de asociacion
crisp y difusas), siguiendo los Algoritmos 3 y 4 de:
    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223.
"""
import pytest

from ARMxtend.ARM.meta_rules import (
    rule_key,
    mine_primary_rule_measures,
    build_crisp_meta_database,
    crisp_meta_association_rules,
    build_fuzzy_meta_database,
    fuzzy_meta_association_rules,
)


def test_rule_key_format():
    assert rule_key(["B", "A"], ["C"]) == "A,B=>C"


def test_mine_primary_rule_measures_finds_expected_rule():
    dataset = [["A", "B"], ["A", "B"], ["A", "B"], ["A"], ["B"]]
    measures = mine_primary_rule_measures([dataset], min_supp=0.5, min_conf=0.5, metric="confidence")

    assert len(measures) == 1
    assert measures[0][rule_key(["A"], ["B"])] == pytest.approx(3 / 4)


def test_build_crisp_meta_database_shape():
    ruleSets = [{"A=>B", "C=>D"}, {"A=>B"}, {"A=>B", "C=>D"}, set()]
    metaDb = build_crisp_meta_database(ruleSets)

    assert set(metaDb.columns) == {"A=>B", "C=>D"}
    assert list(metaDb["A=>B"]) == [1, 1, 1, 0]
    assert list(metaDb["C=>D"]) == [1, 0, 1, 0]


def test_crisp_meta_association_rules_finds_co_occurring_primary_rules():
    # 'A=>B' y 'C=>D' co-ocurren siempre que aparecen (3 de 4 datasets);
    # 'E=>F' aparece aislado en un unico dataset.
    ruleSets = [
        {"A=>B", "C=>D"},
        {"A=>B", "C=>D"},
        {"A=>B", "C=>D"},
        {"E=>F"},
    ]

    metaRules = crisp_meta_association_rules(ruleSets, min_supp=0.5, min_conf=0.9, metric="confidence")

    found = {(frozenset(row["antecedents"]), frozenset(row["consequents"])) for _, row in metaRules.iterrows()}
    assert (frozenset(["A=>B"]), frozenset(["C=>D"])) in found
    assert (frozenset(["C=>D"]), frozenset(["A=>B"])) in found

    rule = metaRules[(metaRules["antecedents"] == frozenset(["A=>B"]))
                     & (metaRules["consequents"] == frozenset(["C=>D"]))].iloc[0]
    assert rule["support"] == pytest.approx(3 / 4)
    assert rule["confidence"] == pytest.approx(1.0)

    # 'E=>F' solo aparece en 1/4 datasets: por debajo de min_supp, no genera meta-reglas
    assert not any("E=>F" in antecedent or "E=>F" in consequent for antecedent, consequent in found)


def test_crisp_meta_association_rules_with_attributes():
    ruleSets = [{"A=>B"}, {"A=>B"}, {"A=>B"}, set()]
    attributes = [{"high_security": 1}, {"high_security": 1}, {"high_security": 1}, {"high_security": 0}]

    metaRules = crisp_meta_association_rules(ruleSets, attributes=attributes, min_supp=0.5, min_conf=0.9)

    found = {(frozenset(row["antecedents"]), frozenset(row["consequents"])) for _, row in metaRules.iterrows()}
    assert (frozenset(["A=>B"]), frozenset(["high_security"])) in found


def test_fuzzy_meta_database_reflects_rule_strength_not_just_presence():
    # 'A=>B' esta SIEMPRE presente (4/4), pero con confianza baja y variable;
    # 'C=>D' esta presente en 3/4, siempre con confianza muy alta.
    ruleMeasureSets = [
        {"A=>B": 0.2, "C=>D": 0.95},
        {"A=>B": 0.3, "C=>D": 0.9},
        {"A=>B": 0.25, "C=>D": 0.92},
        {"A=>B": 0.15},
    ]

    metaDb = build_fuzzy_meta_database(ruleMeasureSets)
    assert list(metaDb["A=>B"]) == [0.2, 0.3, 0.25, 0.15]
    assert list(metaDb["C=>D"]) == [0.95, 0.9, 0.92, 0.0]

    # Version crisp: 'A=>B' se ve tan "fuerte" como 'C=>D' (ambas simplemente presentes)
    ruleSets = [set(rm.keys()) for rm in ruleMeasureSets]
    crispDb = build_crisp_meta_database(ruleSets)
    assert list(crispDb["A=>B"]) == [1, 1, 1, 1]


def test_fuzzy_meta_association_rules_finds_high_confidence_co_occurrence():
    ruleMeasureSets = [
        {"A=>B": 0.85, "C=>D": 0.9},
        {"A=>B": 0.8, "C=>D": 0.88},
        {"A=>B": 0.9, "C=>D": 0.95},
        {"E=>F": 0.9},
    ]

    metaRules = fuzzy_meta_association_rules(ruleMeasureSets, num_alpha=5, min_supp=0.5,
                                             min_conf=0.5, metric="confidence")

    found = {(frozenset(row["antecedents"]), frozenset(row["consequents"])) for _, row in metaRules.iterrows()}
    assert (frozenset(["A=>B"]), frozenset(["C=>D"])) in found


def test_build_fuzzy_meta_database_normalize_scales_to_unit_max():
    ruleMeasureSets = [{"A=>B": 0.1}, {"A=>B": 0.2}, {"A=>B": 0.05}]
    metaDb = build_fuzzy_meta_database(ruleMeasureSets, normalize=True)
    assert metaDb["A=>B"].max() == pytest.approx(1.0)
    assert list(metaDb["A=>B"]) == pytest.approx([0.5, 1.0, 0.25])
