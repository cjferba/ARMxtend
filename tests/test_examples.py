# -*- coding: utf-8 -*-
"""
Ejecuta cada script de ``examples/`` como parte de la suite de tests, para
que la integracion continua (ver .github/workflows/ci.yml) detecte si un
cambio en la libreria rompe alguno de los ejemplos documentados en
docs/sources/examples.md.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

EXAMPLE_MODULES = [
    "example_association_rules",
    "example_fuzzy_association_rules",
    "example_meta_rules",
    "example_preprocessing",
    "example_vizarm",
    "example_rule_types",
    "example_temporal_rules",
    "example_sequences",
    "example_significance",
    "example_rules_in_action",
    "example_fuzzy_sequences",
]


@pytest.fixture(autouse=True, scope="module")
def _examples_on_path():
    sys.path.insert(0, str(EXAMPLES_DIR))
    yield
    sys.path.remove(str(EXAMPLES_DIR))


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_runs_without_error(module_name, capsys):
    module = importlib.import_module(module_name)
    result = module.main()

    assert result is not None
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_fuzzy_example_reproduces_paper_values(capsys):
    """
    example_fuzzy_association_rules.py debe reproducir exactamente los
    valores de FSupp/FConf/FCF publicados para {i1,i3} -> {i4} en Delgado,
    Ruiz, Sanchez & Serrano (2011) / Ruiz et al. (2016, Seccion 4.4).
    """
    module = importlib.import_module("example_fuzzy_association_rules")
    _, rulesDf = module.main()

    rule = next(
        row for _, row in rulesDf.iterrows()
        if set(row["antecedents"]) == {"i1", "i3"} and set(row["consequents"]) == {"i4"}
    )
    assert rule["support"] == pytest.approx(0.266, abs=0.002)
    assert rule["confidence"] == pytest.approx(0.5, abs=0.002)
    assert rule["certainty_factor"] == pytest.approx(0.33, abs=0.002)


def test_rule_types_example_finds_the_planted_patterns():
    module = importlib.import_module("example_rule_types")
    negative, exceptions, anomalies, gradual, (closed, maximal, pruned), fuzzyRules = module.main()

    # quien toma cafe no toma te
    assert ((negative["antecedents"] == frozenset(["coffee"])) & (negative["consequents"] == frozenset(["~tea"]))).any()
    # la excepcion plantada a "fever -> rest"
    assert ((exceptions["antecedents"] == frozenset(["fever"])) & (exceptions["condition"] == frozenset(["must_work"]))).any()
    assert frozenset(["hospital"]) in set(anomalies["anomaly"])
    # cuanto mas edad y menos ejercicio, mas colesterol
    planted = (gradual["antecedents"] == frozenset([("age", "+"), ("exercise", "-")])) \
        & (gradual["consequents"] == frozenset([("cholesterol", "+")]))
    assert planted.any()
    assert set(maximal) == {"a-b-c", "d"} and len(pruned) == 5
    assert ((fuzzyRules["antecedents"] == frozenset(["age_low"]))
            & (fuzzyRules["consequents"] == frozenset(["cholesterol_low"]))).any()


def test_temporal_example_recovers_the_planted_behaviour(capsys):
    """La casa simulada tiene una regla de la tarde (puerta -> calefaccion) y una regla que emerge en la segunda mitad."""
    module = importlib.import_module("example_temporal_rules")
    _, rules, profile, drift = module.main()

    evening = rules[rules["antecedents"].apply(lambda a: {"door_open@lag1", "part_of_day=evening"} <= set(a))]
    assert len(evening) > 0 and evening["confidence"].max() > 0.9
    assert profile.loc[10, "confidence"] == 0.0 and profile.loc[17:23, "confidence"].min() > 0.7
    emerging = drift[(drift["antecedents"] == frozenset({"temp=low@lag1"}))
                     & (drift["consequents"] == frozenset({"heater_on"}))]
    assert len(emerging) == 1 and emerging.iloc[0]["status"] == "emerging" and emerging.iloc[0]["first_window"] == 2


def test_sequences_example_recovers_the_planted_routine(capsys):
    """Los dias tranquilos tienen cafe y luego television; el telefono es ruido y no genera reglas informativas."""
    module = importlib.import_module("example_sequences")
    sequences, patterns, rules = module.main()

    assert len(sequences) == 120
    kitchen = frozenset(["alarm"]), frozenset(["bathroom"]), frozenset(["kitchen"])
    assert any(tuple(pattern[:3]) == kitchen for pattern in patterns["sequence"])
    rule = rules[(rules["antecedents"] == frozenset(["coffee"])) & (rules["consequents"] == frozenset(["tv"]))].iloc[0]
    assert rule["lift"] > 2 and rule["confidence"] > 0.7
    assert rules[(rules["antecedents"] == frozenset(["tv"])) & (rules["consequents"] == frozenset(["coffee"]))].empty
    informative = rules[(rules["confidence"] >= 0.7) & (rules["lift"] > 1.1)]
    assert set(informative["consequents"].apply(lambda y: next(iter(y)))) == {"tv"}


def test_significance_example_removes_the_noise_rules(capsys):
    """Casi todas las reglas que pasan los umbrales son ruido; tras corregir quedan las de la asociacion real."""
    module = importlib.import_module("example_significance")
    rules, fisher, permuted, (pairs, fisher_alarms, shift_alarms) = module.main()

    union = lambda frame: [a | c for a, c in zip(frame["antecedents"], frame["consequents"])]   # noqa: E731
    noise = lambda frame: sum(not {"i00", "i01"} <= items for items in union(frame))              # noqa: E731
    assert len(rules) == 100 and noise(rules) > 40
    assert 0 < len(fisher) < len(rules) / 2 and noise(fisher) <= 3
    best = permuted[permuted["p_permutation"] <= 0.05]
    assert len(best) > 0 and noise(best) == 0
    assert pairs >= 10 and fisher_alarms >= 5 and shift_alarms <= 2


def test_rules_in_action_example_finds_predicts_and_classifies(capsys):
    """Las reglas minadas con el funcionamiento normal encuentran los fallos plantados, predicen y clasifican."""
    module = importlib.import_module("example_rules_in_action")
    rules, (flagged, fault), (quality, upcoming), (hits, after_coffee), (classifier, accuracy, majority) = module.main()

    assert {frozenset(a) for a in rules["antecedents"]} == {frozenset({"motion@lag1"}), frozenset({"door@lag1"})}
    assert int(fault.loc[flagged.index].sum()) == 30                                  # los 30 fallos, entre 39 instantes
    assert len(flagged) < 45
    assert quality.loc["light", "precision"] > 0.9 and quality.loc["chime", "precision"] == pytest.approx(0.8, abs=0.08)
    assert "light" in upcoming.index
    assert after_coffee.index[0] == "tv" and hits["hit_rate"] > hits["baseline_hit_rate"] + 0.05
    assert accuracy > majority + 0.3 and len(classifier.rules_) <= 4


def test_fuzzy_sequences_example_orders_events_and_grades_the_rule(capsys):
    """La regla difusa 'frio por la manana => consumo por la tarde' destaca, solo en ese orden, y cada dia la cumple en su grado."""
    module = importlib.import_module("example_fuzzy_sequences")
    patterns, rules, degrees, cold, (crisp_rules, fuzzy_rule) = module.main()

    pairs = {tuple(next(iter(element)) for element in pattern): value for pattern, value in zip(patterns["sequence"], patterns["support"])}
    assert pairs[("temp_low", "power_high")] > 0.2 and ("power_high", "temp_low") not in pairs
    assert (rules.iloc[0]["antecedents"], rules.iloc[0]["consequents"]) == (frozenset({"temp_low"}), frozenset({"power_high"}))
    assert rules.iloc[0]["lift"] > 2 > rules.iloc[1]["lift"]
    assert fuzzy_rule["lift"] == pytest.approx(rules.iloc[0]["lift"])
    assert float(np.corrcoef(cold, degrees)[0, 1]) > 0.8 and 0 < degrees.max() < 1 and (degrees == 0).sum() > 20
    supports = [rule["support"] for rule in crisp_rules.values()]                       # el corte cambia mucho el soporte
    assert max(supports) > 2.5 * min(supports)
