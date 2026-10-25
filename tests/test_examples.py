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
