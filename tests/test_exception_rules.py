# -*- coding: utf-8 -*-
"""Tests de ARM.exception_rules: reglas de excepcion (Suzuki, 1997) y reglas anomalas."""
import pytest

from ARMxtend.ARM import association_rules
from ARMxtend.ARM.exception_rules import exception_rules, anomalous_rules


def _repeat(items, times):
    return [set(items) for _ in range(times)]


# Regla comun: fever -> rest (60/70). Excepciones: con must_work no hay descanso
# (must_work solo, en cambio, casi siempre descansa: referencia debil); con
# sick_leave_denied tampoco, pero esa condicion implica ya "sin descanso" por
# si sola (referencia fuerte) y por eso NO es una excepcion interesante.
EXCEPTION_DATA = (
    _repeat({"fever", "rest"}, 60)
    + _repeat({"fever", "must_work"}, 5)
    + _repeat({"must_work", "rest"}, 35)
    + _repeat({"fever", "sick_leave_denied"}, 5)
)

# Regla comun: fever -> rest (60/70). Los 10 casos con fiebre y sin descanso
# acaban todos en el hospital, algo que no es habitual con solo fiebre.
ANOMALY_DATA = (
    _repeat({"fever", "rest"}, 60)
    + _repeat({"fever", "hospital"}, 10)
    + _repeat({"rest"}, 30)
)


def test_exception_rule_found_and_matches_hand_computed_measures():
    result = exception_rules(EXCEPTION_DATA, min_supp=0.3, min_conf=0.8,
                             min_exception_supp=0.01, min_exception_conf=0.8, max_reference_conf=0.5)
    row = result[(result["antecedents"] == frozenset(["fever"])) & (result["consequent"] == "rest")
                 & (result["condition"] == frozenset(["must_work"]))]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["exception_support"] == pytest.approx(5 / 105)       # fever ^ must_work ^ ~rest
    assert row["exception_confidence"] == pytest.approx(1.0)        # 5 de 5 casos fever ^ must_work
    assert row["reference_confidence"] == pytest.approx(5 / 40)     # must_work -> ~rest
    assert row["common_confidence"] == pytest.approx(60 / 70)


def test_reference_rule_discards_conditions_that_already_imply_the_opposite():
    result = exception_rules(EXCEPTION_DATA, min_supp=0.3, min_conf=0.8, min_exception_supp=0.01,
                             min_exception_conf=0.8, max_reference_conf=0.5)
    conditions = set(result["condition"])
    assert frozenset(["must_work"]) in conditions
    assert frozenset(["sick_leave_denied"]) not in conditions

    # relajando la referencia, esa condicion si entra
    relaxed = exception_rules(EXCEPTION_DATA, min_supp=0.3, min_conf=0.8, min_exception_supp=0.01,
                              min_exception_conf=0.8, max_reference_conf=1.0)
    assert frozenset(["sick_leave_denied"]) in set(relaxed["condition"])


def test_exception_support_threshold_filters_rare_exceptions():
    result = exception_rules(EXCEPTION_DATA, min_supp=0.3, min_conf=0.8, min_exception_supp=0.2)
    assert len(result) == 0


def test_exception_rules_accept_precomputed_common_rules():
    import pandas as pd
    from ARMxtend.FFIM import fpgrowth
    from ARMxtend.FIM._shared import key_to_itemset

    frequent = fpgrowth(EXCEPTION_DATA, 0.3)
    common = association_rules(pd.DataFrame({"support": list(frequent.values()),
                                             "itemsets": [key_to_itemset(k) for k in frequent]}),
                               min_threshold=0.8)
    viaRules = exception_rules(EXCEPTION_DATA, rules=common, min_exception_supp=0.01)
    direct = exception_rules(EXCEPTION_DATA, min_supp=0.3, min_conf=0.8, min_exception_supp=0.01)
    assert len(viaRules) == len(direct) > 0


def test_exception_rules_condition_size_two():
    # z1 ^ z2 aparecen tambien juntos con b (10 transacciones): la referencia
    # {z1,z2} -> ~b es debil (5/15) y la condicion de tamano 2 es valida
    data = (_repeat({"a", "b"}, 50) + _repeat({"a", "z1", "z2"}, 5) + _repeat({"z1", "z2", "b"}, 10)
            + _repeat({"z1", "b"}, 20) + _repeat({"z2", "b"}, 20))
    single = exception_rules(data, min_supp=0.3, min_exception_supp=0.01)
    pairs = exception_rules(data, min_supp=0.3, min_exception_supp=0.01, max_condition_size=2)
    assert len(pairs) > len(single)
    assert frozenset(["z1", "z2"]) in set(pairs["condition"])


def test_anomalous_rule_found_and_matches_hand_computed_measures():
    result = anomalous_rules(ANOMALY_DATA, min_supp=0.3, min_conf=0.8, min_anomalous_supp=0.05,
                             min_anomalous_conf=0.8, max_reference_conf=0.5)
    row = result[(result["antecedents"] == frozenset(["fever"])) & (result["consequent"] == "rest")
                 & (result["anomaly"] == frozenset(["hospital"]))]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["anomalous_cases"] == 10                                 # fever ^ ~rest
    assert row["anomalous_support"] == pytest.approx(10 / 100)
    assert row["anomalous_confidence"] == pytest.approx(1.0)
    assert row["reference_confidence"] == pytest.approx(10 / 70)        # fever -> hospital
    assert row["common_confidence"] == pytest.approx(60 / 70)


def test_anomalous_rule_requires_reference_rule_to_be_weak():
    # si "hospital" ya fuera habitual con fiebre (A -> C fuerte) no caracteriza la anomalia
    data = _repeat({"fever", "rest", "hospital"}, 60) + _repeat({"fever", "hospital"}, 10) + _repeat({"rest"}, 30)
    result = anomalous_rules(data, min_supp=0.3, min_conf=0.8, min_anomalous_supp=0.05)
    assert frozenset(["hospital"]) not in set(result["anomaly"])


def test_empty_inputs_return_empty_frames_with_columns():
    assert len(exception_rules([])) == 0 and "exception_confidence" in exception_rules([]).columns
    assert len(anomalous_rules([])) == 0 and "anomalous_confidence" in anomalous_rules([]).columns
    # sin reglas comunes fuertes
    assert len(exception_rules([{"a"}, {"b"}], min_supp=0.9)) == 0
