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
