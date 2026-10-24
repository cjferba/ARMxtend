# -*- coding: utf-8 -*-
"""Tests de la interfaz de linea de comandos (ARMxtend.cli)."""
import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ARMxtend import __version__
from ARMxtend.cli import main

TRANSACTIONS = "A,B,C\nB,D\nA,C,D\nB,C\nA,C\nB,D\nA,B,C\nB,C,D\nA,B,D\nB,C,D\n"


@pytest.fixture
def transactions_file(tmp_path):
    path = tmp_path / "tx.csv"
    path.write_text(TRANSACTIONS)
    return str(path)


@pytest.fixture
def numeric_file(tmp_path):
    rng = np.random.RandomState(0)
    age, exercise = rng.uniform(20, 80, 100), rng.uniform(0, 10, 100)
    path = tmp_path / "health.csv"
    pd.DataFrame({"age": age, "exercise": exercise,
                  "chol": 120 + 1.5 * age - 6 * exercise + rng.normal(0, 8, 100)}).to_csv(path, index=False)
    return str(path)


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_rules_table_output(transactions_file, capsys):
    assert main(["rules", transactions_file, "--min-supp", "0.3", "--min-threshold", "0.7"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split() == ["antecedents", "consequents", "support", "confidence", "lift", "certainty_factor"]
    assert len(lines) == 4                                               # cabecera + las 3 reglas de la Tabla 2
    assert lines[1].split()[:2] == ["D", "B"]                            # ordenadas por confianza descendente


def test_rules_csv_and_json_contain_every_measure(transactions_file, capsys):
    main(["rules", transactions_file, "--min-supp", "0.3", "--min-threshold", "0.7", "--format", "csv"])
    frame = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert {"jaccard", "zhangs_metric", "certainty_factor"} <= set(frame.columns) and len(frame) == 3

    main(["rules", transactions_file, "--min-supp", "0.3", "--min-threshold", "0.7", "--format", "json", "--top", "1"])
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 1 and records[0]["antecedents"] == "D" and records[0]["confidence"] == pytest.approx(0.8333, abs=1e-3)


def test_rules_other_metric_sort_and_output_file(transactions_file, tmp_path, capsys):
    target = tmp_path / "out.csv"
    assert main(["rules", transactions_file, "--min-supp", "0.3", "--metric", "lift", "--min-threshold", "1.1",
                 "--format", "csv", "-o", str(target)]) == 0
    assert capsys.readouterr().out == ""
    frame = pd.read_csv(target)
    # el lift es simetrico: salen A -> C y C -> A, y solo ellas
    assert set(frame["antecedents"]) == {"A", "C"} and (frame["lift"] > 1.1).all()


def test_rules_reads_stdin_and_custom_delimiter(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(TRANSACTIONS.replace(",", ";")))
    assert main(["rules", "-", "-d", ";", "--min-supp", "0.3", "--min-threshold", "0.7"]) == 0
    assert "D" in capsys.readouterr().out


def test_rules_html_viewer(transactions_file, tmp_path, capsys):
    target = tmp_path / "viewer.html"
    assert main(["rules", transactions_file, "--min-supp", "0.3", "--min-threshold", "0.7", "--html", str(target)]) == 0
    assert "vizare-data" in target.read_text(encoding="utf-8")
    assert str(target) in capsys.readouterr().err


def test_rules_no_rules_found(transactions_file, capsys):
    assert main(["rules", transactions_file, "--min-supp", "0.3", "--min-threshold", "0.99"]) == 0
    assert "no rules found" in capsys.readouterr().out


def test_prune_flag_reduces_rules(tmp_path, capsys):
    path = tmp_path / "baskets.csv"
    path.write_text("a,b,c\n" * 3 + "a,b\n" + "d\n")
    main(["rules", str(path), "--min-supp", "0.2", "--min-threshold", "0", "--format", "csv"])
    full = len(pd.read_csv(io.StringIO(capsys.readouterr().out)))
    main(["rules", str(path), "--min-supp", "0.2", "--min-threshold", "0", "--format", "csv", "--prune"])
    pruned = len(pd.read_csv(io.StringIO(capsys.readouterr().out)))
    assert 0 < pruned < full


def test_fuzzy_rules(numeric_file, capsys):
    assert main(["fuzzy-rules", numeric_file, "--min-supp", "0.1", "--metric", "zhangs_metric",
                 "--min-threshold", "0.5", "--measures", "zhangs_metric", "--format", "csv"]) == 0
    frame = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert len(frame) > 0 and (frame["zhangs_metric"] >= 0.5).all()
    assert set(frame["antecedents"]) | set(frame["consequents"]) >= {"age_high", "chol_high"}


def test_gradual_rules(numeric_file, capsys):
    assert main(["gradual-rules", numeric_file, "--min-supp", "0.25", "--min-threshold", "0.9", "--top", "1"]) == 0
    assert "chol↑" in capsys.readouterr().out


def test_errors_return_non_zero_and_message(capsys):
    assert main(["rules", "/definitely/not/here.csv"]) == 1
    assert "armxtend: error" in capsys.readouterr().err


def test_fuzzy_rules_constant_column_is_a_clean_error(tmp_path, capsys):
    path = tmp_path / "const.csv"
    pd.DataFrame({"x": [1.0] * 20, "y": range(20)}).to_csv(path, index=False)
    assert main(["fuzzy-rules", str(path)]) == 1
    assert "Atributo 'x'" in capsys.readouterr().err


def test_module_entry_point(transactions_file):
    result = subprocess.run([sys.executable, "-m", "ARMxtend", "rules", transactions_file, "--min-supp", "0.3",
                             "--min-threshold", "0.7"], capture_output=True, text=True)
    assert result.returncode == 0 and "antecedents" in result.stdout
