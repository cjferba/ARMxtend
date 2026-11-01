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


# ------------------------------------------------- reglas temporales y secuencias

@pytest.fixture
def sensors_file(tmp_path):
    """Hora a hora durante 40 dias: la calefaccion se enciende una hora despues de abrirse la puerta; la temperatura es ruido."""
    rng = np.random.RandomState(1)
    index = pd.date_range("2024-01-01", periods=40 * 24, freq="h")
    door = rng.rand(len(index)) < 0.3
    heater = np.zeros(len(index), dtype=int)
    heater[1:] = door[:-1]
    path = tmp_path / "sensors.csv"
    pd.DataFrame({"time": index.strftime("%Y-%m-%d %H:%M:%S"), "temp": rng.normal(15, 4, len(index)),
                  "door": door.astype(int), "heater": heater}).to_csv(path, index=False)
    return str(path)


def test_temporal_rules_finds_the_lagged_rule(sensors_file, capsys):
    assert main(["temporal-rules", sensors_file, "--time-column", "time", "--max-lag", "1", "--min-supp", "0.05",
                 "--min-threshold", "0.95", "--targets", "heater", "--exclude-persistence"]) == 0
    out = capsys.readouterr().out
    assert "door (t-1)" in out and "heater" in out                       # formato legible en la tabla

    main(["temporal-rules", sensors_file, "--time-column", "time", "--min-supp", "0.05", "--min-threshold", "0.95",
          "--targets", "heater", "--exclude-persistence", "--format", "json"])
    records = json.loads(capsys.readouterr().out)
    assert any(r["antecedents"] == "door@lag1" and r["consequents"] == "heater" and r["confidence"] == 1.0
               for r in records)                                          # en json, los nombres crudos
    assert all(r["lags"] == [1] for r in records)


def test_temporal_rules_calendar_and_errors(sensors_file, capsys):
    assert main(["temporal-rules", sensors_file, "--time-column", "time", "--calendar", "part_of_day",
                 "--min-supp", "0.05", "--min-threshold", "0.95", "--targets", "heater", "--format", "csv"]) == 0
    assert len(pd.read_csv(io.StringIO(capsys.readouterr().out))) > 0
    assert main(["temporal-rules", sensors_file, "--calendar", "hour"]) == 1                 # sin --time-column
    assert "--time-column" in capsys.readouterr().err
    assert main(["temporal-rules", sensors_file, "--time-column", "nope"]) == 1
    assert "nope" in capsys.readouterr().err
    assert main(["temporal-rules", sensors_file, "--time-column", "time", "--columns", "door", "ghost"]) == 1
    assert "ghost" in capsys.readouterr().err


def test_rule_drift_reports_statuses(sensors_file, tmp_path, capsys):
    # la segunda mitad cambia la dinamica: la calefaccion deja de seguir a la puerta
    table = pd.read_csv(sensors_file)
    table.loc[table.index >= len(table) // 2, "heater"] = 0
    changed = tmp_path / "changed.csv"
    table.to_csv(changed, index=False)
    assert main(["rule-drift", str(changed), "--time-column", "time", "--window", "10D", "--max-lag", "1",
                 "--min-supp", "0.05", "--min-threshold", "0.95", "--targets", "heater", "--exclude-persistence",
                 "--format", "csv"]) == 0
    frame = pd.read_csv(io.StringIO(capsys.readouterr().out))
    row = frame[(frame["antecedents"] == "door@lag1") & (frame["consequents"] == "heater")].iloc[0]
    assert row["status"] == "vanishing" and row["windows"] == 2 and row["persistence"] == pytest.approx(0.5)

    assert main(["rule-drift", str(changed), "--time-column", "time", "--window", "240", "--max-lag", "1",
                 "--min-supp", "0.05", "--min-threshold", "0.95", "--targets", "heater"]) == 0       # ventanas por filas
    assert capsys.readouterr().out.splitlines()[0].split()[:3] == ["antecedents", "consequents", "status"]
    assert main(["rule-drift", str(changed), "--window", "10D"]) == 1                                  # duracion sin indice temporal
    assert "DatetimeIndex" in capsys.readouterr().err


SEQUENCES = "door,light;kitchen;coffee\ndoor;kitchen;tv;coffee\ndoor;kitchen\ntv;phone\ndoor;kitchen;coffee\n"


@pytest.fixture
def sequences_file(tmp_path):
    path = tmp_path / "routines.txt"
    path.write_text(SEQUENCES)
    return str(path)


def test_sequences_command(sequences_file, capsys):
    assert main(["sequences", sequences_file, "--min-supp", "0.6", "--format", "json"]) == 0
    patterns = {r["sequence"]: r["support"] for r in json.loads(capsys.readouterr().out)}
    assert patterns["<{door} {kitchen} {coffee}>"] == pytest.approx(0.6)
    assert patterns["<{door} {kitchen}>"] == pytest.approx(0.8)
    assert "<{coffee} {door}>" not in patterns

    assert main(["sequences", sequences_file, "--min-supp", "0.6", "--min-len", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and "<{door} {kitchen} {coffee}>" in lines[1]


def test_sequences_from_an_event_log(tmp_path, capsys):
    path = tmp_path / "events.csv"
    pd.DataFrame({"day": ["d1", "d1", "d1", "d2", "d2", "d2"],
                  "when": ["2024-01-01 08:00", "2024-01-01 08:01", "2024-01-01 12:00",
                           "2024-01-02 08:30", "2024-01-02 08:31", "2024-01-02 13:00"],
                  "event": ["door", "light", "coffee", "door", "light", "coffee"]}).to_csv(path, index=False)
    assert main(["sequences", str(path), "--sequence-column", "day", "--time-column", "when", "--item-column", "event",
                 "--bin", "1h", "--min-supp", "1", "--format", "json"]) == 0
    patterns = {r["sequence"] for r in json.loads(capsys.readouterr().out)}
    assert "<{door, light} {coffee}>" in patterns
    assert main(["sequences", str(path), "--sequence-column", "day"]) == 1
    assert "--time-column" in capsys.readouterr().err


def test_sequential_rules_command(sequences_file, capsys):
    assert main(["sequential-rules", sequences_file, "--min-supp", "0.4", "--min-threshold", "0.7",
                 "--max-antecedent", "1", "--max-consequent", "1", "--format", "csv"]) == 0
    frame = pd.read_csv(io.StringIO(capsys.readouterr().out))
    byRule = {(r["antecedents"], r["consequents"]): r for _, r in frame.iterrows()}
    assert byRule[("door", "kitchen")]["confidence"] == pytest.approx(1.0)        # 4 de las 4 secuencias con 'door'
    assert byRule[("door", "kitchen")]["support"] == pytest.approx(0.8)
    assert byRule[("kitchen", "coffee")]["confidence"] == pytest.approx(0.75)     # 3 de las 4 con 'kitchen'
    assert byRule[("kitchen", "coffee")]["support"] == pytest.approx(0.6)
    assert ("coffee", "door") not in byRule and (frame["confidence"] >= 0.7).all()
    assert main(["sequential-rules", sequences_file, "--min-supp", "0.4", "--min-threshold", "0.7", "--top", "1"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
    assert main(["sequential-rules", sequences_file, "--min-supp", "0.4", "--metric", "magic"]) == 1
    assert "metric" in capsys.readouterr().err


# ------------------------------------------------------ significacion estadistica

@pytest.fixture
def noisy_transactions(tmp_path):
    """300 cestas aleatorias de 12 items (p = 0.3) con una asociacion plantada i00 -> i01."""
    rng = np.random.RandomState(1)
    table = rng.rand(300, 12) < 0.3
    table[:, 1] |= table[:, 0] & (rng.rand(300) < 0.6)
    path = tmp_path / "noise.csv"
    path.write_text("\n".join(",".join("i{:02d}".format(i) for i in np.flatnonzero(row)) for row in table if row.any()) + "\n")
    return str(path)


def test_rules_alpha_keeps_only_significant_rules(noisy_transactions, capsys):
    common = ["rules", noisy_transactions, "--min-supp", "0.03", "--min-threshold", "0.45", "--format", "csv"]
    assert main(common) == 0
    naive = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert main(common + ["--alpha", "0.05"]) == 0
    kept = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert 0 < len(kept) < len(naive) and (kept["p_adjusted"] <= 0.05).all()
    assert ((kept["antecedents"] == "i00") & (kept["consequents"] == "i01")).any()
    assert main(common + ["--alpha", "0.05", "--correction", "bonferroni", "--test", "chi2"]) == 0
    assert len(pd.read_csv(io.StringIO(capsys.readouterr().out))) <= len(kept) + 2
    assert main(common + ["--alpha", "1.5"]) == 1
    assert "alpha" in capsys.readouterr().err


def test_temporal_rules_alpha_and_permutations(sensors_file, capsys):
    common = ["temporal-rules", sensors_file, "--time-column", "time", "--min-supp", "0.05", "--min-threshold", "0.95",
              "--targets", "heater", "--exclude-persistence", "--format", "csv"]
    assert main(common + ["--alpha", "0.05"]) == 0
    frame = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert "door@lag1" in set(frame["antecedents"]) and (frame["p_adjusted"] <= 0.05).all()

    assert main(common + ["--permutations", "15", "--alpha", "0.1", "--seed", "3"]) == 0
    permuted = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert len(permuted) > 0 and (permuted["p_permutation"] <= 0.1).all() and "door@lag1" in set(permuted["antecedents"])
    assert permuted["p_permutation"].min() == pytest.approx(1 / 16)

    assert main(["temporal-rules", sensors_file, "--time-column", "time", "--min-supp", "0.05", "--min-threshold", "0.95",
                 "--targets", "heater", "--permutations", "5", "--format", "csv"]) == 0         # sin alpha: no filtra
    assert "p_permutation" in pd.read_csv(io.StringIO(capsys.readouterr().out)).columns


def test_rule_drift_with_alpha(sensors_file, capsys):
    base = ["rule-drift", sensors_file, "--time-column", "time", "--window", "10D", "--max-lag", "1", "--min-supp", "0.05",
            "--min-threshold", "0.95", "--targets", "heater", "--exclude-persistence", "--format", "csv"]
    assert main(base + ["--alpha", "0.05"]) == 0
    frame = pd.read_csv(io.StringIO(capsys.readouterr().out))
    row = frame[(frame["antecedents"] == "door@lag1") & (frame["consequents"] == "heater")].iloc[0]
    assert row["status"] == "stable" and row["windows"] == 4


# ------------------------------------------------------ reglas en accion

@pytest.fixture
def faulty_sensors_file(tmp_path):
    """Como sensors_file, pero la segunda mitad tiene 8 fallos: la calefaccion no se enciende tras abrirse la puerta."""
    rng = np.random.RandomState(1)
    index = pd.date_range("2024-01-01", periods=40 * 24, freq="h")
    door = rng.rand(len(index)) < 0.3
    heater = np.zeros(len(index), dtype=int)
    heater[1:] = door[:-1]
    late = np.flatnonzero(heater[480:]) + 480
    faults = rng.choice(late, 8, replace=False)
    heater[faults] = 0
    path = tmp_path / "faulty.csv"
    pd.DataFrame({"time": index.strftime("%Y-%m-%d %H:%M:%S"), "temp": rng.normal(15, 4, len(index)),
                  "door": door.astype(int), "heater": heater}).to_csv(path, index=False)
    return str(path), set(index[faults].strftime("%Y-%m-%d %H:%M:%S"))


def test_anomalies_command_finds_the_planted_faults(faulty_sensors_file, capsys):
    path, faults = faulty_sensors_file
    base = ["anomalies", path, "--time-column", "time", "--train-fraction", "0.5", "--max-lag", "1", "--min-supp",
            "0.05", "--min-threshold", "0.95", "--targets", "heater", "--exclude-persistence"]
    assert main(base + ["--format", "json"]) == 0
    captured = capsys.readouterr()
    records = json.loads(captured.out)
    assert {pd.Timestamp(r["time"]).strftime("%Y-%m-%d %H:%M:%S") for r in records} == faults
    assert all("door (t-1) => heater (t)" in r["rules"] and r["score"] > 5 for r in records)
    assert "rules learned from the first 480 rows" in captured.err

    assert main(base + ["--top", "3", "--single-consequent", "--alpha", "0.05", "--min-improvement", "0.05"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4 and "door (t-1) => heater (t)" in lines[1]            # texto legible en la tabla
    assert main(base + ["--train-fraction", "1.5"]) == 1
    assert "train-fraction" in capsys.readouterr().err


def test_forecast_command_evaluates_and_predicts_the_next_step(sensors_file, capsys):
    base = ["forecast", sensors_file, "--time-column", "time", "--max-lag", "1", "--min-supp", "0.05",
            "--min-threshold", "0.95", "--targets", "heater", "--exclude-persistence", "--format", "json"]
    assert main(base + ["--train-fraction", "0.7"]) == 0
    record = next(r for r in json.loads(capsys.readouterr().out) if r["item"] == "heater")
    assert record["precision"] == 1.0 and record["recall"] == 1.0 and record["predicted"] == record["positives"] > 20

    assert main(base + ["--next"]) == 0
    forecast = json.loads(capsys.readouterr().out)
    lastDoor = pd.read_csv(sensors_file)["door"].iloc[-1]
    assert (forecast == [{"item": "heater", "score": 1.0}]) if lastDoor else (forecast == [])
    assert main(base[:-2] + ["--next"]) == 0                                           # tabla
    assert capsys.readouterr().out.split()[:2] == (["item", "score"] if lastDoor else ["(no", "rules"])


def test_next_events_command_ranks_pending_events_and_evaluates(sequences_file, capsys):
    base = ["next-events", sequences_file, "--min-supp", "0.4", "--min-threshold", "0.3", "--format", "json"]
    assert main(base + ["--history", "door;kitchen"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"item": "coffee", "score": 0.75}]
    assert main(base + ["--test-fraction", "0.4", "--target", "next", "-k", "2"]) == 0
    measures = {r["measure"]: r["value"] for r in json.loads(capsys.readouterr().out)}
    assert measures["cases"] == 3 and 0 <= measures["hit_rate"] <= 1 and "baseline_hit_rate" in measures
    assert main(base) == 1
    assert "--history" in capsys.readouterr().err


@pytest.fixture
def patients_file(tmp_path):
    rng = np.random.RandomState(2)
    rows = 400
    temp, cough = rng.normal(37, 1, rows), rng.rand(rows) < 0.5
    flu = cough & (temp > np.quantile(temp, 2 / 3))
    diagnosis = np.where(flu, "flu", "ok")
    path = tmp_path / "patients.csv"
    pd.DataFrame({"temp": temp, "cough": cough.astype(int), "color": rng.choice(["red", "blue", "green"], rows),
                  "diagnosis": diagnosis}).to_csv(path, index=False)
    return str(path)


def test_classify_command_learns_the_class_rules(patients_file, capsys):
    assert main(["classify", patients_file, "--class-column", "diagnosis", "--min-supp", "0.1", "--min-threshold", "0.8",
                 "--max-len", "2", "--seed", "0", "--format", "json"]) == 0
    captured = capsys.readouterr()
    rules = json.loads(captured.out)
    assert any(r["consequents"] == "diagnosis=flu" and set(r["antecedents"].split(", ")) == {"cough", "temp=high"}
               for r in rules)
    assert "accuracy on 120 held-out rows: 1.000" in captured.err and "default class: ok" in captured.err

    assert main(["classify", patients_file, "--class-column", "diagnosis", "--test-fraction", "0", "--top", "1"]) == 0
    captured = capsys.readouterr()
    assert "held-out" not in captured.err and len(captured.out.strip().splitlines()) == 2
    assert main(["classify", patients_file, "--class-column", "nope"]) == 1
    assert "columna de clase" in capsys.readouterr().err
    assert main(["classify", patients_file, "--class-column", "diagnosis", "--test-fraction", "1.5"]) == 1
