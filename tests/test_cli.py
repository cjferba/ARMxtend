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


# ---------------------------------------------------- secuencias difusas

@pytest.fixture
def daily_file(tmp_path):
    """40 dias de 6 lecturas cada 4 h: los dias calurosos la temperatura sube por la manana y el consumo despues; filas desordenadas."""
    rng = np.random.RandomState(3)
    rows = []
    for day in range(40):
        hot = rng.rand() < 0.5
        for hour in range(6):
            moment = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=hour * 4)
            temp = (28 if hour < 2 else 18) if hot else 18
            power = (9 if hour >= 3 else 2) if hot else 2
            rows.append((moment.strftime("%Y-%m-%d %H:%M:%S"), moment.strftime("%Y-%m-%d"), temp + rng.normal(0, 1),
                         power + rng.normal(0, 0.5)))
    path = tmp_path / "daily.csv"
    pd.DataFrame(rows, columns=["time", "day", "temp", "power"]).sample(frac=1, random_state=0).to_csv(path, index=False)
    return str(path)


def test_fuzzy_sequences_command_finds_the_ordered_pattern(daily_file, capsys):
    base = ["fuzzy-sequences", daily_file, "--columns", "temp", "power", "--sequence-column", "day", "--time-column",
            "time", "--min-degree", "0.2", "--format", "json"]
    assert main(base + ["--min-supp", "0.3", "--min-len", "2", "--max-len", "2"]) == 0
    patterns = {r["sequence"]: r for r in json.loads(capsys.readouterr().out)}
    assert patterns["<{temp_high} {power_high}>"]["support"] > 0.4 and patterns["<{temp_high} {power_high}>"]["length"] == 2
    assert "<{power_high} {temp_high}>" not in patterns                      # el orden importa

    assert main(base + ["--min-supp", "0.3", "--tnorm", "product", "--max-len", "1"]) == 0
    product = {r["sequence"]: r["support"] for r in json.loads(capsys.readouterr().out)}
    assert 0.4 < product["<{temp_high}>"] < patterns["<{temp_high} {power_high}>"]["support"] + 0.1
    with pytest.raises(SystemExit):                                          # argparse rechaza t-normas desconocidas
        main(base + ["--tnorm", "lukasiewicz"])
    capsys.readouterr()


def test_fuzzy_sequential_rules_command(daily_file, capsys):
    base = ["fuzzy-sequential-rules", daily_file, "--columns", "temp", "power", "--sequence-column", "day",
            "--time-column", "time", "--min-degree", "0.2", "--min-supp", "0.2", "--min-threshold", "0.9",
            "--format", "json"]
    assert main(base) == 0
    rules = json.loads(capsys.readouterr().out)
    rule = next(r for r in rules if r["antecedents"] == "temp_high" and r["consequents"] == "power_high")
    assert rule["confidence"] > 0.9 and rule["lift"] > 1.5
    assert not any(r["antecedents"] == "power_high" and r["consequents"] == "temp_high" for r in rules)

    assert main(base + ["--metric", "lift", "--min-threshold", "1.5", "--max-antecedent", "1", "--max-consequent", "1"]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert rules and all("," not in r["antecedents"] and "," not in r["consequents"] for r in rules)
    assert all(r["lift"] >= 1.5 for r in rules)


def test_fuzzy_sequences_one_sequence_and_errors(daily_file, capsys):
    assert main(["fuzzy-sequences", daily_file, "--columns", "temp", "--time-column", "time", "--min-supp", "0.5",
                 "--max-len", "1", "--top", "2"]) == 0                      # sin --sequence-column: una sola secuencia
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
    assert main(["fuzzy-sequences", daily_file, "--sequence-column", "nope"]) == 1
    assert "'nope'" in capsys.readouterr().err
    assert main(["fuzzy-sequences", daily_file, "--columns", "temp", "--min-supp", "2"]) == 1
    assert "min_supp" in capsys.readouterr().err


# --------------------------------------------------- reglas generalizadas (taxonomia)

@pytest.fixture
def shop_files(tmp_path):
    """El ejemplo de Srikant y Agrawal: seis cestas y una taxonomia de ropa y calzado."""
    baskets = tmp_path / "shop.txt"
    baskets.write_text("Shirts\nJacket,Hiking Boots\nSki Pants,Hiking Boots\nShoes\nShoes\nJacket\n")
    taxonomy = tmp_path / "taxonomy.csv"
    taxonomy.write_text("child,parent\nOuterwear,Clothes\nShirts,Clothes\nJacket,Outerwear\nSki Pants,Outerwear\n"
                        "Shoes,Footwear\nHiking Boots,Footwear\n")
    return str(baskets), str(taxonomy)


def test_generalized_rules_command(shop_files, capsys):
    baskets, taxonomy = shop_files
    base = ["generalized-rules", baskets, "--taxonomy", taxonomy, "--min-supp", "0.3", "--min-threshold", "0.6"]
    assert main(base + ["--format", "json"]) == 0
    rules = {(r["antecedents"], r["consequents"]): r for r in json.loads(capsys.readouterr().out)}
    assert set(rules) == {("Outerwear", "Hiking Boots"), ("Outerwear", "Footwear"), ("Hiking Boots", "Outerwear"),
                          ("Hiking Boots", "Clothes")}
    assert rules[("Outerwear", "Hiking Boots")]["confidence"] == pytest.approx(2 / 3) and rules[("Hiking Boots", "Clothes")]["level"] == 2

    assert main(base + ["--interest", "1.5", "--format", "json"]) == 0           # Hiking Boots => Outerwear no supera lo esperado por 1.5
    kept = {(r["antecedents"], r["consequents"]): r for r in json.loads(capsys.readouterr().out)}
    assert set(kept) == set(rules) - {("Hiking Boots", "Outerwear")}
    assert kept[("Outerwear", "Hiking Boots")]["interest"] == pytest.approx(2.0)

    assert main(base + ["--interest", "1.1", "--top", "1", "--sort-by", "lift"]) == 0
    table = capsys.readouterr().out.strip().splitlines()
    assert len(table) == 2 and "interest" in table[0] and "level" in table[0]


def test_generalized_rules_levels_alpha_and_errors(shop_files, capsys):
    baskets, taxonomy = shop_files
    assert main(["generalized-rules", baskets, "--taxonomy", taxonomy, "--min-supp", "0.6", "--level-supp", "2=0.3", "3=0.3",
                 "--min-threshold", "0.6", "--max-len", "2", "--format", "json"]) == 0
    rules = json.loads(capsys.readouterr().out)
    # con 0.6 solo en el nivel 1, solo sobreviven las reglas entre items de los niveles 2-3 (umbral 0.3)
    assert {(r["antecedents"], r["consequents"]) for r in rules} == {("Hiking Boots", "Outerwear"), ("Outerwear", "Hiking Boots")}
    assert main(["generalized-rules", baskets, "--taxonomy", taxonomy, "--alpha", "0.5", "--min-supp", "0.3",
                 "--min-threshold", "0.6", "--prune"]) == 0
    capsys.readouterr()
    assert main(["generalized-rules", baskets, "--taxonomy", taxonomy, "--level-supp", "2:0.1"]) == 1
    assert "NIVEL=SOPORTE" in capsys.readouterr().err
    assert main(["generalized-rules", baskets, "--taxonomy", taxonomy, "--interest", "0.5"]) == 1
    assert "interest" in capsys.readouterr().err
    assert main(["generalized-rules", baskets, "--taxonomy", baskets + ".missing"]) == 1
    with pytest.raises(SystemExit):                                              # --taxonomy es obligatorio
        main(["generalized-rules", baskets])
    capsys.readouterr()


# ----------------------------------------------- reglas cuantitativas (intervalos)

@pytest.fixture
def customers_file(tmp_path):
    """La edad entre 30 y 45 anos marca los ingresos altos; las mujeres con ingresos altos compran."""
    rng = np.random.RandomState(0)
    rows = 2000
    age = rng.uniform(18, 70, rows)
    income = np.where((age >= 30) & (age < 45), rng.normal(70, 8, rows), rng.normal(35, 10, rows)).clip(5)
    sex = rng.choice(["F", "M"], rows)
    buys = np.where((income > 60) & (sex == "F"), rng.rand(rows) < 0.9, rng.rand(rows) < 0.1)
    path = tmp_path / "customers.csv"
    pd.DataFrame({"age": age, "income": income, "sex": sex, "buys": np.where(buys, "yes", "no")}).to_csv(path, index=False)
    return str(path)


def test_quantitative_rules_command(customers_file, capsys):
    base = ["quantitative-rules", customers_file, "--intervals", "10", "--max-supp", "0.4", "--min-supp", "0.05",
            "--min-threshold", "0.7", "--max-len", "3"]
    assert main(base + ["--format", "json"]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert any(r["consequents"] == "buys=yes" and set(r["antecedents"].split(", ")) >= {"sex=F"} and r["confidence"] > 0.85 for r in rules)

    assert main(base + ["--interest", "1.3", "--format", "json"]) == 0
    kept = json.loads(capsys.readouterr().out)
    assert 0 < len(kept) < len(rules) / 2 and all("interest" in r for r in kept)
    assert main(base + ["--interest", "1.3", "--top", "2", "--sort-by", "lift"]) == 0
    table = capsys.readouterr().out.strip().splitlines()
    assert len(table) == 3 and "interest" in table[0]

    assert main(base + ["--columns", "age", "income", "--method", "uniform", "--quantitative", "age", "income", "--prune",
                        "--alpha", "0.05"]) == 0
    capsys.readouterr()
    assert main(base + ["--columns", "nope"]) == 1
    assert "inexistentes" in capsys.readouterr().err
    assert main(base + ["--interest", "0.5"]) == 1


# ------------------------------------------- reglas top-k, raras y de alta utilidad

@pytest.fixture
def pairs_file(tmp_path):
    """a y d son frecuentes por separado (0.30) y juntos no (0.15); b es muy frecuente y va solo."""
    path = tmp_path / "pairs.txt"
    path.write_text("\n".join(["a,d"] * 15 + ["a"] * 15 + ["d"] * 15 + ["b"] * 55) + "\n")
    return str(path)


@pytest.fixture
def gems_files(tmp_path):
    """3 ventas de una gema y un oro, y 40 de un boligrafo y una gorra: lo que vale no es lo que se vende mas."""
    rows = [{"gem": 1, "gold": 1, "pen": 0, "cap": 0}] * 3 + [{"gem": 0, "gold": 0, "pen": 2, "cap": 1}] * 40
    quantities, profits = tmp_path / "sales.csv", tmp_path / "profits.csv"
    pd.DataFrame(rows).to_csv(quantities, index=False)
    pd.DataFrame({"item": ["gem", "gold", "pen", "cap"], "profit": [500, 400, 1, 1]}).to_csv(profits, index=False)
    return str(quantities), str(profits)


def test_top_k_rules_command(pairs_file, capsys):
    assert main(["top-k-rules", pairs_file, "-k", "5", "--min-conf", "0.5", "--format", "json"]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert [(r["antecedents"], r["consequents"]) for r in rules] == [("a", "d"), ("d", "a")]
    assert all(abs(r["support"] - 0.15) < 1e-9 and abs(r["confidence"] - 0.5) < 1e-9 for r in rules)
    assert main(["top-k-rules", pairs_file, "-k", "1", "--min-conf", "0.5", "--max-len", "0"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2                      # cabecera y una regla
    assert main(["top-k-rules", pairs_file, "-k", "5", "--min-conf", "0.5", "--alpha", "0.05"]) == 0
    assert "p_adjusted" in capsys.readouterr().out
    assert main(["top-k-rules", pairs_file, "-k", "5", "--min-conf", "0.5", "--min-lift", "2"]) == 0
    assert "no rules found" in capsys.readouterr().out
    assert main(["top-k-rules", pairs_file, "-k", "0"]) == 1
    assert "k debe ser" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["top-k-rules", pairs_file])                                              # -k es obligatorio


def test_rare_rules_command(pairs_file, capsys):
    base = ["rare-rules", pairs_file, "--max-supp", "0.2", "--min-conf", "0.5"]
    assert main(base + ["--format", "json"]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert {(r["antecedents"], r["consequents"]) for r in rules} == {("a", "d"), ("d", "a")}
    assert all(r["support"] < 0.2 and abs(r["lift"] - 0.5 / 0.3) < 1e-9 for r in rules)
    assert main(base + ["--min-count", "16"]) == 0
    assert "no rules found" in capsys.readouterr().out
    assert main(base + ["--alpha", "0.05", "--max-antecedent", "1", "--max-consequent", "1"]) == 0
    assert "p_adjusted" in capsys.readouterr().out
    assert main(["rare-rules", pairs_file, "--max-supp", "2"]) == 1
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["rare-rules", pairs_file])                                               # --max-supp es obligatorio


def test_high_utility_rules_command(gems_files, capsys):
    sales, profits = gems_files
    assert main(["high-utility-rules", sales, "--profits", profits, "--min-utility", "0.5", "--min-conf", "0.9",
                 "--format", "json"]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert {(r["antecedents"], r["consequents"]) for r in rules} == {("gem", "gold"), ("gold", "gem")}
    assert all(r["utility"] == 2700 and abs(r["utility_share"] - 2700 / 2820) < 1e-9 for r in rules)
    # sin beneficios todos valen 1: ahora el valioso es lo que mas se vende
    assert main(["high-utility-rules", sales, "--min-utility", "0.5", "--min-conf", "0.9", "--format", "json"]) == 0
    unit = json.loads(capsys.readouterr().out)
    assert {(r["antecedents"], r["consequents"]) for r in unit} == {("pen", "cap"), ("cap", "pen")}
    assert main(["high-utility-rules", sales, "--profits", profits, "--min-utility", "2000", "--absolute", "--top", "1"]) == 0
    assert "utility_share" in capsys.readouterr().out.splitlines()[0]
    assert main(["high-utility-rules", sales, "--profits", profits, "--min-utility", "0.5", "--alpha", "0.05"]) == 0
    capsys.readouterr()


def test_high_utility_rules_errors(gems_files, tmp_path, capsys):
    sales, profits = gems_files
    assert main(["high-utility-rules", sales, "--min-utility", "0"]) == 1
    assert "min_utility" in capsys.readouterr().err
    short = tmp_path / "short.csv"
    short.write_text("gem,gold\n500,400\n")
    assert main(["high-utility-rules", sales, "--profits", str(short)]) == 1                 # faltan pen y cap
    assert "faltan los beneficios" in capsys.readouterr().err
    one_column = tmp_path / "one.csv"
    one_column.write_text("gem\n500\n")
    assert main(["high-utility-rules", sales, "--profits", str(one_column)]) == 1
    assert "primera columna" in capsys.readouterr().err
    assert main(["high-utility-rules", str(tmp_path / "nothing.csv")]) == 1


def test_generate_command_writes_files_that_other_commands_read(tmp_path, capsys):
    baskets, sequences = tmp_path / "tx.txt", tmp_path / "seq.txt"
    assert main(["generate", "transactions", "-n", "300", "--items", "20", "--length", "5", "--seed", "1",
                 "-o", str(baskets)]) == 0
    assert len(baskets.read_text().splitlines()) == 300
    again = tmp_path / "again.txt"
    main(["generate", "transactions", "-n", "300", "--items", "20", "--length", "5", "--seed", "1", "-o", str(again)])
    assert again.read_text() == baskets.read_text()
    assert main(["rules", str(baskets), "--min-supp", "0.2", "--min-threshold", "0.6", "--format", "csv"]) == 0
    assert "antecedents" in capsys.readouterr().out

    assert main(["generate", "sequences", "-n", "100", "--items", "10", "--seed", "2", "-o", str(sequences)]) == 0
    assert main(["sequences", str(sequences), "--min-supp", "0.3", "--format", "csv"]) == 0
    capsys.readouterr()


def test_generate_quantities_and_stdout(tmp_path, capsys):
    sales, profits = tmp_path / "sales.csv", tmp_path / "profits.csv"
    assert main(["generate", "quantities", "-n", "100", "--items", "8", "--seed", "3", "-o", str(sales),
                 "--profits", str(profits)]) == 0
    assert pd.read_csv(sales).shape == (100, 8) and list(pd.read_csv(profits).columns) == ["item", "profit"]
    assert main(["high-utility-rules", str(sales), "--profits", str(profits), "--min-utility", "0.1",
                 "--min-conf", "0.3", "--format", "csv"]) == 0
    capsys.readouterr()
    assert main(["generate", "transactions", "-n", "5", "--items", "6", "--seed", "1"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5
