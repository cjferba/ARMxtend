# -*- coding: utf-8 -*-
"""
Prueba de extremo a extremo de los programas principales de FIMoTS (``SARE.main`` y ``SFIM.main``)
sobre un Spark real, cada uno en su propio proceso (``main`` crea su SparkContext, y solo puede
haber uno por proceso): el modo TEXTFILE (ventana deslizante sobre ficheros) y el modo STREAMING
(Spark Streaming vigilando un directorio al que van llegando ficheros nuevos).
"""
import os
import subprocess
import sys
import threading
import time

import pytest

from conftest import java_available

pytest.importorskip("pyspark")
if not java_available():
    pytest.skip("No hay JDK disponible (defina JAVA_HOME)", allow_module_level=True)

from test_fimots_streaming import brute_force_frequent  # noqa: E402

BATCH_ONE = ["a b c"] * 9 + ["a b"] * 5 + ["a"] * 3 + ["c d"] * 3
BATCH_TWO = ["a b"] * 8 + ["b c"] * 5 + ["a b c"] * 4 + ["d"] * 3


def _write(path, lines):
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _frequent_itemsets_printed(output):
    """Itemsets bajo la ultima cabecera 'Itemsets Frecuentes:' de la salida de FIMoTS (hasta la linea en blanco)."""
    lines = output.replace("\r", "\n").splitlines()
    start = max(i for i, line in enumerate(lines) if line.startswith("Itemsets Frecuentes:"))
    items = set()
    for line in lines[start + 1:]:
        if not line.strip() or line.startswith(("Reglas", "Transacciones", "Estado")):
            break
        items.add("_".join(line.split()))
    return items


@pytest.mark.parametrize("package", ["SARE", "SFIM"])
def test_textfile_mode_mines_a_directory_of_files(tmp_path, package):
    # filesDir SIN barra final: antes las rutas se construian con dirPath + fileName y fallaban
    _write(tmp_path / "batch1.txt", BATCH_ONE)
    _write(tmp_path / "batch2.txt", BATCH_TWO)
    code = ("from ARMxtend.{} import main; main(minSuppNum=3, minSuppDen=10, mode='TEXTFILE', intervals=2, "
            "filesDir={!r}, threads=2)").format(package, str(tmp_path))

    result = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True, timeout=240)

    assert result.returncode == 0, result.stderr[-2000:]
    assert _frequent_itemsets_printed(result.stdout) == brute_force_frequent(BATCH_ONE + BATCH_TWO, 0.3)


def test_streaming_mode_processes_files_that_arrive_over_time(tmp_path):
    watched = tmp_path / "incoming"
    watched.mkdir()
    code = ("from ARMxtend.SARE import main; main(minSuppNum=3, minSuppDen=10, mode='STREAMING', intervals=2, "
            "filesDir={!r}, threads=2, seconds=1, minConf=0.8)").format(str(watched))
    process = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=dict(os.environ))
    output = []
    reader = threading.Thread(target=lambda: output.extend(iter(process.stdout.readline, "")), daemon=True)
    reader.start()

    try:
        deadline = time.time() + 150
        sent = 0
        # Hasta que el algoritmo haya publicado itemsets frecuentes y reglas (arranque de Spark incluido)
        while time.time() < deadline:
            text = "".join(output).replace("\r", "\n")
            if "b --> a" in text and "Total de Itemsets Frecuentes: 0" not in text.rsplit("Estado Actual", 1)[-1]:
                break
            if sent < 12 and "Estado Actual" in text:
                staging = tmp_path / "staging.txt"
                _write(staging, BATCH_ONE if sent % 2 == 0 else BATCH_TWO)
                os.rename(staging, watched / "batch{}.txt".format(sent))     # aparicion atomica
                sent += 1
            time.sleep(1.5)
        text = "".join(output).replace("\r", "\n")
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    assert "Traceback" not in text
    assert "Transacciones Recibidas: 20" in text
    assert {"a", "b", "a_b"} <= _frequent_itemsets_printed(text)
    assert "Reglas de Asociacion (confianza >= 0.8):" in text
