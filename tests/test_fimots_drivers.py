# -*- coding: utf-8 -*-
"""
End-to-end test of the main programs of FIMoTS (``SARE.main`` and ``SFIM.main``)
on a real Spark, each in its own process (``main`` creates its SparkContext, and there can be only
one per process): the TEXTFILE mode (sliding window over files) and the STREAMING mode
(Spark Streaming watching a directory into which new files keep arriving).
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
    pytest.skip("No JDK available (set JAVA_HOME)", allow_module_level=True)

from test_fimots_streaming import brute_force_frequent  # noqa: E402

BATCH_ONE = ["a b c"] * 9 + ["a b"] * 5 + ["a"] * 3 + ["c d"] * 3
BATCH_TWO = ["a b"] * 8 + ["b c"] * 5 + ["a b c"] * 4 + ["d"] * 3


def _write(path, lines):
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _frequent_itemsets_printed(output):
    """Itemsets under the last 'Frequent itemsets:' header of the FIMoTS output (up to the blank line)."""
    lines = output.replace("\r", "\n").splitlines()
    start = max(i for i, line in enumerate(lines) if line.startswith("Frequent itemsets:"))
    items = set()
    for line in lines[start + 1:]:
        if not line.strip() or line.startswith(("Association", "Received", "Current")):
            break
        items.add("_".join(line.split()))
    return items


@pytest.mark.parametrize("package", ["SARE", "SFIM"])
def test_textfile_mode_mines_a_directory_of_files(tmp_path, package):
    # filesDir WITHOUT a trailing slash: paths used to be built with dirPath + fileName and failed
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
        # Until the algorithm has published frequent itemsets and rules (Spark startup included)
        while time.time() < deadline:
            text = "".join(output).replace("\r", "\n")
            if "b --> a" in text and "Total frequent itemsets: 0" not in text.rsplit("Current state", 1)[-1]:
                break
            if sent < 12 and "Current state" in text:
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
    assert "Received transactions: 20" in text
    assert {"a", "b", "a_b"} <= _frequent_itemsets_printed(text)
    assert "Association rules (confidence >= 0.8):" in text
