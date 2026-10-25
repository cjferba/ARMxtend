# -*- coding: utf-8 -*-
"""Fixtures compartidas: un SparkContext real (local[2]) para los tests de integracion."""
import os
import subprocess
import sys

import pytest


def java_available():
    # No basta con shutil.which("java"): macOS trae un /usr/bin/java que
    # existe aunque no haya ningun JDK instalado y falla al ejecutarse.
    java = os.path.join(os.environ["JAVA_HOME"], "bin", "java") if os.environ.get("JAVA_HOME") else "java"
    try:
        return subprocess.run([java, "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.fixture(scope="session")
def real_sc():
    """SparkContext real en modo local[2]; se omite el test si no hay pyspark o un JDK >= 17 (defina JAVA_HOME)."""
    pyspark = pytest.importorskip("pyspark")
    if not java_available():
        pytest.skip("No hay JDK disponible (defina JAVA_HOME)")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    conf = pyspark.SparkConf().setMaster("local[2]").setAppName("armxtend-tests").set("spark.ui.enabled", "false")
    context = pyspark.SparkContext(conf=conf)
    context.setLogLevel("ERROR")
    yield context
    context.stop()
