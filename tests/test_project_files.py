# -*- coding: utf-8 -*-
"""Project files: CITATION.cff, example notebook and comparison with mlxtend."""
import json
import os
import re
import sys

import pytest

import ARMxtend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_citation_file_matches_the_package():
    text = open(os.path.join(ROOT, "CITATION.cff"), encoding="utf-8").read()
    assert re.search(r"^cff-version: 1\.2\.0$", text, re.M)
    assert re.search(r"^version: {}$".format(re.escape(ARMxtend.__version__)), text, re.M)
    assert "repository-code: \"https://github.com/cjferba/ARMxtend\"" in text
    assert "10.1007/s10586-023-04014-w" in text and "10.1016/j.knosys.2018.09.026" in text


def test_notebook_is_valid_and_was_executed():
    notebook = json.load(open(os.path.join(ROOT, "examples", "notebooks", "quickstart.ipynb"), encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert notebook["nbformat"] == 4 and len(code) >= 5
    assert all(cell["outputs"] and not any(o["output_type"] == "error" for o in cell["outputs"]) for cell in code)


def test_mlxtend_comparison_gives_identical_results():
    pytest.importorskip("mlxtend")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import compare_mlxtend

    result = compare_mlxtend.main(["--sizes", "300", "--items", "20", "--min-supp", "0.1", "--repeat", "1"])
    assert result["same itemsets"].all() and result["same rules"].all() and result["rules"].min() > 0
