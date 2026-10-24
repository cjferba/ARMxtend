# -*- coding: utf-8 -*-
"""
Interfaz de linea de comandos de ARMxtend::

    armxtend rules transacciones.csv --min-supp 0.1 --min-threshold 0.7
    armxtend fuzzy-rules datos.csv --labels 3 --alpha 10 --metric certainty_factor
    armxtend gradual-rules datos.csv --min-supp 0.3 --min-threshold 0.8

Tambien se puede invocar como ``python -m ARMxtend``. Todos los comandos
trabajan en una sola maquina (sin Spark) y escriben las reglas en una tabla
de texto (por defecto), CSV o JSON; `rules` y `fuzzy-rules` pueden ademas
generar el visor HTML interactivo VizARE (`--html`).

@author Carlos Fernandez-Basso
"""
import argparse
import json
import sys
from typing import List, Optional, Sequence

import pandas as pd

from . import __version__

_ITEM_COLUMNS = ("antecedents", "consequents")


def _read_transactions(path: str, delimiter: str) -> List[List[str]]:
    """Una transaccion por linea, items separados por `delimiter`; se ignoran las lineas vacias."""
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    try:
        lines = [line.strip() for line in stream]
    finally:
        if stream is not sys.stdin:
            stream.close()
    return [[item.strip() for item in line.split(delimiter) if item.strip()] for line in lines if line]


def _format_items(items) -> str:
    from .ARM.gradual_rules import format_gradual_itemset

    items = list(items)
    if items and isinstance(items[0], tuple):          # items graduales: ('atributo', '+'|'-')
        return format_gradual_itemset(items)
    return ", ".join(sorted(items))


def _emit(rules: pd.DataFrame, args: argparse.Namespace) -> None:
    """Ordena, recorta y escribe las reglas en el formato pedido."""
    sortBy = args.sort_by or args.metric
    if len(rules) and sortBy in rules.columns:
        rules = rules.sort_values(sortBy, ascending=False)
    if args.top:
        rules = rules.head(args.top)
    rules = rules.reset_index(drop=True)

    flat = rules.copy()
    for column in _ITEM_COLUMNS:
        flat[column] = [_format_items(items) for items in rules[column]]

    if args.format == "json":
        text = json.dumps(json.loads(flat.to_json(orient="records")), indent=2, ensure_ascii=False)
    elif args.format == "csv":
        text = flat.to_csv(index=False)
    else:
        if not args.all_measures:
            core = list(_ITEM_COLUMNS) + [c for c in ("support", "confidence", "lift", "certainty_factor", args.metric)
                                          if c in flat.columns]
            flat = flat[list(dict.fromkeys(core))]
        text = flat.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)) if len(flat) \
            else "(no rules found)"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)

    if getattr(args, "html", None):
        from .VizARM import AREtoGraph

        measures = [column for column in rules.columns if column not in _ITEM_COLUMNS]
        AREtoGraph.from_dataframe(rules, rule_measures=measures).to_html(args.html, title="ARMxtend - " + args.command)
        print("Interactive viewer written to {}".format(args.html), file=sys.stderr)


def _run_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import association_rules, prune_redundant_rules
    from .FFIM import fpgrowth
    from .FIM._shared import key_to_itemset

    transactions = _read_transactions(args.input, args.delimiter)
    frequent = fpgrowth(transactions, args.min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()),
                             "itemsets": [key_to_itemset(key) for key in frequent]})
    rules = association_rules(itemsets, metric=args.metric, min_threshold=args.min_threshold)
    return prune_redundant_rules(rules) if args.prune and len(rules) else rules


def _run_fuzzy_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import prune_redundant_rules
    from .FFIM import fuzzy_fpgrowth
    from .FIM.FARE import fuzzy_association_rules
    from .preprocessing import FuzzyLib

    library = FuzzyLib()
    library.data = pd.read_csv(args.input, sep=args.delimiter)
    library.AutoFuzzification(Atri=args.columns, n_labels=args.labels, method=args.method,
                              membership=args.membership)
    transactions = library.GetFuzzyTransactions()
    frequent = fuzzy_fpgrowth(transactions, args.min_supp, args.alpha)
    rules = fuzzy_association_rules(frequent, args.alpha, metric=args.metric, min_threshold=args.min_threshold,
                                    measures=args.measures)
    return prune_redundant_rules(rules) if args.prune and len(rules) else rules


def _run_gradual_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import gradual_rules

    table = pd.read_csv(args.input, sep=args.delimiter)
    return gradual_rules(table, args.min_supp, args.min_threshold, attributes=args.columns,
                         max_rows=args.max_rows)


def _common_output_options(parser: argparse.ArgumentParser, metric: str) -> None:
    parser.add_argument("-d", "--delimiter", default=",", help="field/item delimiter (default: ',')")
    parser.add_argument("--min-supp", type=float, default=0.1, help="minimum support (default: 0.1)")
    parser.add_argument("--metric", default=metric, help="measure used to filter and sort the rules (default: %(default)s)")
    parser.add_argument("--min-threshold", type=float, default=0.7,
                        help="minimum value of --metric (default: 0.7)")
    parser.add_argument("--sort-by", help="column to sort by, descending (default: --metric)")
    parser.add_argument("--top", type=int, help="keep only the first N rules after sorting")
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table", help="output format")
    parser.add_argument("--all-measures", action="store_true",
                        help="in the table format, show every measure column (csv/json always do)")
    parser.add_argument("-o", "--output", help="write the rules to this file instead of stdout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="armxtend", description="Association rule mining from the command line (ARMxtend).")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.required = True

    rules = commands.add_parser("rules", help="crisp association rules from a file of transactions")
    rules.add_argument("input", help="file with one transaction per line, items separated by --delimiter ('-' = stdin)")
    _common_output_options(rules, "confidence")
    rules.add_argument("--prune", action="store_true", help="drop redundant rules")
    rules.add_argument("--html", metavar="FILE", help="also write the interactive VizARE viewer to FILE")
    rules.set_defaults(run=_run_rules)

    fuzzy = commands.add_parser(
        "fuzzy-rules", help="fuzzy association rules from a CSV of numeric columns (automatic fuzzification)")
    fuzzy.add_argument("input", help="CSV file with a header; numeric columns are fuzzified")
    _common_output_options(fuzzy, "confidence")
    fuzzy.add_argument("--columns", nargs="+", help="columns to fuzzify (default: all numeric columns)")
    fuzzy.add_argument("--labels", type=int, default=3, help="fuzzy labels per column (default: 3)")
    fuzzy.add_argument("--method", choices=("uniform", "quantile", "kmeans"), default="quantile",
                       help="how the partition is computed from the data (default: quantile)")
    fuzzy.add_argument("--membership", choices=("triangular", "trapezoidal", "gaussian"), default="triangular")
    fuzzy.add_argument("--alpha", type=int, default=10, help="number of alpha-cuts (default: 10)")
    fuzzy.add_argument("--measures", nargs="+", help="extra interest measures to add as columns")
    fuzzy.add_argument("--prune", action="store_true", help="drop redundant rules")
    fuzzy.add_argument("--html", metavar="FILE", help="also write the interactive VizARE viewer to FILE")
    fuzzy.set_defaults(run=_run_fuzzy_rules)

    gradual = commands.add_parser("gradual-rules", help="gradual rules ('the more A, the more B') from numeric columns")
    gradual.add_argument("input", help="CSV file with a header")
    _common_output_options(gradual, "confidence")
    gradual.add_argument("--columns", nargs="+", help="columns to use (default: all numeric columns)")
    gradual.add_argument("--max-rows", type=int, help="work on a random sample of this many rows")
    gradual.set_defaults(run=_run_gradual_rules)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Punto de entrada de la CLI; devuelve el codigo de salida (0 = correcto)."""
    args = build_parser().parse_args(argv)
    try:
        rules = args.run(args)
        _emit(rules, args)
    except (ValueError, OSError, KeyError) as error:
        print("armxtend: error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
