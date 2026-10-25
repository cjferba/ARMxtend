# -*- coding: utf-8 -*-
"""
Interfaz de linea de comandos de ARMxtend::

    armxtend rules transacciones.csv --min-supp 0.1 --min-threshold 0.7
    armxtend fuzzy-rules datos.csv --labels 3 --alpha 10 --metric certainty_factor
    armxtend gradual-rules datos.csv --min-supp 0.3 --min-threshold 0.8
    armxtend temporal-rules sensores.csv --time-column hora --max-lag 2 --targets calefaccion
    armxtend rule-drift sensores.csv --time-column hora --window 15D --max-lag 1
    armxtend sequences rutinas.txt --min-supp 0.3
    armxtend sequential-rules rutinas.txt --min-supp 0.3 --min-threshold 0.8

Tambien se puede invocar como ``python -m ARMxtend``. Todos los comandos
trabajan en una sola maquina (sin Spark) y escriben el resultado en una tabla
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


def _read_lines(path: str) -> List[str]:
    """Las lineas no vacias de un fichero (o de la entrada estandar si es '-'), sin espacios en los extremos."""
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    try:
        lines = [line.strip() for line in stream]
    finally:
        if stream is not sys.stdin:
            stream.close()
    return [line for line in lines if line]


def _read_transactions(path: str, delimiter: str) -> List[List[str]]:
    """Una transaccion por linea, items separados por `delimiter`; se ignoran las lineas vacias."""
    return [[item.strip() for item in line.split(delimiter) if item.strip()] for line in _read_lines(path)]


def _read_sequences(args: argparse.Namespace) -> list:
    """
    Secuencias desde un fichero de texto (una secuencia por linea, elementos separados por
    `--element-delimiter` e items de un elemento por `--delimiter`) o, con `--sequence-column`,
    desde un registro de eventos en CSV.
    """
    from .FIM.sequential import sequences_from_events

    if args.sequence_column:
        if not (args.time_column and args.item_column):
            raise ValueError("--sequence-column necesita tambien --time-column y --item-column")
        events = pd.read_csv(args.input, sep=args.delimiter)
        if args.time_column in events.columns and events[args.time_column].dtype == object:
            events[args.time_column] = pd.to_datetime(events[args.time_column])
        return sequences_from_events(events, args.sequence_column, args.time_column, args.item_column, bin=args.bin)
    return [[[item.strip() for item in element.split(args.delimiter) if item.strip()]
             for element in line.split(args.element_delimiter)] for line in _read_lines(args.input)]


def _read_items(args: argparse.Namespace) -> pd.DataFrame:
    """
    Tabla de items booleanos desde un CSV de sensores: las columnas numericas se simbolizan
    (`--symbols`, `--method`), las 0/1 y las booleanas son eventos, las de texto dan items
    ``columna=valor``; `--time-column` fija el indice temporal y `--calendar` anade items de calendario.
    """
    from .preprocessing.timeseries import calendar_items, discretize, to_items

    table = pd.read_csv(args.input, sep=args.delimiter)
    if args.time_column:
        if args.time_column not in table.columns:
            raise ValueError("la columna de tiempo '{}' no existe".format(args.time_column))
        table = table.set_index(pd.to_datetime(table.pop(args.time_column))).sort_index()
    columns = list(args.columns) if args.columns else list(table.columns)
    missing = [name for name in columns if name not in table.columns]
    if missing:
        raise ValueError("columnas inexistentes: {}".format(", ".join(missing)))
    table = table[columns].copy()
    numeric = []
    for name in columns:
        column = table[name]
        if pd.api.types.is_bool_dtype(column) or not pd.api.types.is_numeric_dtype(column):
            continue
        if set(column.dropna().unique()) <= {0, 1}:
            table[name] = column.fillna(0).astype(bool)                   # evento 0/1
        else:
            numeric.append(name)
    items = to_items(discretize(table, numeric, n_symbols=args.symbols, method=args.method))
    if args.calendar:
        if not isinstance(items.index, pd.DatetimeIndex):
            raise ValueError("--calendar necesita --time-column")
        items = items.join(calendar_items(items.index, args.calendar))
    return items


def _format_items(items, pretty: bool = False) -> str:
    from .ARM.gradual_rules import format_gradual_itemset
    from .ARM.temporal_rules import format_lagged_item

    items = list(items)
    if items and isinstance(items[0], tuple):          # items graduales: ('atributo', '+'|'-')
        return format_gradual_itemset(items)
    if pretty:                                         # 'temp=low@lag1' -> 'temp=low (t-1)'
        items = [format_lagged_item(item) for item in items]
    return ", ".join(sorted(items))


def _emit(rules: pd.DataFrame, args: argparse.Namespace) -> None:
    """Ordena, recorta y escribe las reglas en el formato pedido."""
    sortBy = args.sort_by or ("" if getattr(args, "keep_order", False) else args.metric)
    if len(rules) and sortBy and sortBy in rules.columns:
        rules = rules.sort_values(sortBy, ascending=False)
    if args.top:
        rules = rules.head(args.top)
    rules = rules.reset_index(drop=True)

    flat = rules.copy()
    for column in _ITEM_COLUMNS:
        if column in flat.columns:
            flat[column] = [_format_items(items, pretty=args.format == "table" and getattr(args, "pretty_lags", False))
                            for items in rules[column]]
    if "sequence" in flat.columns:
        from .FIM.sequential import format_sequence

        flat["sequence"] = [format_sequence(sequence) for sequence in rules["sequence"]]

    if args.format == "json":
        text = json.dumps(json.loads(flat.to_json(orient="records")), indent=2, ensure_ascii=False)
    elif args.format == "csv":
        text = flat.to_csv(index=False)
    else:
        if not args.all_measures:
            core = getattr(args, "table_columns", None) or (
                list(_ITEM_COLUMNS) + ["support", "confidence", "lift", "certainty_factor", args.metric])
            flat = flat[[c for c in dict.fromkeys(core) if c in flat.columns]]
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


def _run_temporal_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import prune_redundant_rules, temporal_association_rules

    args.pretty_lags = True
    rules = temporal_association_rules(
        _read_items(args), args.max_lag, args.min_supp, metric=args.metric, min_threshold=args.min_threshold,
        lags=args.lags, targets=args.targets, require_lagged_antecedent=not args.allow_no_lag,
        exclude_persistence=args.exclude_persistence)
    return prune_redundant_rules(rules) if args.prune and len(rules) else rules


def _window_size(text: str):
    """'20' -> 20 filas; '7D', '12h', ... -> duracion (``window_rules`` exige un indice temporal)."""
    return int(text) if text.isdigit() else text


def _run_rule_drift(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import rule_drift, temporal_association_rules, window_rules

    miner = None
    if args.max_lag > 0:
        def miner(window):
            return temporal_association_rules(window, args.max_lag, args.min_supp, metric=args.metric,
                                              min_threshold=args.min_threshold, targets=args.targets,
                                              exclude_persistence=args.exclude_persistence)

    windows = window_rules(_read_items(args), _window_size(args.window),
                           _window_size(args.step) if args.step else None, min_supp=args.min_supp,
                           metric=args.metric, min_threshold=args.min_threshold, miner=miner)
    drift = rule_drift(windows, measure=args.metric, stable_fraction=args.stable_fraction)
    args.pretty_lags = args.max_lag > 0
    args.table_columns = ["antecedents", "consequents", "status", "windows", "persistence", "first_window",
                          "last_window", "mean_" + args.metric, "slope"]
    return drift


def _run_sequences(args: argparse.Namespace) -> pd.DataFrame:
    from .FIM.sequential import prefixspan

    return prefixspan(_read_sequences(args), args.min_supp, max_len=args.max_len, min_len=args.min_len)


def _run_sequential_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import sequential_rules

    return sequential_rules(_read_sequences(args), args.min_supp,
                            min_conf=args.min_threshold if args.metric == "confidence" else 0.0,
                            max_antecedent=args.max_antecedent, max_consequent=args.max_consequent,
                            metric=args.metric, min_threshold=args.min_threshold)


def _output_options(parser: argparse.ArgumentParser, what: str = "rules") -> None:
    parser.add_argument("--sort-by", help="column to sort by, descending")
    parser.add_argument("--top", type=int, help="keep only the first N {} after sorting".format(what))
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table", help="output format")
    parser.add_argument("--all-measures", action="store_true",
                        help="in the table format, show every column (csv/json always do)")
    parser.add_argument("-o", "--output", help="write the {} to this file instead of stdout".format(what))


def _common_output_options(parser: argparse.ArgumentParser, metric: str) -> None:
    parser.add_argument("-d", "--delimiter", default=",", help="field/item delimiter (default: ',')")
    parser.add_argument("--min-supp", type=float, default=0.1, help="minimum support (default: 0.1)")
    parser.add_argument("--metric", default=metric, help="measure used to filter and sort the rules (default: %(default)s)")
    parser.add_argument("--min-threshold", type=float, default=0.7,
                        help="minimum value of --metric (default: 0.7)")
    _output_options(parser)


def _series_options(parser: argparse.ArgumentParser) -> None:
    """Opciones para convertir un CSV de sensores en items (temporal-rules y rule-drift)."""
    parser.add_argument("input", help="CSV file with a header, one row per time step")
    parser.add_argument("--time-column", help="column with the timestamps (becomes the index; rows are sorted by it)")
    parser.add_argument("--columns", nargs="+", help="columns to use (default: all)")
    parser.add_argument("--symbols", type=int, default=3,
                        help="symbols per numeric column, e.g. low/medium/high (default: 3)")
    parser.add_argument("--method", choices=("quantile", "uniform", "sax"), default="quantile",
                        help="how numeric columns are discretised (default: quantile)")
    parser.add_argument("--calendar", nargs="+", choices=("hour", "weekday", "daytype", "month", "part_of_day"),
                        help="add calendar items (needs --time-column)")


def _sequence_options(parser: argparse.ArgumentParser) -> None:
    """Opciones de entrada de sequences y sequential-rules."""
    parser.add_argument("input", help="text file, one sequence per line: elements separated by --element-delimiter, "
                                      "items of an element by --delimiter ('-' = stdin); or, with --sequence-column, "
                                      "a CSV event log")
    parser.add_argument("-d", "--delimiter", default=",", help="item (or CSV field) delimiter (default: ',')")
    parser.add_argument("--element-delimiter", default=";", help="element delimiter in text files (default: ';')")
    parser.add_argument("--sequence-column", help="event-log mode: column identifying the sequence (patient, day...)")
    parser.add_argument("--time-column", help="event-log mode: column with the event time")
    parser.add_argument("--item-column", help="event-log mode: column with the event")
    parser.add_argument("--bin", help="event-log mode: events of a sequence within the same interval ('5min', '1h') "
                                      "form a single element")
    parser.add_argument("--min-supp", type=float, default=0.1, help="minimum support (default: 0.1)")


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

    temporal = commands.add_parser(
        "temporal-rules", help="lagged rules ('A at t-2 => B at t') from a CSV of sensor/time-series data")
    _series_options(temporal)
    _common_output_options(temporal, "confidence")
    temporal.add_argument("--max-lag", type=int, default=1, help="how many steps back antecedents may look (default: 1)")
    temporal.add_argument("--lags", type=int, nargs="+", help="only these lags (default: 1..--max-lag)")
    temporal.add_argument("--targets", nargs="+", help="items the consequent is restricted to")
    temporal.add_argument("--allow-no-lag", action="store_true",
                          help="also keep rules whose antecedent has no lagged item")
    temporal.add_argument("--exclude-persistence", action="store_true",
                          help="drop rules that merely say a state persists (x at t-k => x at t)")
    temporal.add_argument("--prune", action="store_true", help="drop redundant rules")
    temporal.set_defaults(run=_run_temporal_rules)

    drift = commands.add_parser(
        "rule-drift", help="how rules appear, persist and vanish across consecutive time windows")
    _series_options(drift)
    drift.add_argument("-d", "--delimiter", default=",", help="CSV delimiter (default: ',')")
    drift.add_argument("--window", required=True, help="window width: a number of rows ('96') or a duration ('15D', '12h')")
    drift.add_argument("--step", help="distance between window starts (default: --window, i.e. no overlap)")
    drift.add_argument("--min-supp", type=float, default=0.1, help="minimum support per window (default: 0.1)")
    drift.add_argument("--metric", default="confidence", help="measure to filter rules and track over time (default: %(default)s)")
    drift.add_argument("--min-threshold", type=float, default=0.8, help="minimum value of --metric (default: 0.8)")
    drift.add_argument("--max-lag", type=int, default=0,
                       help="0 (default): plain rules in each window; N > 0: lagged rules up to N steps back")
    drift.add_argument("--targets", nargs="+", help="with --max-lag: items the consequent is restricted to")
    drift.add_argument("--exclude-persistence", action="store_true", help="with --max-lag: drop persistence rules")
    drift.add_argument("--stable-fraction", type=float, default=0.8,
                       help="share of windows a rule must appear in to be 'stable' (default: 0.8)")
    _output_options(drift)
    drift.set_defaults(run=_run_rule_drift, keep_order=True)

    sequences = commands.add_parser("sequences", help="frequent sequential patterns (PrefixSpan)")
    _sequence_options(sequences)
    sequences.add_argument("--max-len", type=int, help="maximum number of items in a pattern")
    sequences.add_argument("--min-len", type=int, default=1, help="minimum number of items in a pattern (default: 1)")
    _output_options(sequences, "patterns")
    sequences.set_defaults(run=_run_sequences, metric=None, keep_order=True,
                           table_columns=["sequence", "support", "length"])

    sequential = commands.add_parser(
        "sequential-rules", help="sequential rules ('if X happens, Y happens afterwards') with support and confidence")
    _sequence_options(sequential)
    sequential.add_argument("--metric", default="confidence", help="measure used to filter and sort (default: %(default)s)")
    sequential.add_argument("--min-threshold", type=float, default=0.7, help="minimum value of --metric (default: 0.7)")
    sequential.add_argument("--max-antecedent", type=int, default=2, help="maximum items in X (default: 2)")
    sequential.add_argument("--max-consequent", type=int, default=2, help="maximum items in Y (default: 2)")
    _output_options(sequential)
    sequential.set_defaults(run=_run_sequential_rules)

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
