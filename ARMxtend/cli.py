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
    armxtend rules transacciones.csv --min-supp 0.05 --alpha 0.05      # solo reglas significativas
    armxtend temporal-rules sensores.csv --time-column hora --permutations 200 --alpha 0.05
    armxtend quantitative-rules clientes.csv --intervals 10 --max-supp 0.4 --min-supp 0.05 --interest 1.3
    armxtend generalized-rules cesta.txt --taxonomy jerarquia.csv --min-supp 0.2 --level-supp 3=0.05 --interest 1.1
    armxtend top-k-rules cesta.txt -k 20 --min-conf 0.7
    armxtend rare-rules cesta.txt --max-supp 0.1 --min-count 3 --min-conf 0.6 --alpha 0.05
    armxtend high-utility-rules ventas.csv --profits beneficios.csv --min-utility 0.05 --min-conf 0.5
    armxtend fuzzy-sequences sensores.csv --sequence-column dia --time-column hora --min-supp 0.3
    armxtend fuzzy-sequential-rules sensores.csv --sequence-column dia --time-column hora --min-threshold 0.8
    armxtend anomalies sensores.csv --time-column hora --train-fraction 0.5 --max-lag 1 --min-threshold 0.95
    armxtend forecast sensores.csv --time-column hora --train-fraction 0.7 --max-lag 2 --targets calefaccion
    armxtend next-events rutinas.txt --min-supp 0.2 --min-threshold 0.3 --history "alarma;bano"
    armxtend classify pacientes.csv --class-column diagnostico --test-fraction 0.3
    armxtend generate transactions -n 5000 --items 80 --seed 1 -o cesta.txt     # datos sinteticos para probar

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
    table = pd.read_csv(args.input, sep=args.delimiter)
    if args.time_column:
        if args.time_column not in table.columns:
            raise ValueError("la columna de tiempo '{}' no existe".format(args.time_column))
        table = table.set_index(pd.to_datetime(table.pop(args.time_column))).sort_index()
    return _items_from_table(table, args)


def _items_from_table(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Items booleanos de una tabla: ver `_read_items`."""
    from .preprocessing.timeseries import calendar_items, discretize, to_items

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
        text = json.dumps(json.loads(flat.to_json(orient="records", date_format="iso")), indent=2, ensure_ascii=False)
    elif args.format == "csv":
        text = flat.to_csv(index=False)
    else:
        if not args.all_measures:
            core = getattr(args, "table_columns", None) or (
                list(_ITEM_COLUMNS) + ["support", "confidence", "lift", "certainty_factor", args.metric,
                                       "p_adjusted", "p_permutation"])
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

    from .ARM import significant_rules

    transactions = _read_transactions(args.input, args.delimiter)
    frequent = fpgrowth(transactions, args.min_supp)
    itemsets = pd.DataFrame({"support": list(frequent.values()),
                             "itemsets": [key_to_itemset(key) for key in frequent]})
    rules = association_rules(itemsets, metric=args.metric, min_threshold=args.min_threshold)
    if args.alpha is not None and len(rules):
        rules = significant_rules(rules, args.alpha, args.correction, n_transactions=len(transactions),
                                  method=args.test)
    return prune_redundant_rules(rules) if args.prune and len(rules) else rules


def _run_generalized_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import generalized_rules, prune_redundant_rules, significant_rules
    from .FIM.taxonomy import Taxonomy

    transactions = _read_transactions(args.input, args.delimiter)
    taxonomy = Taxonomy(pd.read_csv(args.taxonomy, sep=args.delimiter))
    min_supp = args.min_supp
    if args.level_supp:
        min_supp = {1: args.min_supp}
        for entry in args.level_supp:
            level, _, value = entry.partition("=")
            if not value:
                raise ValueError("--level-supp espera NIVEL=SOPORTE, recibido '{}'".format(entry))
            min_supp[int(level)] = float(value)
    rules = generalized_rules(transactions, taxonomy, min_supp, metric=args.metric, min_threshold=args.min_threshold,
                              max_len=args.max_len, interest=args.interest)
    if args.alpha is not None and len(rules):
        rules = significant_rules(rules, args.alpha, args.correction, n_transactions=len(transactions),
                                  method=args.test)
    if args.prune and len(rules):
        rules = prune_redundant_rules(rules)
    args.table_columns = (list(_ITEM_COLUMNS) + ["support", "confidence", "lift", "certainty_factor", args.metric, "level"]
                          + (["interest"] if args.interest is not None else []) + ["p_adjusted"])
    return rules


def _run_quantitative_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import prune_redundant_rules, quantitative_rules, significant_rules

    table = pd.read_csv(args.input, sep=args.delimiter)
    rules = quantitative_rules(table, args.min_supp, n_intervals=args.intervals, max_supp=args.max_supp,
                               columns=args.columns, quantitative=args.quantitative, method=args.method,
                               metric=args.metric, min_threshold=args.min_threshold, max_len=args.max_len,
                               interest=args.interest)
    if args.alpha is not None and len(rules):
        rules = significant_rules(rules, args.alpha, args.correction, n_transactions=len(table), method=args.test)
    if args.prune and len(rules):
        rules = prune_redundant_rules(rules)
    args.table_columns = (list(_ITEM_COLUMNS) + ["support", "confidence", "lift", "certainty_factor", args.metric]
                          + (["interest"] if args.interest is not None else []) + ["p_adjusted"])
    return rules


def _special_rules(rules: pd.DataFrame, args: argparse.Namespace, n_transactions: int, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Significacion opcional y columnas de la tabla de las reglas top-k, raras y de utilidad (ya vienen ordenadas)."""
    from .ARM import significant_rules

    if args.alpha is not None and len(rules):
        rules = significant_rules(rules, args.alpha, args.correction, n_transactions=n_transactions, method=args.test)
    args.table_columns = list(_ITEM_COLUMNS) + ["support", "confidence", "lift"] + list(extra) + ["p_adjusted"]
    return rules


def _run_top_k_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import top_k_rules

    transactions = _read_transactions(args.input, args.delimiter)
    rules = top_k_rules(transactions, args.k, min_conf=args.min_conf, min_lift=args.min_lift,
                        max_antecedent=args.max_antecedent, max_consequent=args.max_consequent,
                        max_len=args.max_len or None, min_supp=args.min_supp)
    return _special_rules(rules, args, len(transactions))


def _run_rare_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import rare_rules

    transactions = _read_transactions(args.input, args.delimiter)
    rules = rare_rules(transactions, args.max_supp, min_supp=args.min_supp, min_count=args.min_count,
                       min_conf=args.min_conf, min_lift=args.min_lift, max_antecedent=args.max_antecedent,
                       max_consequent=args.max_consequent, max_len=args.max_len)
    return _special_rules(rules, args, len(transactions))


def _run_high_utility_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import high_utility_rules

    quantities = pd.read_csv(args.input, sep=args.delimiter)
    profits = None
    if args.profits:
        table = pd.read_csv(args.profits, sep=args.delimiter)
        if table.shape[1] < 2:
            raise ValueError("--profits espera un CSV con el item en la primera columna y su beneficio en la segunda")
        profits = dict(zip(table.iloc[:, 0].astype(str), table.iloc[:, 1]))
    rules = high_utility_rules(quantities, profits, args.min_utility, absolute=args.absolute, min_conf=args.min_conf,
                               min_lift=args.min_lift, max_antecedent=args.max_antecedent,
                               max_consequent=args.max_consequent, max_len=args.max_len)
    return _special_rules(rules, args, len(quantities), extra=("utility", "utility_share"))


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
    from .ARM import (
        permutation_null, permutation_p_values, prune_redundant_rules, significant_rules, temporal_association_rules,
    )

    args.pretty_lags = True
    items = _read_items(args)

    def miner(table):
        return temporal_association_rules(
            table, args.max_lag, args.min_supp, metric=args.metric, min_threshold=args.min_threshold,
            lags=args.lags, targets=args.targets, require_lagged_antecedent=not args.allow_no_lag,
            exclude_persistence=args.exclude_persistence)

    rules = miner(items)
    if args.permutations and len(rules):
        null = permutation_null(items, miner, n_permutations=args.permutations, null=args.null,
                                random_state=args.seed, test=args.test)
        rules = permutation_p_values(rules, null)
        if args.alpha is not None:
            rules = rules[rules["p_permutation"] <= args.alpha].sort_values("p_permutation", kind="mergesort")
    elif args.alpha is not None and len(rules):
        rules = significant_rules(rules, args.alpha, args.correction, method=args.test)
    return prune_redundant_rules(rules) if args.prune and len(rules) else rules


def _read_fuzzy_sequences(args: argparse.Namespace) -> list:
    """
    Secuencias difusas desde un CSV de series: las columnas numericas se fuzzifican
    (`--labels`, `--method`, `--membership`), cada fila es un elemento y `--sequence-column`
    (por ejemplo el dia) agrupa las filas en secuencias, ordenadas por `--time-column`.
    """
    from .FIM.fuzzy_sequential import fuzzy_sequences_from_memberships
    from .preprocessing import FuzzyLib

    table = pd.read_csv(args.input, sep=args.delimiter)
    for name in [args.time_column, args.sequence_column]:
        if name and name not in table.columns:
            raise ValueError("la columna '{}' no existe".format(name))
    if args.time_column:
        table[args.time_column] = pd.to_datetime(table[args.time_column])
        table = table.sort_values([c for c in [args.sequence_column, args.time_column] if c], kind="mergesort")
    library = FuzzyLib()
    library.data = table.reset_index(drop=True)
    names = library.AutoFuzzification(Atri=args.columns, n_labels=args.labels, method=args.method,
                                      membership=args.membership)
    groups = library.data[args.sequence_column] if args.sequence_column else None
    return fuzzy_sequences_from_memberships(library.data[names], by=groups, min_degree=args.min_degree)


def _run_fuzzy_sequences(args: argparse.Namespace) -> pd.DataFrame:
    from .FIM.fuzzy_sequential import fuzzy_prefixspan

    return fuzzy_prefixspan(_read_fuzzy_sequences(args), args.min_supp, max_len=args.max_len, min_len=args.min_len,
                            tnorm=args.tnorm)


def _run_fuzzy_sequential_rules(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import fuzzy_sequential_rules

    return fuzzy_sequential_rules(_read_fuzzy_sequences(args), args.min_supp,
                                  min_conf=args.min_threshold if args.metric == "confidence" else 0.0,
                                  max_antecedent=args.max_antecedent, max_consequent=args.max_consequent,
                                  metric=args.metric, min_threshold=args.min_threshold, tnorm=args.tnorm)


def _train_cut(rows: int, fraction: float) -> int:
    """Primera fila de la parte de prueba cuando las `fraction` primeras filas son de entrenamiento."""
    if not 0 < fraction < 1:
        raise ValueError("--train-fraction debe estar en (0, 1)")
    cut = int(rows * fraction)
    if cut < 2 or rows - cut < 2:
        raise ValueError("hay muy pocas filas ({}) para repartirlas con --train-fraction {}".format(rows, fraction))
    return cut


def _mine_normal_rules(items: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Reglas del periodo de entrenamiento para `anomalies` y `forecast` (solo con antecedente del pasado)."""
    from .ARM import crisp_rule_miner, prune_unproductive_rules, significant_rules, temporal_association_rules

    if args.max_lag > 0:
        rules = temporal_association_rules(items, args.max_lag, args.min_supp, metric=args.metric,
                                           min_threshold=args.min_threshold, targets=args.targets,
                                           same_time_antecedent=False, exclude_persistence=args.exclude_persistence)
    else:
        rules = crisp_rule_miner(args.min_supp, args.metric, args.min_threshold)(items)
    if args.alpha is not None and len(rules):
        rules = significant_rules(rules, args.alpha, args.correction, method=args.test)
    if args.single_consequent and len(rules):
        rules = rules[rules["consequents"].map(len) == 1]
    if args.min_improvement is not None and len(rules):
        rules = prune_unproductive_rules(rules, args.min_improvement)
    return rules


def _run_anomalies(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import detect_anomalies

    args.pretty_lags = True
    items = _read_items(args)
    cut = _train_cut(len(items), args.train_fraction)
    rules = _mine_normal_rules(items.iloc[:cut], args)
    print("{} rules learned from the first {} rows".format(len(rules), cut), file=sys.stderr)
    flagged = detect_anomalies(items, rules, min_score=args.min_score, weight=args.weight)
    flagged = flagged[items.index.get_indexer(flagged.index) >= cut] if items.index.is_unique else flagged
    flagged = flagged.rename_axis("time" if args.time_column else "row").reset_index()
    flagged["rules"] = [" | ".join(texts) for texts in flagged["rules"]]
    args.table_columns = [flagged.columns[0], "score", "violations", "rules"]
    return flagged


def _run_forecast(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import evaluate_forecast, forecast_next

    args.pretty_lags = True
    items = _read_items(args)
    if args.next:
        rules = _mine_normal_rules(items, args)
        print("{} rules learned from all {} rows".format(len(rules), len(items)), file=sys.stderr)
        args.table_columns = ["item", "score"]
        return forecast_next(items, rules).reset_index()
    cut = _train_cut(len(items), args.train_fraction)
    rules = _mine_normal_rules(items.iloc[:cut], args)
    print("{} rules learned from the first {} rows; evaluated on the remaining {}".format(
        len(rules), cut, len(items) - cut), file=sys.stderr)
    args.table_columns = ["item", "n", "positives", "predicted", "hits", "precision", "recall", "f1", "base_rate"]
    return evaluate_forecast(items.iloc[cut:], rules, threshold=args.threshold).reset_index()


def _run_next_events(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import evaluate_next_events, predict_next_events, sequential_rules

    sequences = _read_sequences(args)
    training = sequences
    if args.test_fraction:
        if not 0 < args.test_fraction < 1:
            raise ValueError("--test-fraction debe estar en (0, 1)")
        cut = len(sequences) - max(1, int(len(sequences) * args.test_fraction))
        training = sequences[:cut]
    elif not args.history:
        raise ValueError("indique --history (los eventos ya ocurridos) o --test-fraction (para evaluar)")
    rules = sequential_rules(training, args.min_supp, min_conf=args.min_threshold, max_antecedent=args.max_antecedent,
                             max_consequent=args.max_consequent)
    print("{} sequential rules learned from {} sequences".format(len(rules), len(training)), file=sys.stderr)
    if args.history:
        history = [[item.strip() for item in element.split(args.delimiter) if item.strip()]
                   for element in args.history.split(args.element_delimiter)]
        args.table_columns = ["item", "score"]
        return predict_next_events(history, rules).reset_index()
    result = evaluate_next_events(sequences[len(training):], rules, k=args.k, min_prefix=args.min_prefix,
                                  train=training, target=args.target)
    args.table_columns = ["measure", "value"]
    return pd.DataFrame({"measure": list(result), "value": [float(value) for value in result.values()]})


def _run_classify(args: argparse.Namespace) -> pd.DataFrame:
    import numpy as np

    from .ARM import RuleClassifier

    table = pd.read_csv(args.input, sep=args.delimiter)
    if args.class_column not in table.columns:
        raise ValueError("la columna de clase '{}' no existe".format(args.class_column))
    labels = table.pop(args.class_column)
    items = _items_from_table(table, args)
    if not 0 <= args.test_fraction < 1:
        raise ValueError("--test-fraction debe estar en [0, 1)")
    order = np.random.RandomState(args.seed).permutation(len(items))
    cut = len(items) - int(len(items) * args.test_fraction)
    train, test = order[:cut], order[cut:]
    model = RuleClassifier(args.min_supp, args.min_threshold, args.max_len, args.support, args.min_improvement)
    model.fit(items.iloc[train], labels.iloc[train])
    message = "{} class rules selected from {} candidates; default class: {}".format(
        len(model.rules_), len(model.candidate_rules_), model.default_class_)
    if len(test):
        majority = float((labels.iloc[test] == labels.iloc[train].mode()[0]).mean())
        message += "; accuracy on {} held-out rows: {:.3f} (majority class: {:.3f})".format(
            len(test), model.score(items.iloc[test], labels.iloc[test]), majority)
    print(message, file=sys.stderr)
    return model.rules_.drop(columns=["class"])


def _window_size(text: str):
    """'20' -> 20 filas; '7D', '12h', ... -> duracion (``window_rules`` exige un indice temporal)."""
    return int(text) if text.isdigit() else text


def _run_rule_drift(args: argparse.Namespace) -> pd.DataFrame:
    from .ARM import crisp_rule_miner, rule_drift, significant_miner, temporal_association_rules, window_rules

    if args.max_lag > 0:
        def miner(window):
            return temporal_association_rules(window, args.max_lag, args.min_supp, metric=args.metric,
                                              min_threshold=args.min_threshold, targets=args.targets,
                                              exclude_persistence=args.exclude_persistence)
    else:
        miner = crisp_rule_miner(args.min_supp, args.metric, args.min_threshold)
    if args.alpha is not None:
        miner = significant_miner(miner, args.alpha, args.correction, args.test)

    windows = window_rules(_read_items(args), _window_size(args.window),
                           _window_size(args.step) if args.step else None, miner=miner)
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


def _common_output_options(parser: argparse.ArgumentParser, metric: str, min_threshold: float = 0.7) -> None:
    parser.add_argument("-d", "--delimiter", default=",", help="field/item delimiter (default: ',')")
    parser.add_argument("--min-supp", type=float, default=0.1, help="minimum support (default: 0.1)")
    parser.add_argument("--metric", default=metric, help="measure used to filter and sort the rules (default: %(default)s)")
    parser.add_argument("--min-threshold", type=float, default=min_threshold,
                        help="minimum value of --metric (default: %(default)s)")
    _output_options(parser)


def _significance_options(parser: argparse.ArgumentParser, permutations: bool = False, what: str = "rules") -> None:
    """Opciones de significacion estadistica (ver ARM.significance)."""
    parser.add_argument("--alpha", type=float,
                        help="keep only the {} that are statistically significant at this level, after "
                             "correcting for multiple tests".format(what))
    parser.add_argument("--correction", choices=("bonferroni", "holm", "fdr_bh", "fdr_by"), default="fdr_bh",
                        help="multiple-testing correction (default: fdr_bh)")
    parser.add_argument("--test", choices=("fisher", "chi2"), default="fisher",
                        help="test of independence of antecedent and consequent (default: fisher)")
    if permutations:
        parser.add_argument("--permutations", type=int, metavar="N",
                            help="instead of Fisher + correction, test every rule against N shuffled copies of the "
                                 "data (min p-value over their rules); adds p_permutation, keeps p_permutation <= --alpha")
        parser.add_argument("--null", choices=("shift", "permute"), default="shift",
                            help="how the data are shuffled: circular shifts keep each series' autocorrelation "
                                 "(default) or independent permutations of the rows")
        parser.add_argument("--seed", type=int, default=0, help="random seed of the shuffles (default: 0)")


def _rule_filters(parser: argparse.ArgumentParser, min_conf: float) -> None:
    """Opciones de las reglas top-k, raras y de alta utilidad: filtros, significacion y salida."""
    parser.add_argument("-d", "--delimiter", default=",", help="field/item delimiter (default: ',')")
    parser.add_argument("--min-conf", type=float, default=min_conf, help="minimum confidence (default: %(default)s)")
    parser.add_argument("--min-lift", type=float, help="minimum lift")
    parser.add_argument("--max-antecedent", type=int, help="maximum items in the antecedent")
    parser.add_argument("--max-consequent", type=int, help="maximum items in the consequent")
    _significance_options(parser)
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


def _fuzzy_sequence_options(parser: argparse.ArgumentParser) -> None:
    """Opciones de entrada de fuzzy-sequences y fuzzy-sequential-rules."""
    parser.add_argument("input", help="CSV file with a header; numeric columns are fuzzified, one row per time step")
    parser.add_argument("-d", "--delimiter", default=",", help="CSV delimiter (default: ',')")
    parser.add_argument("--sequence-column",
                        help="column that groups the rows into sequences (day, patient...); default: one sequence")
    parser.add_argument("--time-column", help="column with the timestamps (rows are sorted by it within a sequence)")
    parser.add_argument("--columns", nargs="+", help="columns to fuzzify (default: all numeric columns)")
    parser.add_argument("--labels", type=int, default=3, help="fuzzy labels per column (default: 3)")
    parser.add_argument("--method", choices=("uniform", "quantile", "kmeans"), default="quantile",
                        help="how the partition is computed (default: quantile)")
    parser.add_argument("--membership", choices=("triangular", "trapezoidal", "gaussian"), default="triangular")
    parser.add_argument("--min-degree", type=float, default=0.0,
                        help="ignore memberships up to this degree (default: 0)")
    parser.add_argument("--min-supp", type=float, default=0.1, help="minimum fuzzy support (default: 0.1)")
    parser.add_argument("--tnorm", choices=("min", "product"), default="min",
                        help="t-norm that combines the degrees of the events of a pattern (default: min)")


def _normal_rules_options(parser: argparse.ArgumentParser) -> None:
    """Opciones de las reglas del funcionamiento normal (anomalies y forecast)."""
    parser.add_argument("--train-fraction", type=float, default=0.5,
                        help="share of the first rows used to learn the rules; the rest is analysed (default: 0.5)")
    parser.add_argument("--max-lag", type=int, default=1,
                        help="steps back the antecedents may look; 0 = rules within the same row (default: 1)")
    parser.add_argument("--targets", nargs="+", help="items the consequent is restricted to")
    parser.add_argument("--exclude-persistence", action="store_true",
                        help="drop rules that merely say a state persists (x at t-k => x at t)")
    parser.add_argument("--single-consequent", action="store_true", help="keep only rules with a single item as consequent")
    parser.add_argument("--min-improvement", type=float,
                        help="drop rules whose extra conditions improve the confidence of a more general rule by "
                             "no more than this")
    _significance_options(parser)


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
    _significance_options(rules)
    rules.add_argument("--html", metavar="FILE", help="also write the interactive VizARE viewer to FILE")
    rules.set_defaults(run=_run_rules)

    generalized = commands.add_parser(
        "generalized-rules", help="multilevel rules over a taxonomy of items (products and categories)")
    generalized.add_argument("input", help="file with one transaction per line, items separated by --delimiter ('-' = stdin)")
    generalized.add_argument("--taxonomy", required=True, metavar="FILE",
                             help="CSV with a header and two columns, child and parent (the same --delimiter)")
    _common_output_options(generalized, "confidence", 0.7)
    generalized.add_argument("--level-supp", nargs="+", metavar="LEVEL=SUPPORT",
                             help="minimum support of the deeper levels (levels start at 1; --min-supp is level 1 "
                                  "and the default of the unlisted ones), for example: 2=0.1 3=0.04")
    generalized.add_argument("--max-len", type=int, help="maximum number of items in a rule")
    generalized.add_argument("--interest", type=float, metavar="R",
                             help="keep only the R-interesting rules (R >= 1): those that beat what their ancestor rules "
                                  "predict by a factor R")
    generalized.add_argument("--prune", action="store_true", help="drop redundant rules")
    _significance_options(generalized)
    generalized.set_defaults(run=_run_generalized_rules)

    quantitative = commands.add_parser(
        "quantitative-rules", help="rules over numeric and categorical columns with merged intervals (Srikant-Agrawal)")
    quantitative.add_argument("input", help="CSV file with a header")
    _common_output_options(quantitative, "confidence", 0.8)
    quantitative.add_argument("--columns", nargs="+", help="columns to use (default: all)")
    quantitative.add_argument("--quantitative", nargs="+",
                              help="columns treated as numeric (default: the numeric ones with more distinct values "
                                   "than --intervals; the rest give column=value items)")
    quantitative.add_argument("--intervals", type=int, default=8, help="base intervals per numeric column (default: 8)")
    quantitative.add_argument("--method", choices=("quantile", "uniform"), default="quantile",
                              help="equal-frequency or equal-width base intervals (default: quantile)")
    quantitative.add_argument("--max-supp", type=float, default=0.5,
                              help="largest support of a range of several base intervals (default: 0.5)")
    quantitative.add_argument("--max-len", type=int, help="maximum number of items in a rule")
    quantitative.add_argument("--interest", type=float, metavar="R",
                              help="keep only the R-interesting rules (R >= 1): those that beat what the rules with wider "
                                   "ranges predict by a factor R")
    quantitative.add_argument("--prune", action="store_true", help="drop redundant rules")
    _significance_options(quantitative)
    quantitative.set_defaults(run=_run_quantitative_rules)

    topK = commands.add_parser(
        "top-k-rules", help="the k rules with the highest support, with no support threshold to guess")
    topK.add_argument("input", help="file with one transaction per line, items separated by --delimiter ('-' = stdin)")
    topK.add_argument("-k", type=int, required=True, help="number of rules to return")
    topK.add_argument("--min-supp", type=float, default=0.0, help="optional support floor (default: 0)")
    topK.add_argument("--max-len", type=int, default=6,
                      help="maximum items in a rule (default: 6; 0 = no limit, can be very slow on dense data)")
    _rule_filters(topK, 0.6)
    topK.set_defaults(run=_run_top_k_rules, metric=None, keep_order=True)

    rare = commands.add_parser(
        "rare-rules", help="rules among items that are frequent separately and rare together (minimal rare itemsets)")
    rare.add_argument("input", help="file with one transaction per line, items separated by --delimiter ('-' = stdin)")
    rare.add_argument("--max-supp", type=float, required=True,
                      help="an itemset is frequent from this support up and rare below it")
    rare.add_argument("--min-supp", type=float, default=0.0, help="minimum support of a rare itemset (default: 0)")
    rare.add_argument("--min-count", type=int, default=2,
                      help="minimum number of transactions of a rare itemset (default: 2; 1 keeps what happened once)")
    rare.add_argument("--max-len", type=int, help="maximum items in a rule")
    _rule_filters(rare, 0.6)
    rare.set_defaults(run=_run_rare_rules, metric=None, keep_order=True)

    utility = commands.add_parser(
        "high-utility-rules", help="rules from the itemsets that bring the most profit (not the most frequent)")
    utility.add_argument("input", help="CSV file with a header: one row per transaction, one column per item (quantity bought)")
    utility.add_argument("--profits", metavar="FILE",
                         help="CSV with a header: item in the first column, unit profit (>= 0) in the second "
                              "(default: profit 1 for every item)")
    utility.add_argument("--min-utility", type=float, default=0.1,
                         help="minimum utility of an itemset: a fraction of the total utility (default: 0.1) or, with "
                              "--absolute, a value")
    utility.add_argument("--absolute", action="store_true", help="read --min-utility as an absolute value")
    utility.add_argument("--max-len", type=int, help="maximum items in a rule")
    _rule_filters(utility, 0.6)
    utility.set_defaults(run=_run_high_utility_rules, metric=None, keep_order=True)

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
    _significance_options(temporal, permutations=True)
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
    _significance_options(drift, what="rules of each window")
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

    fuzzySequences = commands.add_parser(
        "fuzzy-sequences", help="frequent fuzzy sequential patterns from a CSV of numeric time series")
    _fuzzy_sequence_options(fuzzySequences)
    fuzzySequences.add_argument("--max-len", type=int, help="maximum number of items in a pattern")
    fuzzySequences.add_argument("--min-len", type=int, default=1, help="minimum number of items in a pattern (default: 1)")
    _output_options(fuzzySequences, "patterns")
    fuzzySequences.set_defaults(run=_run_fuzzy_sequences, metric=None, keep_order=True,
                                table_columns=["sequence", "support", "length"])

    fuzzySequential = commands.add_parser(
        "fuzzy-sequential-rules", help="fuzzy sequential rules ('if X happens, Y happens afterwards') from numeric time series")
    _fuzzy_sequence_options(fuzzySequential)
    fuzzySequential.add_argument("--metric", default="confidence", help="measure used to filter and sort (default: %(default)s)")
    fuzzySequential.add_argument("--min-threshold", type=float, default=0.7,
                                 help="minimum value of --metric (default: 0.7)")
    fuzzySequential.add_argument("--max-antecedent", type=int, default=2, help="maximum items in X (default: 2)")
    fuzzySequential.add_argument("--max-consequent", type=int, default=2, help="maximum items in Y (default: 2)")
    _output_options(fuzzySequential)
    fuzzySequential.set_defaults(run=_run_fuzzy_sequential_rules)

    anomalies = commands.add_parser(
        "anomalies", help="rows that break the rules learned from the normal behaviour (rule violation)")
    _series_options(anomalies)
    _common_output_options(anomalies, "confidence", 0.9)
    _normal_rules_options(anomalies)
    anomalies.add_argument("--min-score", type=float, default=0.0,
                           help="report only rows with a score above this (default: 0, any violation)")
    anomalies.add_argument("--weight", choices=("surprise", "confidence", "count"), default="surprise",
                           help="weight of a violated rule: -log2(1 - confidence) bits (default), its confidence or 1")
    anomalies.set_defaults(run=_run_anomalies, keep_order=True)

    forecast = commands.add_parser(
        "forecast", help="predict items at the next time step with temporal rules; evaluated on held-out rows")
    _series_options(forecast)
    _common_output_options(forecast, "confidence", 0.8)
    _normal_rules_options(forecast)
    forecast.add_argument("--threshold", type=float, default=0.5,
                          help="predict an item when the confidence of the rules that fire is at least this (default: 0.5)")
    forecast.add_argument("--next", action="store_true",
                          help="instead of evaluating, print the forecast for the step after the last row")
    forecast.set_defaults(run=_run_forecast, keep_order=True)

    nextEvents = commands.add_parser(
        "next-events", help="events expected to happen next in a sequence (sequential rules); evaluated on held-out sequences")
    _sequence_options(nextEvents)
    nextEvents.add_argument("--min-threshold", type=float, default=0.3, help="minimum confidence of the rules (default: 0.3)")
    nextEvents.add_argument("--max-antecedent", type=int, default=2, help="maximum items in X (default: 2)")
    nextEvents.add_argument("--max-consequent", type=int, default=1, help="maximum items in Y (default: 1)")
    nextEvents.add_argument("--history", help="events already observed, elements separated by --element-delimiter: "
                                              "prints the ranking of the events still to come")
    nextEvents.add_argument("--test-fraction", type=float,
                            help="evaluate instead: learn from the first sequences and test on this fraction of the last ones")
    nextEvents.add_argument("-k", type=int, default=1, help="evaluation: events predicted per cut (default: 1)")
    nextEvents.add_argument("--min-prefix", type=int, default=1, help="evaluation: elements of history at least (default: 1)")
    nextEvents.add_argument("--target", choices=("future", "next"), default="future",
                            help="evaluation: a prediction is right if the event happens later (default) or next")
    _output_options(nextEvents, "events")
    nextEvents.set_defaults(run=_run_next_events, metric=None, keep_order=True)

    classify = commands.add_parser(
        "classify", help="class association rules (CBA classifier) from a CSV with a class column")
    classify.add_argument("input", help="CSV file with a header; numeric columns are discretised")
    classify.add_argument("--class-column", required=True, help="column with the class to predict")
    classify.add_argument("--columns", nargs="+", help="attribute columns to use (default: all but the class)")
    classify.add_argument("--symbols", type=int, default=3, help="symbols per numeric column (default: 3)")
    classify.add_argument("--method", choices=("quantile", "uniform", "sax"), default="quantile",
                          help="how numeric columns are discretised (default: quantile)")
    classify.add_argument("-d", "--delimiter", default=",", help="CSV delimiter (default: ',')")
    classify.add_argument("--min-supp", type=float, default=0.05,
                          help="minimum support of a rule, within its class (see --support; default: 0.05)")
    classify.add_argument("--min-threshold", type=float, default=0.6, help="minimum confidence (default: 0.6)")
    classify.add_argument("--max-len", type=int, default=3, help="maximum conditions in a rule (default: 3)")
    classify.add_argument("--support", choices=("class", "global"), default="class",
                          help="whether --min-supp is relative to the rows of the class (default) or to all rows")
    classify.add_argument("--min-improvement", type=float,
                          help="drop rules whose extra conditions improve the confidence of a more general rule by "
                               "no more than this")
    classify.add_argument("--test-fraction", type=float, default=0.3,
                          help="share of the rows held out to measure the accuracy (default: 0.3; 0 = use all to learn)")
    classify.add_argument("--seed", type=int, default=0, help="random seed of the split (default: 0)")
    _output_options(classify)
    classify.set_defaults(run=_run_classify, metric=None, keep_order=True, time_column=None, calendar=None,
                          table_columns=["antecedents", "consequents", "support", "confidence", "lift"])

    generate = commands.add_parser(
        "generate", help="synthetic data with planted patterns, in the input formats of the other commands")
    generate.add_argument("kind", choices=("transactions", "sequences", "quantities"),
                          help="transactions (for rules, top-k-rules...), sequences (for sequences...) or quantities "
                               "(CSV for high-utility-rules)")
    generate.add_argument("-n", type=int, default=1000, help="number of transactions or sequences (default: 1000)")
    generate.add_argument("--items", type=int, default=100, help="distinct items (default: 100)")
    generate.add_argument("--length", type=float, default=8.0,
                          help="average items per transaction, or elements per sequence (default: 8)")
    generate.add_argument("--seed", type=int, help="random seed, to repeat the same data")
    generate.add_argument("-d", "--delimiter", default=",", help="item delimiter (default: ',')")
    generate.add_argument("--element-delimiter", default=";", help="element delimiter in sequences (default: ';')")
    generate.add_argument("--profits", metavar="FILE", help="quantities: also write the unit profits of the items to FILE")
    generate.add_argument("-o", "--output", metavar="FILE", help="write to FILE instead of the standard output")
    generate.set_defaults(run=_run_generate)

    return parser


def _run_generate(args: argparse.Namespace) -> None:
    """Escribe datos sinteticos (con los mismos formatos de entrada que `rules` y `sequences`) en un fichero o en stdout."""
    from .datasets import make_quantities, make_sequences, make_transactions

    if args.kind == "transactions":
        data = make_transactions(args.n, args.items, args.length, random_state=args.seed)
        lines = [args.delimiter.join(basket) for basket in data]
    elif args.kind == "sequences":
        data = make_sequences(args.n, args.items, args.length, random_state=args.seed)
        lines = [args.element_delimiter.join(args.delimiter.join(element) for element in sequence) for sequence in data]
    else:
        quantities, profits = make_quantities(args.n, args.items, args.length, random_state=args.seed)
        lines = quantities.to_csv(index=False).splitlines()
        if args.profits:
            pd.Series(profits, name="profit").rename_axis("item").to_csv(args.profits)
    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text)
    else:
        sys.stdout.write(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Punto de entrada de la CLI; devuelve el codigo de salida (0 = correcto)."""
    args = build_parser().parse_args(argv)
    try:
        rules = args.run(args)
        if rules is not None:
            _emit(rules, args)
    except (ValueError, OSError, KeyError) as error:
        print("armxtend: error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
