# -*- coding: utf-8 -*-
"""
Prediccion con reglas: usar reglas ya minadas para anticipar lo que ocurrira.

* Series de items (sensores, logs): las reglas temporales (``ARM.temporal_rules``)
  ``A@lag1, B@lag2 => C`` predicen que ``C`` ocurrira en el instante actual a
  partir de lo que paso antes. ``rule_scores`` da, para cada instante, la
  confianza con que las reglas que se cumplen predicen cada item;
  ``forecast_next`` lo hace para el instante que viene tras el ultimo dato y
  ``evaluate_forecast`` mide la calidad de la prediccion sobre datos reales.
* Secuencias de eventos: las reglas secuenciales (``ARM.sequential_rules``)
  ``X => Y`` ("si ocurre X, despues Y") predicen los eventos que faltan por
  ocurrir. ``predict_next_events`` ordena los eventos pendientes dado el
  historial de una secuencia y ``evaluate_next_events`` mide los aciertos.

Cuando varias reglas predicen el mismo item se combinan con ``combine``:
``max`` (la confianza de la mejor regla, por defecto) o ``noisy_or``
(``1 - prod(1 - confianza)``, que supone reglas independientes y por tanto
sobreestima si hay reglas redundantes: pode antes con ``prune_redundant_rules``).

@author Carlos Fernandez-Basso
"""
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..FIM.sequential import _normalize
from ..preprocessing.timeseries import split_lag
from ._apply import ItemTable, max_lag, require_columns

COMBINE = ("max", "noisy_or")


def _check_combine(combine: str) -> None:
    if combine not in COMBINE:
        raise ValueError("combine debe ser una de {}, recibida: '{}'".format(COMBINE, combine))


def _usable(rules: pd.DataFrame, past_only: bool) -> pd.DataFrame:
    """Reglas aplicables: consecuente en el instante actual y, con `past_only`, antecedente solo del pasado."""
    require_columns(rules, "antecedents", "consequents", "confidence")
    keep = []
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        if any(split_lag(item)[1] != 0 for item in consequent):
            raise ValueError("el consecuente de una regla debe estar en el instante actual (sin '@lag'), "
                             "pero hay {}".format(sorted(consequent)))
        keep.append(not (past_only and any(split_lag(item)[1] == 0 for item in antecedent)))
    return rules[np.asarray(keep, dtype=bool)]


def rule_scores(items: pd.DataFrame, rules: pd.DataFrame, combine: str = "max", past_only: bool = True) -> pd.DataFrame:
    """
    Para cada instante, la confianza con la que las reglas que se cumplen
    predicen cada item.

    Argumentos:
        items (pandas.DataFrame): items booleanos (0/1), una fila por instante y
            ordenadas en el tiempo. Los items de las reglas deben ser columnas
        rules (pandas.DataFrame): reglas temporales con 'antecedents',
            'consequents' (en el instante actual) y 'confidence', p.ej. la
            salida de ``temporal_association_rules``
        combine (str): 'max' o 'noisy_or' (ver el docstring del modulo)
        past_only (bool): si True (por defecto), se ignoran las reglas con
            algun item del antecedente en el instante actual (``same_time_antecedent=True``
            al minarlas), porque al predecir ese dato aun no se conoce

    Retorna:
        pandas.DataFrame con el indice de `items` y una columna por item que
        es consecuente de alguna regla, con valores en [0, 1] (0: ninguna regla
        lo predice). Las primeras filas, sin historia para los retardos, no
        disparan las reglas que la necesitan.
    """
    _check_combine(combine)
    usable = _usable(rules, past_only)
    table = ItemTable(items)
    names = sorted({item for consequent in usable["consequents"] for item in consequent})
    scores = {name: np.zeros(table.rows) for name in names}
    for antecedent, consequent, confidence in zip(usable["antecedents"], usable["consequents"], usable["confidence"]):
        fired = table.holds(antecedent) * float(confidence)
        for name in consequent:
            scores[name] = np.maximum(scores[name], fired) if combine == "max" else 1.0 - (1.0 - scores[name]) * (1.0 - fired)
    return pd.DataFrame(scores, index=items.index, columns=names)


def forecast_next(items: pd.DataFrame, rules: pd.DataFrame, combine: str = "max") -> pd.Series:
    """
    Prediccion del instante que viene tras la ultima fila de `items`: la
    confianza con que las reglas, aplicadas a lo ya observado, predicen cada item.

    Solo se usan las reglas cuyo antecedente es del pasado (retardos >= 1); las
    que piden mas historia de la que hay no se disparan. Retorna una ``Series`` 'score' indexada por item, de mayor a menor (solo
    los items con alguna regla que se cumple).
    """
    _check_combine(combine)
    usable = _usable(rules, past_only=True)
    future = pd.DataFrame(False, index=[0], columns=items.columns)
    extended = pd.concat([items.reset_index(drop=True), future], ignore_index=True)
    last = rule_scores(extended, usable, combine=combine, past_only=True).iloc[-1]
    last = last[last > 0].sort_values(ascending=False, kind="mergesort")
    return last.rename("score").rename_axis("item")


def evaluate_forecast(items: pd.DataFrame, rules: pd.DataFrame, threshold: float = 0.5, combine: str = "max",
                      past_only: bool = True) -> pd.DataFrame:
    """
    Calidad de la prediccion de las reglas sobre datos reales: en cada instante
    se predice que ocurre el item si su puntuacion (`rule_scores`) es >=
    `threshold`, y se compara con lo ocurrido. Para una evaluacion honesta, `rules`
    debe haberse minado con otros datos que `items` (p.ej. con el periodo anterior).

    Se excluyen las primeras filas, sin historia para los retardos de las reglas.

    Retorna:
        pandas.DataFrame indexado por item con las columnas 'n' (instantes
        evaluados), 'positives' (veces que ocurrio), 'predicted' (veces que se
        predijo), 'hits' (predichos y ocurridos), 'precision' y 'recall' (NaN
        si no hay predicciones, o no ocurrio nunca), 'f1' y 'base_rate'
        (positives / n: la precision de predecir siempre que ocurre)
    """
    usable = _usable(rules, past_only)
    scores = rule_scores(items, usable, combine=combine, past_only=past_only)
    start = max_lag(usable["antecedents"])
    table = ItemTable(items)
    rows = []
    for name in scores.columns:
        actual = table.item(name)[start:]
        predicted = scores[name].to_numpy()[start:] >= threshold
        hits = int(np.count_nonzero(actual & predicted))
        positives, guesses = int(np.count_nonzero(actual)), int(np.count_nonzero(predicted))
        precision = hits / guesses if guesses else np.nan
        recall = hits / positives if positives else np.nan
        if np.isnan(precision) or np.isnan(recall):
            f1 = np.nan
        else:
            f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
        rows.append({"item": name, "n": len(actual), "positives": positives, "predicted": guesses, "hits": hits,
                     "precision": precision, "recall": recall, "f1": f1,
                     "base_rate": positives / len(actual) if len(actual) else np.nan})
    return pd.DataFrame(rows, columns=["item", "n", "positives", "predicted", "hits", "precision", "recall", "f1",
                                       "base_rate"]).set_index("item")


# --- Secuencias de eventos ------------------------------------------------------------------------

Compiled = List[Tuple[FrozenSet[str], FrozenSet[str], float]]
SequenceLike = Iterable[Union[str, Iterable[str]]]


def _compile(rules: pd.DataFrame) -> Compiled:
    require_columns(rules, "antecedents", "consequents", "confidence")
    return [(frozenset(a), frozenset(c), float(p))
            for a, c, p in zip(rules["antecedents"], rules["consequents"], rules["confidence"])]


def _pending(history: Sequence[FrozenSet[str]], compiled: Compiled, combine: str) -> Dict[str, float]:
    """Puntuacion de cada evento pendiente dado el historial (lista de elementos)."""
    firstSeen: Dict[str, int] = {}
    lastSeen: Dict[str, int] = {}
    for position, element in enumerate(history):
        for item in element:
            firstSeen.setdefault(item, position)
            lastSeen[item] = position
    scores: Dict[str, float] = {}
    for antecedent, consequent, confidence in compiled:
        if not all(item in firstSeen for item in antecedent):
            continue
        completed = max(firstSeen[item] for item in antecedent)
        for item in consequent:
            if lastSeen.get(item, -1) > completed:      # ya ocurrio despues de X: la regla ya se cumplio para este item
                continue
            previous = scores.get(item, 0.0)
            scores[item] = max(previous, confidence) if combine == "max" else 1.0 - (1.0 - previous) * (1.0 - confidence)
    return scores


def _ranked(scores: Dict[str, float]) -> pd.Series:
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return pd.Series([value for _, value in ordered], index=pd.Index([name for name, _ in ordered], name="item"),
                     name="score", dtype=float)


def predict_next_events(history: SequenceLike, rules: pd.DataFrame, combine: str = "max") -> pd.Series:
    """
    Eventos que cabe esperar dado el historial de una secuencia, segun las
    reglas secuenciales ``X => Y``.

    Una regla se dispara si todos los items de X han ocurrido ya en el historial;
    predice cada item de Y que todavia esta PENDIENTE, es decir, que no ha
    vuelto a ocurrir despues de completarse X (si ya ocurrio, la regla ya se ha
    cumplido para ese item).

    Argumentos:
        history: la secuencia observada hasta ahora: una lista de elementos
            (conjuntos de items simultaneos, o un item suelto)
        rules (pandas.DataFrame): reglas secuenciales con 'antecedents',
            'consequents' y 'confidence' (salida de ``sequential_rules``)
        combine (str): 'max' o 'noisy_or'

    Retorna:
        pandas.Series 'score' indexada por evento, de mayor a menor puntuacion
        (a igual puntuacion, por orden alfabetico)
    """
    _check_combine(combine)
    return _ranked(_pending(_normalize([history])[0], _compile(rules), combine))


def evaluate_next_events(sequences: Iterable[SequenceLike], rules: pd.DataFrame, k: int = 1, min_prefix: int = 1,
                         train: Optional[Iterable[SequenceLike]] = None, combine: str = "max",
                         target: str = "future") -> Dict[str, float]:
    """
    Aciertos de `predict_next_events` sobre secuencias reales: cada secuencia se
    corta tras cada elemento (a partir de `min_prefix` elementos de historial
    y hasta el penultimo) y se comprueba si los `k` eventos mejor puntuados
    ocurren de verdad en lo que queda de la secuencia. Para una evaluacion
    honesta, `rules` debe haberse minado con otras secuencias que `sequences`.

    Argumentos:
        sequences: secuencias a evaluar
        rules: reglas secuenciales (ver `predict_next_events`)
        k (int): numero de eventos predichos por corte
        min_prefix (int): elementos minimos del historial, >= 1
        train: secuencias con las que calcular una prediccion de referencia:
            los `k` eventos mas frecuentes (por numero de secuencias que los
            contienen) que aun no han ocurrido en el historial
        combine (str): 'max' o 'noisy_or'
        target (str): 'future' (por defecto): un evento predicho acierta si
            ocurre en cualquier elemento posterior al corte, que es lo que
            afirman las reglas secuenciales; 'next': solo si ocurre en el
            elemento inmediatamente siguiente (mas exigente)

    Retorna:
        dict con 'cases' (cortes evaluados), 'coverage' (parte de los cortes con
        alguna prediccion), 'hit_rate' (parte de los cortes en que algun
        evento predicho ocurre despues; un corte sin predicciones cuenta como
        fallo) y 'precision' (media, sobre los cortes con prediccion, de la
        parte de los eventos predichos que ocurren despues); con `train`, tambien
        'baseline_hit_rate' y 'baseline_precision'
    """
    _check_combine(combine)
    if k < 1 or min_prefix < 1:
        raise ValueError("k y min_prefix deben ser >= 1")
    if target not in ("future", "next"):
        raise ValueError("target debe ser 'future' o 'next', recibido: '{}'".format(target))
    compiled = _compile(rules)
    popularity: List[str] = []
    if train is not None:
        counts: Dict[str, int] = {}
        for sequence in _normalize(train):
            for item in set().union(*sequence) if sequence else ():
                counts[item] = counts.get(item, 0) + 1
        popularity = [name for name, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]

    cases = covered = hits = baselineHits = 0
    precisions: List[float] = []
    baselinePrecisions: List[float] = []
    for sequence in _normalize(sequences):
        for cut in range(min_prefix, len(sequence)):
            history, future = sequence[:cut], set().union(*(sequence[cut:cut + 1] if target == "next" else sequence[cut:]))
            cases += 1
            top = list(_ranked(_pending(history, compiled, combine)).index[:k])
            if top:
                covered += 1
                correct = sum(item in future for item in top)
                hits += correct > 0
                precisions.append(correct / len(top))
            if train is not None:
                seen = set().union(*history)
                guess = [item for item in popularity if item not in seen][:k]
                correct = sum(item in future for item in guess)
                baselineHits += correct > 0
                baselinePrecisions.append(correct / len(guess) if guess else 0.0)
    result = {"cases": cases, "coverage": covered / cases if cases else float("nan"),
              "hit_rate": hits / cases if cases else float("nan"),
              "precision": float(np.mean(precisions)) if precisions else float("nan")}
    if train is not None:
        result["baseline_hit_rate"] = baselineHits / cases if cases else float("nan")
        result["baseline_precision"] = float(np.mean(baselinePrecisions)) if baselinePrecisions else float("nan")
    return result
