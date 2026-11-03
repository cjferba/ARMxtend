# -*- coding: utf-8 -*-
"""
Prediction with rules: use already mined rules to anticipate what will happen.

* Series of items (sensors, logs): the temporal rules (``ARM.temporal_rules``) ``A@lag1, B@lag2 => C`` predict that
  ``C`` will occur at the current instant from what happened before. ``rule_scores`` gives, for each instant, the
  confidence with which the rules that hold predict each item; ``forecast_next`` does it for the instant that follows
  the last datum and ``evaluate_forecast`` measures the quality of the prediction on real data.
* Sequences of events: the sequential rules (``ARM.sequential_rules``) ``X => Y`` ("if X occurs, then Y") predict the
  events that are still to occur. ``predict_next_events`` ranks the pending events given the history of a sequence and
  ``evaluate_next_events`` measures the hits.

When several rules predict the same item they are combined with ``combine``: ``max`` (the confidence of the best
rule, default) or ``noisy_or`` (``1 - prod(1 - confidence)``, which assumes independent rules and therefore
overestimates if there are redundant rules: prune first with ``prune_redundant_rules``).

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
        raise ValueError("combine must be one of {}, received: '{}'".format(COMBINE, combine))


def _usable(rules: pd.DataFrame, past_only: bool) -> pd.DataFrame:
    """Applicable rules: consequent at the current instant and, with `past_only`, antecedent only from the past."""
    require_columns(rules, "antecedents", "consequents", "confidence")
    keep = []
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        if any(split_lag(item)[1] != 0 for item in consequent):
            raise ValueError("the consequent of a rule must be at the current instant (without '@lag'), "
                             "but there is {}".format(sorted(consequent)))
        keep.append(not (past_only and any(split_lag(item)[1] == 0 for item in antecedent)))
    return rules[np.asarray(keep, dtype=bool)]


def rule_scores(items: pd.DataFrame, rules: pd.DataFrame, combine: str = "max", past_only: bool = True) -> pd.DataFrame:
    """
    For each instant, the confidence with which the rules that hold predict each item.

    Args:
        items (pandas.DataFrame): boolean (0/1) items, one row per instant and ordered in time. The items of the rules
            must be columns
        rules (pandas.DataFrame): temporal rules with 'antecedents', 'consequents' (at the current instant) and
            'confidence', e.g. the output of ``temporal_association_rules``
        combine (str): 'max' or 'noisy_or' (see the module docstring)
        past_only (bool): if True (default), the rules with any antecedent item at the current instant
            (``same_time_antecedent=True`` when mining them) are ignored, because when predicting that datum is not
            yet known

    Returns:
        pandas.DataFrame with the index of `items` and a column per item that is the consequent of some rule, with
        values in [0, 1] (0: no rule predicts it). The first rows, without history for the lags, do not fire the rules
        that need it.
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
    Prediction of the instant that follows the last row of `items`: the confidence with which the rules, applied to
    what has already been observed, predict each item.

    Only the rules whose antecedent is from the past (lags >= 1) are used; those that ask for more history than there
    is do not fire. Returns a 'score' ``Series`` indexed by item, from highest to lowest (only the items with some
    rule that holds).
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
    Quality of the prediction of the rules on real data: at each instant the item is predicted to occur if its score
    (`rule_scores`) is >= `threshold`, and this is compared with what happened. For an honest evaluation, `rules` must
    have been mined with data other than `items` (e.g. with the previous period).

    The first rows, without history for the lags of the rules, are excluded.

    Returns:
        pandas.DataFrame indexed by item with the columns 'n' (instants evaluated), 'positives' (times it occurred),
        'predicted' (times it was predicted), 'hits' (predicted and occurred), 'precision' and 'recall' (NaN if there
        are no predictions, or it never occurred), 'f1' and 'base_rate' (positives / n: the precision of predicting it
        always)
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
    """Score of each pending event given the history (list of elements)."""
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
            if lastSeen.get(item, -1) > completed:      # it already occurred after X: the rule already held for this item
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
    Events to expect given the history of a sequence, according to the sequential rules ``X => Y``.

    A rule fires if all the items of X have already occurred in the history; it predicts each item of Y that is still
    PENDING, that is, that has not occurred again after X was completed (if it already occurred, the rule has already
    held for that item).

    Args:
        history: the sequence observed so far: a list of elements (sets of simultaneous items, or a single item)
        rules (pandas.DataFrame): sequential rules with 'antecedents', 'consequents' and 'confidence' (output of
            ``sequential_rules``)
        combine (str): 'max' or 'noisy_or'

    Returns:
        pandas.Series 'score' indexed by event, from highest to lowest score (with equal score, in alphabetical order)
    """
    _check_combine(combine)
    return _ranked(_pending(_normalize([history])[0], _compile(rules), combine))


def evaluate_next_events(sequences: Iterable[SequenceLike], rules: pd.DataFrame, k: int = 1, min_prefix: int = 1,
                         train: Optional[Iterable[SequenceLike]] = None, combine: str = "max",
                         target: str = "future") -> Dict[str, float]:
    """
    Hits of `predict_next_events` on real sequences: each sequence is cut after each element (from `min_prefix`
    elements of history up to the second to last) and it is checked whether the `k` best-scored events really occur in
    what remains of the sequence. For an honest evaluation, `rules` must have been mined with sequences other than
    `sequences`.

    Args:
        sequences: sequences to evaluate
        rules: sequential rules (see `predict_next_events`)
        k (int): number of events predicted per cut
        min_prefix (int): minimum elements of history, >= 1
        train: sequences from which to compute a reference prediction: the `k` most frequent events (by number of
            sequences that contain them) that have not yet occurred in the history
        combine (str): 'max' or 'noisy_or'
        target (str): 'future' (default): a predicted event is a hit if it occurs in any element after the cut, which
            is what the sequential rules claim; 'next': only if it occurs in the immediately following element (more
            demanding)

    Returns:
        dict with 'cases' (cuts evaluated), 'coverage' (share of the cuts with some prediction), 'hit_rate' (share of
        the cuts in which some predicted event occurs later; a cut without predictions counts as a miss) and
        'precision' (mean, over the cuts with prediction, of the share of predicted events that occur later); with
        `train`, also 'baseline_hit_rate' and 'baseline_precision'
    """
    _check_combine(combine)
    if k < 1 or min_prefix < 1:
        raise ValueError("k and min_prefix must be >= 1")
    if target not in ("future", "next"):
        raise ValueError("target must be 'future' or 'next', got: '{}'".format(target))
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
