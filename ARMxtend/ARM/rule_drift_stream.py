# -*- coding: utf-8 -*-
"""
INCREMENTAL rule drift: ``RuleDriftMonitor`` receives a stream of rows in chunks (``update``) and mines only the
windows that the new rows complete, instead of re-mining everything as ``window_rules`` does on the whole table. The
rows that no later window needs are dropped, so memory stays bounded however long the stream, and with ``history`` only
the rules of the last windows are kept.

The windows are the same as those of ``window_rules`` (by number of rows or by duration; same `width`, `step`,
`min_rows`), but only COMPLETE windows are mined: a window by rows when all its rows have arrived; a window by duration
when a row at or after its end has arrived (so that no later row can still belong to it). Therefore, after feeding all
the data, the windows mined are those of ``window_rules`` over the same data that are complete, and the rules are the
same (the test suite checks it).

@author Carlos Fernandez-Basso
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .rule_drift import Miner, Width, compare_rule_sets, crisp_rule_miner, rule_drift


class RuleDriftMonitor:
    """
    Follow the rules of a stream window by window.

    Args:
        width, step: size and displacement of the windows, as in ``window_rules`` (rows, or durations if the index is a
            ``DatetimeIndex``); `step` defaults to `width`
        min_supp, metric, min_threshold, miner: as in ``window_rules`` (default miner: ``crisp_rule_miner``)
        min_rows (int): complete windows with fewer rows are omitted and not numbered
        history (int, optional): keep only the rules of the last `history` windows (the window numbers keep growing);
            by default all

    Attributes: ``n_windows`` (windows mined so far), ``rules_`` (the rules of the windows kept, in the format of
    ``window_rules``) and ``buffered_rows`` (rows waiting for their window to complete).
    """

    def __init__(self, width: Width, step: Optional[Width] = None, min_supp: float = 0.1, metric: str = "confidence",
                 min_threshold: float = 0.8, miner: Optional[Miner] = None, min_rows: int = 2,
                 history: Optional[int] = None) -> None:
        by_rows = isinstance(width, (int, np.integer))
        if by_rows:
            self._width: Union[int, pd.Timedelta] = int(width)       # type: ignore[arg-type]
            self._step: Union[int, pd.Timedelta] = self._width if step is None else int(step)   # type: ignore[arg-type]
            if self._width < 1 or self._step < 1:
                raise ValueError("width and step must be >= 1")
        else:
            self._width = pd.Timedelta(width)
            self._step = self._width if step is None else pd.Timedelta(step)
            if self._width <= pd.Timedelta(0) or self._step <= pd.Timedelta(0):
                raise ValueError("width and step must be positive durations")
        if history is not None and history < 1:
            raise ValueError("history must be >= 1")
        self._byRows = by_rows
        self._miner = miner if miner is not None else crisp_rule_miner(min_supp, metric, min_threshold)
        self._minRows = min_rows
        self._history = history
        self._buffer: Optional[pd.DataFrame] = None
        self._cursor: Union[int, pd.Timestamp, None] = 0 if by_rows else None      # start of the next window
        self._dropped = 0                                        # rows removed from the front of the buffer (by rows)
        self.n_windows = 0
        self._frames: List[pd.DataFrame] = []

    @property
    def buffered_rows(self) -> int:
        return 0 if self._buffer is None else len(self._buffer)

    @property
    def rules_(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=["window", "start", "end", "n_rows", "antecedents", "consequents"])
        result = pd.concat(self._frames, ignore_index=True)
        result.attrs["n_windows"] = self.n_windows
        return result

    def _mine(self, window: pd.DataFrame) -> pd.DataFrame:
        rules = self._miner(window).copy()
        rules.insert(0, "window", self.n_windows)
        rules.insert(1, "start", window.index[0])
        rules.insert(2, "end", window.index[-1])
        rules.insert(3, "n_rows", len(window))
        return rules

    def update(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Add the new rows (in time order, after the previous ones) and mine the windows they complete.

        Returns:
            the rules of the windows mined by this call (possibly an empty frame), with the columns of ``window_rules``
        """
        if len(rows) == 0:
            return self.rules_.iloc[0:0]
        if self._buffer is not None and list(rows.columns) != list(self._buffer.columns):
            raise ValueError("the new rows must have the same columns as the previous ones")
        if not self._byRows and not isinstance(rows.index, pd.DatetimeIndex):
            raise ValueError("a window by duration ('7D', '1h', ...) requires a sorted DatetimeIndex")
        self._buffer = rows.copy() if self._buffer is None else pd.concat([self._buffer, rows])
        buffer = self._buffer
        if not self._byRows and not buffer.index.is_monotonic_increasing:
            raise ValueError("the rows must arrive in time order")
        new: List[pd.DataFrame] = []
        if self._byRows:
            width, step = int(self._width), int(self._step)                  # type: ignore[arg-type]
            while True:
                start = int(self._cursor) - self._dropped                    # type: ignore[arg-type]
                if start + width > len(buffer):
                    break
                window = buffer.iloc[start:start + width]
                if len(window) >= self._minRows:
                    new.append(self._mine(window))
                    self.n_windows += 1
                self._cursor = int(self._cursor) + step                      # type: ignore[arg-type]
            keep = int(self._cursor) - self._dropped                         # type: ignore[arg-type]
            if keep > 0:
                self._buffer = buffer.iloc[keep:]
                self._dropped += min(keep, len(buffer))
        else:
            span, stride = pd.Timedelta(self._width), pd.Timedelta(self._step)
            if self._cursor is None:
                self._cursor = buffer.index[0]
            last = buffer.index[-1]
            while last >= self._cursor + span:                               # type: ignore[operator]
                lo = int(buffer.index.searchsorted(self._cursor, side="left"))
                hi = int(buffer.index.searchsorted(self._cursor + span, side="left"))     # type: ignore[operator]
                if hi - lo >= self._minRows:
                    new.append(self._mine(buffer.iloc[lo:hi]))
                    self.n_windows += 1
                self._cursor = self._cursor + stride                         # type: ignore[operator]
            self._buffer = buffer.iloc[int(buffer.index.searchsorted(self._cursor, side="left")):]
        self._frames.extend(new)
        if self._history is not None:
            self._frames = self._frames[-self._history:]
        return pd.concat(new, ignore_index=True) if new else self.rules_.iloc[0:0]

    def drift(self, measure: str = "confidence", stable_fraction: float = 0.8) -> pd.DataFrame:
        """``rule_drift`` over the windows kept (renumbered from 0 if `history` dropped the oldest ones)."""
        rules = self.rules_
        if self.n_windows == 0:
            raise ValueError("no window has been completed yet")
        first = max(self.n_windows - self._history, 0) if self._history is not None else 0
        rules = rules.assign(window=rules["window"] - first)
        return rule_drift(rules, measure=measure, stable_fraction=stable_fraction, n_windows=self.n_windows - first)

    def latest_changes(self, measure: str = "confidence", tolerance: float = 0.05) -> pd.DataFrame:
        """``compare_rule_sets`` between the two most recent windows (what appeared, vanished or changed)."""
        if self.n_windows < 2:
            raise ValueError("at least two windows are needed")
        rules = self.rules_
        last = self.n_windows - 1
        return compare_rule_sets(rules[rules["window"] == last - 1], rules[rules["window"] == last], measure, tolerance)
