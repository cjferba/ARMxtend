# `preprocessing.timeseries` -- from time series and sensor data to items

Association rules work on **items**; sensors and logs produce numbers, timestamps and states. This module
turns an ordered table (one row per time step) into the item tables used by
[`ARM.temporal_rules`](../ARM/temporal_rules.md), [`ARM.rule_drift`](../ARM/rule_drift.md) and the other
miners of the library.

```python
import pandas as pd
from ARMxtend.preprocessing.timeseries import discretize, to_items, calendar_items, lag_items

house = pd.read_csv("house.csv", index_col="time", parse_dates=True)   # temp, door_open, heater_on

symbols = discretize(house, ["temp"], n_symbols=3, method="quantile")  # temp -> 'low' | 'medium' | 'high'
items = to_items(symbols)                                              # boolean table of items
items = items.join(calendar_items(items.index, ["part_of_day"]))       # + 'part_of_day=evening', ...
items.columns   # temp=high, temp=low, temp=medium, door_open, heater_on, part_of_day=afternoon, ...
```

## Conventions

- An **item table** is a `DataFrame` with one row per time step (sorted in time) and one column per item.
  Boolean columns are crisp items; numeric columns in `[0, 1]` are **membership degrees** (fuzzy items, e.g.
  the ones [`FuzzyLib`](FuzzyLib.md) produces).
- Categorical items are named `column=value` (`temp=high`); boolean sensor events keep the column name
  (`door_open`).
- The item `x` seen `k` steps earlier is named `x@lagk` (`temp=high@lag2`); the present (lag 0) has no suffix.
  `lagged`, `split_lag` and `item_variable` build and parse these names.
- Item names **must not contain `-`** (it is the item separator of the itemset keys in `FIM._shared`):
  `to_items` replaces it by `_`, and the miners raise `ValueError` if they find one.

## Functions

| Function | What it does |
|---|---|
| `symbolize(values, n_symbols=3, method="quantile", labels=None)` | numeric series -> symbols (`low`/`medium`/`high`, ...). `method`: `quantile` (equal frequency), `uniform` (equal width) or `sax` (z-normalise, then Gaussian breakpoints, after Lin et al. 2007; point-wise, **without** the PAA aggregation of the original paper). `NaN` stays missing. |
| `discretize(df, columns=None, n_symbols=3, method="quantile", labels=None)` | `symbolize` applied to several columns (default: all numeric non-boolean ones); the rest are left untouched. |
| `to_items(df, columns=None, sep="=")` | categorical columns -> `column=value` items; boolean columns -> one item named after the column (present when `True`). |
| `calendar_items(index, parts=("hour", "weekday"))` | calendar items for each timestamp: `hour=08`, `weekday=Mon`, `daytype=weekend`/`workday`, `month=Jan`, `part_of_day=night`/`morning`/`afternoon`/`evening` (0-5, 6-11, 12-17, 18-23 h). Lets a rule hold "only in the evening". |
| `lag_items(df, lags, drop_incomplete=True)` | adds the items of the previous steps: one column `x@lagk` per lag `k` in `lags` (0 keeps the present). The first `max(lags)` rows have no history and are dropped unless `drop_incomplete=False`. Works for boolean and for membership-degree tables. |
| `window_items(df, width, step=1, how="any")` | one row per window of `width` steps: an item is present if it appears in `any` / `all` of the window (max / min of the degrees if fuzzy) or with the fraction of the window (`mean`). For "within a window" co-occurrence without fixing the order. |
| `to_transactions(items)`, `to_fuzzy_transactions(memberships, min_degree=0.0)` | the formats `FFIM.fpgrowth` and `FFIM.fuzzy_fpgrowth` take. |
| `check_item_names(names)` | raises `ValueError` for names containing `-`. |

## Choosing the number of symbols

The cost of lagged rule mining grows with *(items) x (lags)*. With `n_symbols=3` a numeric column yields three
items; with 10 it yields ten and most itemsets will not reach any reasonable support. Start with 3 (or 5
at most) and use `method="quantile"` unless the physical thresholds matter (then bin with `pandas.cut` and
pass the result to `to_items`).
