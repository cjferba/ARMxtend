# Command line: `armxtend`

Installing the package adds the `armxtend` command (also available as `python -m ARMxtend`). It works on a
single machine, without Spark, and covers the most common one-shot tasks:

```bash
armxtend rules transactions.csv --min-supp 0.1 --min-threshold 0.7
armxtend fuzzy-rules sensors.csv --labels 3 --alpha 10 --metric certainty_factor --min-threshold 0.5
armxtend gradual-rules health.csv --min-supp 0.3 --min-threshold 0.9
armxtend temporal-rules house.csv --time-column time --max-lag 2 --targets heater_on --exclude-persistence
armxtend rule-drift house.csv --time-column time --window 15D --max-lag 1 --targets heater_on
armxtend sequences routines.txt --min-supp 0.3 --min-len 3
armxtend sequential-rules routines.txt --min-supp 0.3 --min-threshold 0.8
armxtend --version
```

## `rules` -- crisp association rules

`transactions.csv`: one transaction per line, items separated by `--delimiter` (default `,`), `-` reads from
standard input. Mines the frequent itemsets with FP-Growth ([`FFIM.fpgrowth`](FFIM.md)) and the rules with
[`ARM.association_rules`](ARM/association_rules.md).

```text
$ armxtend rules transactions.csv --min-supp 0.3 --min-threshold 0.7
antecedents consequents  support  confidence   lift  certainty_factor
          D           B   0.5000      0.8333 1.0417            0.1667
          A           C   0.4000      0.8000 1.1429            0.3333
          C           B   0.5000      0.7143 0.8929           -0.1071
```

## `fuzzy-rules` -- fuzzy association rules from numeric columns

A CSV with a header. Numeric columns (or those in `--columns`) are fuzzified automatically with
[`FuzzyLib.AutoFuzzification`](preprocessing/FuzzyLib.md) (`--labels`, `--method uniform|quantile|kmeans`,
`--membership triangular|trapezoidal|gaussian`), mined with `fuzzy_fpgrowth` (`--alpha` alpha-cuts) and turned
into rules with [`FIM.FARE`](FIM/FARE.md). `--measures` adds extra interest measures as columns.

## `gradual-rules`

"The more A, the more B" rules directly on the numeric columns ([`ARM.gradual_rules`](ARM/gradual_rules.md));
`--max-rows` works on a random sample for big tables.

## `temporal-rules` and `rule-drift` -- sensor CSV files

A CSV with a header and one row per time step, for example:

```text
time,temp,door_open,heater_on
2024-01-01 00:00:00,6.1,0,0
2024-01-01 01:00:00,5.4,1,0
2024-01-01 02:00:00,5.0,0,1
```

`--time-column` makes that column the (sorted) time index. The columns are converted into items as in
[`preprocessing.timeseries`](preprocessing/timeseries.md): numeric columns are split into `--symbols` symbols
(default 3; `--method quantile|uniform|sax`), columns containing only 0/1 (or booleans) are events, text columns give
`column=value` items, and `--calendar hour weekday daytype month part_of_day` adds calendar items (needs
`--time-column`). `--columns` restricts the columns used.

**`temporal-rules`** mines [lagged rules](ARM/temporal_rules.md): `--max-lag N` (default 1), `--lags` (a subset),
`--targets` (items or variables allowed in the consequent), `--exclude-persistence`, `--allow-no-lag` and
`--prune`. In the table format the items read `door_open (t-1)`, `heater_on (t)`; `csv` and `json` keep the raw
names (`door_open@lag1`) and the `lags` / `max_lag` columns.

```text
$ armxtend temporal-rules house.csv --time-column time --calendar part_of_day --max-lag 1 --min-supp 0.02 \
      --min-threshold 0.9 --targets heater_on --exclude-persistence --prune --top 2
                                            antecedents   consequents  support  confidence   lift  certainty_factor
door_open (t-1), part_of_day=evening (t), temp=high (t-1) heater_on (t)   0.0222      1.0000 4.3739            1.0000
                 door_open (t-1), part_of_day=evening (t) heater_on (t)   0.0507      0.9481 4.1466            0.9327
```

**`rule-drift`** mines the rules in consecutive windows and reports how each one evolves
([`ARM.rule_drift`](ARM/rule_drift.md)): `--window` (required: a number of rows such as `96`, or a duration such as
`15D` or `12h` if there is a time column), `--step` (default: the window width), `--min-supp`,
`--metric` / `--min-threshold` (measure that filters the rules and is tracked over time; default `confidence` >= 0.8),
`--max-lag N` (`0`, the default, mines plain rules in each window; `N > 0` lagged rules, with `--targets` and
`--exclude-persistence`) and `--stable-fraction`. The output has one row per rule with `status`
(`stable`, `emerging`, `vanishing`, `intermittent`), `windows`, `persistence`, `first_window`, `last_window`,
the mean of the measure and its `slope`; it is sorted by status and persistence unless `--sort-by` is given.

## `sequences` and `sequential-rules`

Input for both: either a **text file** with one sequence per line, elements separated by `--element-delimiter`
(default `;`) and the simultaneous items of an element by `--delimiter` (default `,`; `-` reads standard input):

```text
alarm;bathroom;kitchen;coffee;tv
alarm;bathroom;kitchen;door
alarm;bathroom,light;kitchen;coffee
```

or an **event log** in CSV with `--sequence-column` (what identifies a sequence: patient, user, day...),
`--time-column` and `--item-column`; `--bin 5min` makes the events of a sequence within the same interval a single
element.

- **`sequences`** mines the [frequent sequential patterns](FIM/sequential.md) (PrefixSpan): `--min-supp`,
  `--min-len`, `--max-len`. Columns: `sequence` (such as `<{alarm} {bathroom} {kitchen}>`), `support`, `length`.
- **`sequential-rules`** mines [sequential rules](ARM/sequential_rules.md) `X => Y`: `--min-supp`, `--metric` /
  `--min-threshold` (default `confidence` >= 0.7; use `--metric lift --min-threshold 1.2` to keep informative
  rules), `--max-antecedent` and `--max-consequent` (default 2).

```text
$ armxtend sequential-rules routines.txt --min-supp 0.4 --min-threshold 0.7 --max-antecedent 1 --max-consequent 1 --top 3
antecedents consequents  support  confidence   lift  certainty_factor
       door     kitchen   0.8000      1.0000 1.2500            1.0000
    kitchen      coffee   0.6000      0.7500 1.2500            0.3750
       door      coffee   0.6000      0.7500 1.2500            0.3750
```

## Options shared by the commands

| Option | Meaning |
|---|---|
| `--min-supp` | minimum support (default 0.1; per window in `rule-drift`) |
| `--metric`, `--min-threshold` | measure used to filter (and by default sort) the rules, and its minimum (default `confidence` >= 0.7; not in `sequences`) |
| `--sort-by`, `--top N` | sort column (descending) and keep the first N rules |
| `--time-column`, `--columns`, `--symbols`, `--method`, `--calendar` | (`temporal-rules`, `rule-drift`) how the CSV becomes items |
| `--format table\|csv\|json` | output format; `table` shows the main columns (`--all-measures` for all), `csv`/`json` always everything |
| `-o FILE` | write to a file instead of standard output |
| `--prune` | (`rules`, `fuzzy-rules`, `temporal-rules`) drop [redundant rules](ARM/redundancy.md) |
| `--html FILE` | (`rules`, `fuzzy-rules`) also write the [interactive VizARE viewer](VizARM.md#interactive-viewer) |

Errors (missing file, a constant column that cannot be fuzzified, ...) print `armxtend: error: ...` to standard
error and exit with status 1.
