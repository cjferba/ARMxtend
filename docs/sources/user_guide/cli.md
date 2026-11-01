# Command line: `armxtend`

Installing the package adds the `armxtend` command (also available as `python -m ARMxtend`). It works on a
single machine, without Spark, and covers the most common one-shot tasks:

```bash
armxtend rules transactions.csv --min-supp 0.1 --min-threshold 0.7
armxtend generalized-rules basket.txt --taxonomy taxonomy.csv --min-supp 0.2 --level-supp 2=0.1 3=0.04 --interest 1.2
armxtend fuzzy-rules sensors.csv --labels 3 --alpha 10 --metric certainty_factor --min-threshold 0.5
armxtend gradual-rules health.csv --min-supp 0.3 --min-threshold 0.9
armxtend temporal-rules house.csv --time-column time --max-lag 2 --targets heater_on --exclude-persistence
armxtend rule-drift house.csv --time-column time --window 15D --max-lag 1 --targets heater_on
armxtend sequences routines.txt --min-supp 0.3 --min-len 3
armxtend sequential-rules routines.txt --min-supp 0.3 --min-threshold 0.8
armxtend fuzzy-sequences readings.csv --sequence-column day --time-column time --min-degree 0.2 --min-supp 0.3
armxtend fuzzy-sequential-rules readings.csv --sequence-column day --time-column time --min-degree 0.2 --metric lift --min-threshold 1.5
armxtend anomalies house.csv --time-column time --train-fraction 0.6 --max-lag 1 --min-threshold 0.95
armxtend forecast house.csv --time-column time --train-fraction 0.6 --max-lag 1 --targets heater_on
armxtend next-events routines.txt --min-supp 0.2 --min-threshold 0.3 --history "alarm;bathroom"
armxtend classify patients.csv --class-column diagnosis --test-fraction 0.3
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

## `generalized-rules`

Multilevel rules over a [taxonomy](FIM/taxonomy.md): the same transaction file as `rules` (one transaction per line) plus
`--taxonomy FILE`, a CSV with a header and two columns, `child` and `parent` (the same `--delimiter`). The options
`--min-supp`, `--metric`, `--min-threshold` and `--prune` are those of `rules`, and the [significance options](#significance-of-the-rules)
work too. Besides:

- `--level-supp LEVEL=SUPPORT ...` sets a minimum support for the deeper levels (`--min-supp` is level 1 and the default of the
  levels not listed), for example `--level-supp 2=0.1 3=0.04`;
- `--interest R` keeps only the [R-interesting rules](ARM/generalized_rules.md#rules-that-only-echo-their-category-interesting_rules)
  and adds the `interest` column; `--max-len N` bounds the size of a rule.

```text
$ armxtend generalized-rules shop.txt --taxonomy taxonomy.csv --min-supp 0.3 --min-threshold 0.6 --interest 1.1
 antecedents  consequents  support  confidence   lift  certainty_factor  level  interest
Hiking Boots      Clothes   0.3333      1.0000 1.5000            1.0000      2       NaN
Hiking Boots    Outerwear   0.3333      1.0000 2.0000            1.0000      2    1.3333
   Outerwear     Footwear   0.3333      0.6667 1.0000            0.0000      2       NaN
   Outerwear Hiking Boots   0.3333      0.6667 2.0000            0.5000      2    2.0000
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

## `fuzzy-sequences` and `fuzzy-sequential-rules`

The same two tasks as `sequences` and `sequential-rules`, for **numeric** readings. The input is a CSV with a header
and one row per time step, for example `time,day,temp,power`. The numeric columns (or those in `--columns`) are fuzzified
with [`FuzzyLib.AutoFuzzification`](preprocessing/FuzzyLib.md) (`--labels`, `--method uniform|quantile|kmeans`,
`--membership triangular|trapezoidal|gaussian`), each row becomes an event with the labels whose membership is above
`--min-degree` (default 0; `0.2` removes the noise of tiny memberships), and `--sequence-column` groups the rows into
sequences (the day, the patient; without it the whole file is one sequence). `--time-column` sorts the rows inside each
sequence; without it the file order is used. `--tnorm min|product` chooses how the degrees of a pattern are combined.

- **`fuzzy-sequences`** mines the [fuzzy sequential patterns](FIM/fuzzy_sequential.md): `--min-supp` (the fuzzy support),
  `--min-len`, `--max-len`. Columns: `sequence`, `support`, `length`.
- **`fuzzy-sequential-rules`** mines the [fuzzy sequential rules](ARM/fuzzy_sequential_rules.md) `X => Y`: `--min-supp`,
  `--metric` / `--min-threshold` (default `confidence` >= 0.7; fuzzy confidences are usually lower than crisp ones, so
  `--metric lift --min-threshold 1.5` is often a better filter), `--max-antecedent` and `--max-consequent` (default 2).

```text
$ armxtend fuzzy-sequential-rules readings.csv --columns temp power --sequence-column day --time-column time \
      --min-degree 0.2 --min-supp 0.2 --metric lift --min-threshold 1.5 --max-antecedent 1 --max-consequent 1
antecedents consequents  support  confidence   lift  certainty_factor
  temp_high  power_high   0.4343      0.9759 2.1261            0.9554
```

## Rules in action: `anomalies`, `forecast`, `next-events` and `classify`

These four commands learn rules and apply them straight away ([`ARM.anomaly`](ARM/anomaly_detection.md),
[`ARM.prediction`](ARM/prediction.md), [`ARM.classification`](ARM/classification.md)).

**`anomalies`** and **`forecast`** read a sensor CSV like `temporal-rules` (`--time-column`, `--columns`, `--symbols`,
`--method`, `--calendar`) and use the first `--train-fraction` of the rows (default 0.5) to learn the rules -- lagged
rules up to `--max-lag` steps back (default 1; `0` for rules within the same row) with only past items in the antecedent --
and the rest of the rows to apply them. The rule options are those of `temporal-rules` (`--min-supp`, `--metric`,
`--min-threshold`, `--targets`, `--exclude-persistence`), plus `--single-consequent` and `--min-improvement X` (drop the
[unproductive rules](ARM/redundancy.md#unproductive-rules)) and the [significance options](#significance-of-the-rules)
(`--alpha`, `--correction`, `--test`) to keep only rules that are not noise.

```text
$ armxtend anomalies house.csv --time-column time --train-fraction 0.6 --min-supp 0.02 --min-threshold 0.75 \
      --targets light chime --exclude-persistence --single-consequent --alpha 0.05 --min-improvement 0.05 \
      --min-score 6 --top 5
2 rules learned from the first 3000 rows
               time  score  violations                                               rules
2024-03-23 15:00:00 9.3699           2 motion (t-1) => light (t) | door (t-1) => chime (t)
2024-03-29 18:40:00 9.3699           2 motion (t-1) => light (t) | door (t-1) => chime (t)
2024-03-22 19:40:00 7.2400           1                           motion (t-1) => light (t)
2024-03-22 21:20:00 7.2400           1                           motion (t-1) => light (t)
2024-03-23 13:20:00 7.2400           1                           motion (t-1) => light (t)
```

`anomalies` lists the analysed rows that violate some rule, highest `score` first (`--min-score`, `--weight
surprise|confidence|count`, `--top`). `forecast` prints how well the rules predict each item on the analysed rows
(`--threshold`: the confidence from which an item is predicted) or, with `--next`, the prediction for the step after
the last row:

```text
$ armxtend forecast house.csv --time-column time --train-fraction 0.6 --min-supp 0.02 --min-threshold 0.75 \
      --targets light chime --exclude-persistence --single-consequent --threshold 0.7
7 rules learned from the first 3000 rows; evaluated on the remaining 2000
 item    n  positives  predicted  hits  precision  recall     f1  base_rate
chime 1999        387        373   298     0.7989  0.7700 0.7842     0.1936
light 1999        725        633   594     0.9384  0.8193 0.8748     0.3627
```

**`next-events`** takes the same input as [`sequential-rules`](#sequences-and-sequential-rules) and mines
sequential rules (`--min-supp`, `--min-threshold` = minimum confidence, default 0.3, `--max-antecedent`,
`--max-consequent`). With `--history "alarm;bathroom;kitchen"` (elements separated by `--element-delimiter`) it ranks the
events still to come; with `--test-fraction 0.3` it learns from the first sequences and measures `hit_rate` on the last
30 % (`-k`, `--min-prefix`, `--target future|next`), next to the popularity baseline:

```text
$ armxtend next-events events.csv --sequence-column day --time-column time --item-column event \
      --min-supp 0.2 --history "alarm;bathroom;kitchen;coffee"
40 sequential rules learned from 120 sequences
 item  score
   tv 0.7727
phone 0.6667
 door 0.5917
```

(`door` is still listed after `coffee` because the rule `kitchen => door` does not know that a coffee rules out the
rushed branch: rules cannot veto each other.)

**`classify`** reads a CSV with a column of classes (`--class-column`); numeric columns are discretised
(`--symbols`, `--method`), 0/1 columns are events and text columns give `column=value` items. It learns a
[CBA classifier](ARM/classification.md) (`--min-supp` within each class, `--min-threshold` = minimum confidence,
`--max-len`, `--support class|global`, `--min-improvement`) and prints the selected rules, in the order they are
applied; the accuracy on the `--test-fraction` of rows held out at random (default 0.3, `--seed`) goes to standard error:

```text
$ armxtend classify patients.csv --class-column diagnosis --min-threshold 0.7 --min-improvement 0.1
2 class rules selected from 16 candidates; default class: healthy; accuracy on 360 held-out rows: 0.958 (majority class: 0.550)
    antecedents       consequents  support  confidence   lift
   cough, fever     diagnosis=flu   0.2179      0.9632 4.1278
itchy, sneezing diagnosis=allergy   0.2012      0.7412 3.3839
```

## Significance of the rules

`rules`, `temporal-rules` and `rule-drift` can drop the rules that are not statistically significant
([`ARM.significance`](ARM/significance.md)):

```bash
armxtend rules baskets.csv --min-supp 0.03 --min-threshold 0.45 --alpha 0.05                 # Fisher + FDR
armxtend rules baskets.csv --alpha 0.05 --correction holm --test chi2
armxtend temporal-rules house.csv --time-column time --max-lag 1 --alpha 0.05                # Fisher + FDR
armxtend temporal-rules house.csv --time-column time --max-lag 1 --permutations 200 --alpha 0.05   # circular shifts
armxtend rule-drift house.csv --time-column time --window 15D --alpha 0.05                    # per window
```

| Option | Meaning |
|---|---|
| `--alpha A` | keep only the rules significant at level `A` after correcting for multiple tests (adds `p_value` and `p_adjusted`) |
| `--correction` | `bonferroni`, `holm`, `fdr_bh` (default) or `fdr_by` |
| `--test` | `fisher` (default, exact) or `chi2` |
| `--permutations N` | (`temporal-rules`) test every rule against `N` shuffled copies of the series instead (adds `p_permutation`; keeps `p_permutation <= --alpha` if given); the right choice for time series |
| `--null`, `--seed` | how the series are shuffled (`shift`, the default, keeps each series' autocorrelation; or `permute`) and the random seed |

`fuzzy-rules` and the sequence commands have no significance test.

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
