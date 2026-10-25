# Command line: `armxtend`

Installing the package adds the `armxtend` command (also available as `python -m ARMxtend`). It works on a
single machine, without Spark, and covers the most common one-shot tasks:

```bash
armxtend rules transactions.csv --min-supp 0.1 --min-threshold 0.7
armxtend fuzzy-rules sensors.csv --labels 3 --alpha 10 --metric certainty_factor --min-threshold 0.5
armxtend gradual-rules health.csv --min-supp 0.3 --min-threshold 0.9
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

## Options shared by all commands

| Option | Meaning |
|---|---|
| `--min-supp` | minimum support (default 0.1) |
| `--metric`, `--min-threshold` | measure used to filter (and by default sort) the rules, and its minimum (default `confidence` >= 0.7) |
| `--sort-by`, `--top N` | sort column (descending) and keep the first N rules |
| `--format table\|csv\|json` | output format; `table` shows the main columns (`--all-measures` for all), `csv`/`json` always everything |
| `-o FILE` | write to a file instead of standard output |
| `--prune` | (`rules`, `fuzzy-rules`) drop [redundant rules](ARM/redundancy.md) |
| `--html FILE` | (`rules`, `fuzzy-rules`) also write the [interactive VizARE viewer](VizARM.md#interactive-viewer) |

Errors (missing file, a constant column that cannot be fuzzified, ...) print `armxtend: error: ...` to standard
error and exit with status 1.
