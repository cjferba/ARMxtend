# `preprocessing.FuzzyLib`

Loads a tabular dataset and fuzzifies its numeric attributes into named fuzzy sets (e.g.
`Temperature` -> `cold`/`comfort`/`warm`), ready to be used as fuzzy items in
[`FIM.FARE`](../FIM/FARE.md), [`FIM.BD_FARE`](../FIM/BD_FARE.md), [`FIM.Eclat.FuzzyDECLAT`](../FIM/Eclat.md)
or [`FFIM.fuzzy_fpgrowth`](../FFIM.md).

## Loading data

```python
fuzzy_lib = FuzzyLib()
fuzzy_lib.LoadData("path/to/data.csv")   # or set fuzzy_lib.data directly to an existing DataFrame
fuzzy_lib.GetData()        # the underlying pandas.DataFrame
fuzzy_lib.GetAtributes()   # column names
fuzzy_lib.GetTypes()       # column dtypes
```

## Fuzzification

```python
Fuzzification(Atri, thresholds, FuzzyLabel, membership="triangular")
```

- **Atri** (`list[str]`): names of the numeric attributes to fuzzify.
- **thresholds** (`list[list]`): for each attribute, the parameters of its fuzzy sets, one per label:
  ascending *peaks* (`membership="triangular"`), `(a, b, c, d)` vertices (`"trapezoidal"`) or
  `(center, sigma)` pairs (`"gaussian"`).
- **FuzzyLabel** (`list[list[str]]`): for each attribute, the name of each fuzzy label (same length
  as its `thresholds` entry).
- **membership** (`str`): `"triangular"` (default), `"trapezoidal"` or `"gaussian"` -- see below.

With the default `membership="triangular"`, this computes a **triangular Ruspini partition**: given `n` ascending peaks
`p_1 < ... < p_n` (one per label), the two extreme labels are "shoulders" (full membership beyond
their own peak) and the intermediate ones are triangles centered on their peak, so that the
membership degrees of all `n` labels always sum to 1 for every row. It adds one new column
`<attribute>_<label>` per label, with its membership degree, and returns the list of added column
names.

```python
import pandas as pd
from ARMxtend.preprocessing import FuzzyLib

fuzzy_lib = FuzzyLib()
fuzzy_lib.data = pd.DataFrame({"temperature": [15, 19, 22, 25, 28]})
added_columns = fuzzy_lib.Fuzzification(["temperature"], [[18, 22, 26]], [["cold", "comfort", "warm"]])
print(fuzzy_lib.data)
#    temperature  temperature_cold  temperature_comfort  temperature_warm
# 0           15              1.00                 0.00              0.00
# 1           19              0.75                 0.25              0.00
# 2           22              0.00                 1.00              0.00
# 3           25              0.00                 0.25              0.75
# 4           28              0.00                 0.00              1.00
```

Each row's fuzzified columns for that attribute sum to 1: `fuzzy_lib.data[added_columns].sum(axis=1)`.

### Trapezoidal and Gaussian sets

```python
fuzzy_lib.Fuzzification(["temperature"], [[(-float("inf"), -float("inf"), 15, 20),   # cold: 1 up to 15
                                           (15, 20, 25, 30),                         # comfort: core 20-25
                                           (25, 30, float("inf"), float("inf"))]],   # warm: 1 from 30
                        [["cold", "comfort", "warm"]], membership="trapezoidal")

fuzzy_lib.Fuzzification(["temperature"], [[(15.0, 3.0), (22.0, 3.0)]], [["cold", "warm"]],
                        membership="gaussian")        # exp(-(x-center)^2 / (2 sigma^2))
```

A trapezoid `(a, b, c, d)` is 0 up to `a`, rises to 1 at `b`, stays 1 until `c` and falls to 0 at `d`
(`a == b` / `c == d` give vertical sides; `-inf`/`+inf` give shoulders). Trapezoids and Gaussians
need not sum to 1 per row. The functions are also available on their own in
`ARMxtend.preprocessing.membership`.

## Automatic fuzzification

```python
AutoFuzzification(Atri=None, n_labels=3, method="quantile", FuzzyLabel=None, membership="triangular")
```

Computes each attribute's partition **from its data** (all numeric columns if `Atri` is omitted), so
you do not need to choose breakpoints by hand:

| `method` | Peaks |
|---|---|
| `"uniform"` | evenly spaced between the min and max |
| `"quantile"` (default) | evenly spaced quantiles (min, ..., median, ..., max): every label covers about the same number of rows, which matters for skewed attributes |
| `"kmeans"` | centers of a 1-D k-means (deterministic, quantile-initialised): follows natural clusters |

and `membership` is `"triangular"` and `"trapezoidal"` (both Ruspini partitions, rows sum to 1) or
`"gaussian"` (width chosen so neighbouring sets cross at degree 0.5). Default labels are `low/high`,
`low/medium/high`, `low/medium_low/medium_high/high`, `very_low/.../very_high`, or `L1..Ln`. The
resulting parameters are stored in `fuzzy_lib.partitions[attribute]["params"]` so you can reuse them
with `Fuzzification` on new data. A constant attribute (not enough distinct values for `n_labels`
peaks) raises a `ValueError` naming it.

```python
fuzzy_lib.AutoFuzzification(["age", "cholesterol"], n_labels=3, method="quantile")
# -> ['age_low', 'age_medium', 'age_high', 'cholesterol_low', 'cholesterol_medium', 'cholesterol_high']
```

## Feeding the fuzzy miners

```python
GetFuzzyTransactions(columns=None, min_degree=0.0)
```

Returns one list of `(item, degree)` pairs per row -- the input format of
[`FFIM.fuzzy_fpgrowth`](../FFIM.md) and of the fuzzy rule functions
([`ARM.negative_rules`](../ARM/negative_rules.md), [`ARM.meta_rules`](../ARM/meta_rules.md)) -- using
the columns added by `Fuzzification`/`AutoFuzzification` (or those in `columns`) and skipping items
whose degree is not greater than `min_degree`:

```python
from ARMxtend.FFIM import fuzzy_fpgrowth
from ARMxtend.FIM.FARE import fuzzy_association_rules

itemsets = fuzzy_fpgrowth(fuzzy_lib.GetFuzzyTransactions(), min_supp=0.1, num_alpha=10)
rules = fuzzy_association_rules(itemsets, 10, metric="certainty_factor", min_threshold=0.5)
```
