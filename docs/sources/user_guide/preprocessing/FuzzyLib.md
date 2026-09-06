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
Fuzzification(Atri, thresholds, FuzzyLabel)
```

- **Atri** (`list[str]`): names of the numeric attributes to fuzzify.
- **thresholds** (`list[list[float]]`): for each attribute, one ascending *peak* per fuzzy label.
- **FuzzyLabel** (`list[list[str]]`): for each attribute, the name of each fuzzy label (same length
  as its `thresholds` entry).

For each attribute, this computes a **triangular Ruspini partition**: given `n` ascending peaks
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

To turn a fuzzified row into the `(item, degree)` pairs expected by the fuzzy mining algorithms,
pair each added column name with its value, e.g.
`[(col, row[col]) for col in added_columns if row[col] > 0]`.
