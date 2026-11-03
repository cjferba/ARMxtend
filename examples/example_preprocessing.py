# -*- coding: utf-8 -*-
"""
Example: fuzzification of a numeric attribute (preprocessing.FuzzyLib) as a
previous step to the fuzzy itemset/association rule mining algorithms
(FFIM.fuzzy_fpgrowth, FIM.FARE, FIM.BD_FARE).
"""
import pandas as pd

from ARMxtend.preprocessing import FuzzyLib


def main():
    fuzzyLib = FuzzyLib()
    fuzzyLib.data = pd.DataFrame({
        "temperature": [17.0, 19.5, 21.0, 23.0, 25.5, 28.0],
        "sensor_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
    })
    fuzzyLib.atributes = list(fuzzyLib.data.columns)

    addedColumns = fuzzyLib.Fuzzification(
        Atri=["temperature"],
        thresholds=[[18, 21, 25]],
        FuzzyLabel=[["cold", "comfortable", "warm"]],
    )

    print("Fuzzy columns added:", addedColumns)
    print()
    print(fuzzyLib.GetData().to_string(index=False))

    return fuzzyLib.GetData()


if __name__ == "__main__":
    main()
