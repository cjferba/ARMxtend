# Carlos Fernandez-Basso 2015-2020
# ARMxtend Association Rule Mining Library Extensions
# Author: Carlos Fernandez Basso
"""
Frequent Itemset Mining algorithms.

``FARE`` (sequential, fuzzy association rule extraction) has no dependency
on PySpark and is imported eagerly. The Big Data (Spark) algorithms --
``apriori`` (DApriori/DAprioriTID), ``Eclat`` (DECLAT) and the rule-mining
helpers in ``BD_ARE``/``BD_FARE`` -- require ``pyspark`` and are imported
lazily/explicitly, e.g. ``from ARMxtend.FIM.apriori import DApriori``.
"""

from . import FARE

__all__ = ["FARE", "apriori", "Eclat", "BD_ARE", "BD_FARE"]
