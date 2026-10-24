# Carlos Fernandez-Basso 2015-2020
# ARMxtend Association Rule Mining Library Extensions
# Author: Carlos Fernandez Basso
"""
Frequent Itemset Mining algorithms.

``FARE`` (sequential, fuzzy association rule extraction) and the closed/maximal
itemset helpers in ``condensed`` have no dependency on PySpark and are imported
eagerly. The Big Data (Spark) algorithms --
``apriori`` (DApriori/DAprioriTID), ``Eclat`` (DECLAT) and the rule-mining
helpers in ``BD_ARE``/``BD_FARE`` -- require ``pyspark`` and are imported
lazily/explicitly, e.g. ``from ARMxtend.FIM.apriori import DApriori``.
"""

from . import FARE
from .condensed import closed_itemsets, maximal_itemsets, support_from_closed

__all__ = ["FARE", "closed_itemsets", "maximal_itemsets", "support_from_closed", "apriori", "Eclat", "BD_ARE", "BD_FARE"]
