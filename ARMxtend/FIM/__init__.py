# Carlos Fernandez-Basso 2015-2020
# ARMxtend Association Rule Mining Library Extensions
# Author: Carlos Fernandez Basso
"""
Frequent Itemset Mining algorithms.

``FARE`` (sequential, fuzzy association rule extraction), the closed/maximal
itemset helpers in ``condensed`` and the sequential-pattern miner ``prefixspan``
have no dependency on PySpark and are imported eagerly (``prefixspan_spark``
imports ``pyspark`` only when it is called). The Big Data (Spark) algorithms --
``apriori`` (DApriori/DAprioriTID), ``Eclat`` (DECLAT) and the rule-mining
helpers in ``BD_ARE``/``BD_FARE`` -- require ``pyspark`` and are imported
lazily/explicitly, e.g. ``from ARMxtend.FIM.apriori import DApriori``.
"""

from . import FARE
from .condensed import closed_itemsets, maximal_itemsets, support_from_closed
from .sequential import (
    contains_sequence,
    format_sequence,
    prefixspan,
    prefixspan_spark,
    sequences_from_events,
    sequences_from_items,
)

__all__ = ["FARE", "closed_itemsets", "maximal_itemsets", "support_from_closed", "prefixspan", "prefixspan_spark",
           "contains_sequence", "format_sequence", "sequences_from_events", "sequences_from_items",
           "apriori", "Eclat", "BD_ARE", "BD_FARE"]
