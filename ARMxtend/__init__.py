# Carlos Fernandez-Basso 2015-2020
# ARMxtend Association Rule Mining Library Extensions
# Author: Carlos Fernandez Basso
"""
ARMxtend: a collection of association rule mining tools, covering the crisp
and fuzzy cases, on a single machine (``ARM``, ``FIM``, ``FFIM``) as well as
on Spark for Big Data and streaming scenarios (``FIM.BD_ARE``,
``FIM.BD_FARE``, ``SARE``, ``SFIM``).

Only ``ARM``, ``FIM.FARE``, ``FFIM``, ``VizARM`` and ``preprocessing`` are
imported eagerly here, since they have no dependency on PySpark. The Big
Data/streaming submodules (``FIM.apriori``, ``FIM.Eclat``, ``FIM.BD_ARE``,
``FIM.BD_FARE``, ``SARE``, ``SFIM``) require ``pyspark`` and should be
imported explicitly by the code that needs them, e.g.
``from ARMxtend.SARE import main``.
"""

__version__ = "0.1.0"

from . import ARM
from . import FFIM
from . import VizARM
from . import preprocessing
from .FIM import FARE

__all__ = ["ARM", "FIM", "FFIM", "SARE", "SFIM", "VizARM", "preprocessing", "FARE"]
