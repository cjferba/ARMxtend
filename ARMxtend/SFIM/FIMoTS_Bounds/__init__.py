# -*- coding: utf-8 -*-
"""
SFIM comparte con SARE la implementacion de FIMoTS (la unica diferencia entre ambos es que
SARE ademas extrae reglas de asociacion): este modulo reexporta `ARMxtend.SARE.FIMoTS_Bounds`.
Antes habia aqui una copia identica que termino divergiendo cuando solo se corrigio una.
"""
from ...SARE.FIMoTS_Bounds import *  # noqa: F401,F403
