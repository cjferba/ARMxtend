# -*- coding: utf-8 -*-
"""
SFIM shares the FIMoTS implementation with SARE (the only difference between them is that
SARE also extracts association rules): this module re-exports `ARMxtend.SARE.FIMoTS_Structure`.
There used to be an identical copy here that ended up diverging when only one was fixed.
"""
from ...SARE.FIMoTS_Structure import *  # noqa: F401,F403
