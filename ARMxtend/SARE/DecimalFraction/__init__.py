#!/usr/bin/env python
# -*- coding: utf-8 -*-

class DecimalFraction(object):
    """
    Utility class to define a simple fraction of integers, as a representation of a decimal value.

    Attributes:
        numeratorFraction (int, optional, default 0): Numerator value of the fraction
        denominatorFraction (int, optional, default -1): Denominator value of the fraction

    @author Abel Francisco (2016)
    """

    def __init__(self, numeratorFraction=0, denominatorFraction=1):
        """
        Initialization of a new fraction of integer values.

        Args:
            numeratorFraction (int, optional, default 0): New numerator value of the fraction
            denominatorFraction (int, optional, default -1): New denominator value of the fraction
        """
        self.numeratorFraction = numeratorFraction
        self.denominatorFraction = denominatorFraction

    def __repr__(self):
        """
        Mechanism to display or print a fraction of integer values

        Returns:
            A string that visually represents (text) a fraction of integers
        """
        return '{} / {}'.format(self.numeratorFraction, self.denominatorFraction)
