#!/usr/bin/env python
# -*- coding: utf-8 -*-

class DecimalFraction(object):
    """
	Clase utilitaria para definir de una fracción simple de números enteros, como 
	representación de un valor decimal.

	Atributos:
    	numeratorFraction (int, opcional, por defecto 0): Valor numerador de la fracción
    	denominatorFraction (int, opcional, por defecto -1): Valor denominador de la fracción
    		
	@author Abel Francisco (2016)
    """

    def __init__(self, numeratorFraction=0, denominatorFraction=1):
        """
    	Inicialización de una nueva fracción de valores enteros.
    	
    	Argumentos:
    		numeratorFraction (int, opcional, por defecto 0): Nuevo valor numerador de la fracción
    		denominatorFraction (int, opcional, por defecto -1): Nuevo valor denominador de la fracción
        """
        self.numeratorFraction = numeratorFraction
        self.denominatorFraction = denominatorFraction

    def __repr__(self):
        """
    	Mecanismo de presentación o escritura en pantalla de una fracción de valores enteros
    	
    	Retorna:
    		Una cadena de caracteres que representa visualmente (texto) a una fracción
    		de números enteros
        """
        return '{} / {}'.format(self.numeratorFraction, self.denominatorFraction)
