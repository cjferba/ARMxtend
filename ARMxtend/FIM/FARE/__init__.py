# -*- coding: utf-8 -*-
"""
FARE (Fuzzy Association Rule Extraction): generacion de reglas de asociacion
a partir de itemsets frecuentes difusos, cuyo soporte es un vector de
soportes por alpha-corte (numpy.ndarray), tal y como lo producen
``ARMxtend.FIM.Eclat.FuzzyDECLAT``, ``ARMxtend.FIM.BD_FARE.FuzzyDAprioriTID``
o ``ARMxtend.FFIM`` (FP-Growth difuso).

Es el equivalente difuso, secuencial (sin Spark), de
``ARMxtend.ARM.association_rules``: dado que, fijado un alpha-corte, un
itemset difuso se comporta exactamente como un itemset crisp (su soporte en
ese nivel es un numero en [0, 1]), ambas funciones de este modulo reutilizan
directamente ``ARM.association_rules`` tras extraer el soporte escalar del
nivel alpha de interes, en lugar de reimplementar el calculo de metricas.

@author Carlos Fernandez-Basso
"""
import pandas as pd

from ARMxtend.ARM.association_rules import association_rules


def fuzzy_association_rules(df, alpha_level=0, metric="confidence",
                            min_threshold=0.8, support_only=False):
    """
    Genera las reglas de asociacion difusas validas en un alpha-corte dado.

    Argumentos:
        df : pandas.DataFrame con columnas ['support', 'itemsets'], donde
            'support' es, para cada itemset, un array (o secuencia) con su
            soporte relativo en cada alpha-corte (indice 0 = alpha-corte
            menos exigente)
        alpha_level (int, por defecto 0): indice del alpha-corte sobre el
            que se evaluan y filtran las reglas
        metric, min_threshold, support_only: ver `ARM.association_rules`

    Retorna:
        El mismo formato de pandas.DataFrame que `ARM.association_rules`,
        con las metricas (soporte, confianza, lift, ...) evaluadas en el
        alpha-corte `alpha_level` indicado.
    """
    scalarDf = df.copy()
    scalarDf["support"] = scalarDf["support"].apply(lambda supportVector: supportVector[alpha_level])
    return association_rules(scalarDf, metric=metric, min_threshold=min_threshold, support_only=support_only)


def fuzzy_association_rules_all_alphas(df, num_alpha, metric="confidence",
                                       min_threshold=0.8, support_only=False):
    """
    Igual que `fuzzy_association_rules`, pero evaluando y concatenando el
    resultado para todos los alpha-cortes de 0 a `num_alpha - 1`, anadiendo
    una columna 'alpha' que identifica el nivel de cada fila. Util para
    estudiar como evoluciona una regla (soporte, confianza, ...) a medida
    que se exige un grado de pertenencia mas alto a sus items.

    Retorna:
        pandas.DataFrame con las mismas columnas que `ARM.association_rules`
        mas una columna adicional 'alpha'.
    """
    framesPerAlpha = []
    for alphaLevel in range(num_alpha):
        rulesAtAlpha = fuzzy_association_rules(df, alpha_level=alphaLevel, metric=metric,
                                               min_threshold=min_threshold, support_only=support_only)
        rulesAtAlpha = rulesAtAlpha.copy()
        rulesAtAlpha["alpha"] = alphaLevel
        framesPerAlpha.append(rulesAtAlpha)

    if not framesPerAlpha:
        return pd.DataFrame(columns=["antecedents", "consequents", "alpha"])

    return pd.concat(framesPerAlpha, ignore_index=True)
