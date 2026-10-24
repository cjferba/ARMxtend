# -*- coding: utf-8 -*-
"""
Poda de reglas de asociacion redundantes.

Una regla  X2 -> Y2  es REDUNDANTE si existe otra regla distinta  X1 -> Y1
mas general -- con antecedente menor o igual (X1 incluido en X2) y
consecuente mayor o igual (Y1 incluye a Y2) -- que tiene EXACTAMENTE el
mismo soporte y la misma confianza (Bastide, Taouil, Pasquier, Stumme &
Lakhal, 2000, "Mining minimal non-redundant association rules using frequent
closed itemsets"; ver tambien Zaki, 2004). La regla general transmite toda la informacion de la
redundante: exige menos condiciones y concluye al menos lo mismo.

Funciona con cualquier DataFrame de reglas con columnas 'antecedents' y
'consequents' (frozensets) mas 'support' y 'confidence': la salida de
``ARM.association_rules``, ``FIM.FARE.fuzzy_association_rules``, las reglas
con items ausentes de ``ARM.negative_rules`` o las graduales de
``ARM.gradual_rules``.

@author Carlos Fernandez-Basso
"""
import math


def prune_redundant_rules(rules, tol=1e-9, support_column="support", confidence_column="confidence"):
    """
    Elimina las reglas redundantes de `rules`.

    Argumentos:
        rules (pandas.DataFrame): reglas con columnas 'antecedents',
            'consequents', `support_column` y `confidence_column`
        tol (float): tolerancia absoluta para considerar iguales dos valores
            de soporte/confianza
        support_column, confidence_column (str): nombres de las columnas

    Retorna:
        pandas.DataFrame con las reglas no redundantes (misma estructura y
        orden relativo que `rules`; las reglas duplicadas se reducen a una).
    """
    if len(rules) == 0:
        return rules.copy()

    support = rules[support_column].to_numpy(dtype=float)
    confidence = rules[confidence_column].to_numpy(dtype=float)
    antecedents = [frozenset(a) for a in rules["antecedents"]]
    consequents = [frozenset(c) for c in rules["consequents"]]

    # Solo pueden ser equivalentes las reglas con (soporte, confianza) iguales:
    # se agrupan por esos valores redondeados a `tol` y se comparan dentro del grupo.
    digits = max(0, int(round(-math.log10(tol)))) if tol > 0 else 12
    groups = {}
    for position in range(len(rules)):
        groups.setdefault((round(support[position], digits), round(confidence[position], digits)), []).append(position)

    redundant = set()
    for positions in groups.values():
        for candidate in positions:
            for general in positions:
                if general == candidate or general in redundant:
                    continue
                if abs(support[general] - support[candidate]) > tol or abs(confidence[general] - confidence[candidate]) > tol:
                    continue
                isMoreGeneral = antecedents[general] <= antecedents[candidate] and consequents[general] >= consequents[candidate]
                isIdentical = antecedents[general] == antecedents[candidate] and consequents[general] == consequents[candidate]
                # Una regla identica solo elimina a las de indice mayor (se conserva la primera)
                if isMoreGeneral and (not isIdentical or general < candidate):
                    redundant.add(candidate)
                    break

    return rules.iloc[[p for p in range(len(rules)) if p not in redundant]].reset_index(drop=True)
