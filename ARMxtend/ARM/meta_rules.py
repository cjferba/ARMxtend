# -*- coding: utf-8 -*-
"""
Meta-reglas de asociacion (crisp y difusas): mineria de asociaciones sobre
un conjunto de reglas de asociacion YA extraidas de multiples datasets
D_1, ..., D_k, en lugar de sobre los datos originales. Implementa los
Algoritmos 3 (crisp) y 4 (difuso) de:

    Ruiz, M.D., Gomez-Romero, J., Molina-Solana, M., Campana, J.R.,
    Martin-Bautista, M.J. (2016). "Meta-association rules for mining
    interesting associations in multiple datasets". Applied Soft Computing,
    49, 212-223.

Idea general (Definicion 3 del articulo): dados k conjuntos de reglas
R_1, ..., R_k extraidos de k datasets D_1, ..., D_k (mismo dominio, quiza
distinta estructura), se construye una "meta-base de datos" D en la que cada
FILA es un dataset D_j y cada COLUMNA (item) es una regla r_i (mas,
opcionalmente, atributos adicionales sobre el dataset). Sobre esa
meta-base de datos se aplica de nuevo un algoritmo de mineria de reglas de
asociacion, obteniendo asi meta-reglas como r1 ^ r2 -> r3 (co-ocurrencia de
reglas en una alta proporcion de datasets) o r1 -> at2 (relacion entre una
regla y un atributo del dataset).

Se distinguen dos variantes:
    * Meta-reglas CRISP (Algoritmo 3): la meta-base de datos es booleana,
      1 si la regla r_i fue extraida en el dataset D_j, 0 si no. Solo tiene
      en cuenta la presencia/ausencia de la regla, no su fuerza.
    * Meta-reglas DIFUSAS (Algoritmo 4): la meta-base de datos usa como
      grado de pertenencia el valor de una medida de la regla (soporte o
      factor de certeza) en cada dataset, permitiendo distinguir reglas
      "primarias" muy fiables de otras marginales.

NOTA IMPORTANTE: los identificadores de regla e items no deben contener el
caracter '-' (usado como separador de items en las claves de itemset de
`FIM._shared`); use `rule_key()` para construir identificadores seguros.

@author Carlos Fernandez-Basso
"""
import pandas as pd

from ..FIM._shared import key_to_itemset
from ..FIM.FARE import fuzzy_association_rules
from ..FFIM import fpgrowth, fuzzy_fpgrowth
from .association_rules import association_rules


def rule_key(antecedent, consequent):
    """
    Construye un identificador de regla seguro (sin '-') para usarlo como
    item en una meta-base de datos, p.ej. rule_key(['A','B'], ['C']) ->
    'A,B=>C'.
    """
    return "{}=>{}".format(",".join(sorted(antecedent)), ",".join(sorted(consequent)))


def mine_primary_rule_measures(datasets, min_supp=0.1, min_conf=0.5, metric="confidence"):
    """
    Mina, para cada dataset de `datasets`, sus reglas de asociacion crisp
    (pasos 1-3 de los Algoritmos 3 y 4: "for all Di, obtain Ri using
    Algorithm 1"), usando `FFIM.fpgrowth` + `ARM.association_rules`.

    Argumentos:
        datasets (Sequence[Sequence[Iterable[str]]]): un dataset D_j por
            elemento, cada uno una lista de transacciones (colecciones de
            items)
        min_supp (float): soporte minimo para la mineria de itemsets
            frecuentes en cada dataset
        min_conf (float): umbral minimo de `metric` para conservar una regla
        metric (str): medida de `ARM.association_rules` usada como umbral y
            como valor asociado a la regla en la meta-base de datos difusa

    Retorna:
        list[dict[str, float]]: para cada dataset, un diccionario
        {rule_key(antecedente, consecuente): valor de `metric`}
    """
    ruleMeasuresPerDataset = []
    for transactions in datasets:
        freqItemsets = fpgrowth(transactions, min_supp)
        itemsetsDf = pd.DataFrame({
            "support": list(freqItemsets.values()),
            "itemsets": [key_to_itemset(key) for key in freqItemsets],
        })
        rulesDf = association_rules(itemsetsDf, metric=metric, min_threshold=min_conf)

        ruleMeasures = {
            rule_key(row["antecedents"], row["consequents"]): float(row[metric])
            for _, row in rulesDf.iterrows()
        }
        ruleMeasuresPerDataset.append(ruleMeasures)

    return ruleMeasuresPerDataset


def build_crisp_meta_database(rule_sets, attributes=None):
    """
    Construye la meta-base de datos booleana D (Tabla 7 del articulo).

    Argumentos:
        rule_sets (Sequence[Set[str]] | Sequence[dict[str, float]]): para
            cada dataset D_j, el conjunto de claves de las reglas
            encontradas en R_j (o un dict, del que solo se usan las claves,
            p.ej. la salida de `mine_primary_rule_measures`)
        attributes (Sequence[dict[str, int]], opcional): para cada dataset,
            atributos booleanos adicionales {nombre_atributo: 0/1} (ver
            Definicion 3)

    Retorna:
        pandas.DataFrame con una fila por dataset y una columna booleana
        (0/1) por cada regla y atributo distintos encontrados
    """
    ruleKeys = sorted(set().union(*(set(rs) for rs in rule_sets))) if rule_sets else []

    rows = []
    for index, ruleSet in enumerate(rule_sets):
        row = {ruleKey: int(ruleKey in ruleSet) for ruleKey in ruleKeys}
        if attributes:
            row.update(attributes[index])
        rows.append(row)

    return pd.DataFrame(rows)


def crisp_meta_association_rules(rule_sets, attributes=None, min_supp=0.5, min_conf=0.5, metric="confidence"):
    """
    Algoritmo 3: mineria de meta-reglas de asociacion crisp.

    Argumentos:
        rule_sets (Sequence[Set[str]] | Sequence[dict[str, float]]): ver
            `build_crisp_meta_database`
        attributes (Sequence[dict[str, int]], opcional): ver
            `build_crisp_meta_database`
        min_supp (float): soporte minimo para que una meta-regla se
            considere frecuente (proporcion de datasets en los que
            co-ocurren sus reglas/atributos)
        min_conf (float): umbral minimo de `metric` para una meta-regla
        metric (str): medida de `ARM.association_rules` para filtrar

    Retorna:
        pandas.DataFrame de meta-reglas (mismo formato que
        `ARM.association_rules`): sus 'antecedents'/'consequents' son
        frozensets de claves de regla y/o nombres de atributo.
    """
    metaDatabase = build_crisp_meta_database(rule_sets, attributes)
    transactions = [
        {column for column in metaDatabase.columns if row[column] == 1}
        for _, row in metaDatabase.iterrows()
    ]

    freqItemsets = fpgrowth(transactions, min_supp)
    itemsetsDf = pd.DataFrame({
        "support": list(freqItemsets.values()),
        "itemsets": [key_to_itemset(key) for key in freqItemsets],
    })
    return association_rules(itemsetsDf, metric=metric, min_threshold=min_conf)


def build_fuzzy_meta_database(rule_measure_sets, attributes=None, normalize=False):
    """
    Construye la meta-base de datos difusa D~ (Tabla 8 del articulo): el
    grado de pertenencia de la regla r_i en el dataset D_j es el valor de la
    medida (soporte o factor de certeza) obtenida al minarla en D_j (0 si no
    fue extraida).

    Argumentos:
        rule_measure_sets (Sequence[dict[str, float]]): para cada dataset
            D_j, {rule_key: valor en [0, 1]} (ver `mine_primary_rule_measures`)
        attributes (Sequence[dict[str, float]], opcional): para cada
            dataset, grados de pertenencia difusos de atributos adicionales
        normalize (bool): si True, normaliza cada columna de regla al rango
            [0, 1] dividiendo por su maximo entre datasets (la variante
            D~_NS del articulo, Seccion 5.2.2, util cuando el soporte de las
            reglas primarias es muy bajo en todos los datasets)

    Retorna:
        pandas.DataFrame con una fila por dataset y una columna (grado de
        pertenencia en [0, 1]) por cada regla y atributo distintos
    """
    ruleKeys = sorted(set().union(*(set(rm) for rm in rule_measure_sets))) if rule_measure_sets else []

    rows = []
    for index, ruleMeasures in enumerate(rule_measure_sets):
        row = {ruleKey: float(ruleMeasures.get(ruleKey, 0.0)) for ruleKey in ruleKeys}
        if attributes:
            row.update(attributes[index])
        rows.append(row)

    metaDatabase = pd.DataFrame(rows)

    if normalize:
        for ruleKey in ruleKeys:
            maxValue = metaDatabase[ruleKey].max()
            if maxValue > 0:
                metaDatabase[ruleKey] = metaDatabase[ruleKey] / maxValue

    return metaDatabase


def fuzzy_meta_association_rules(rule_measure_sets, attributes=None, num_alpha=10,
                                 min_supp=0.1, min_conf=0.5, metric="confidence", normalize=False):
    """
    Algoritmo 4: mineria de meta-reglas de asociacion difusas.

    Argumentos:
        rule_measure_sets (Sequence[dict[str, float]]): ver
            `build_fuzzy_meta_database` (p.ej. la salida de
            `mine_primary_rule_measures`)
        attributes (Sequence[dict[str, float]], opcional): ver
            `build_fuzzy_meta_database`
        num_alpha (int): numero de alpha-cortes para la mineria difusa (el
            articulo de Fernandez-Basso et al. 2021 recomienda 10)
        min_supp (float): FSupp minimo para que una meta-regla se considere
            frecuente
        min_conf (float): umbral minimo de `metric` ('confidence' -> FConf,
            'certainty_factor' -> FCF) para una meta-regla
        metric (str): 'support', 'confidence' o 'certainty_factor'
        normalize (bool): ver `build_fuzzy_meta_database`

    Retorna:
        pandas.DataFrame de meta-reglas difusas (mismo formato que
        `FIM.FARE.fuzzy_association_rules`)
    """
    metaDatabase = build_fuzzy_meta_database(rule_measure_sets, attributes, normalize)
    fuzzyTransactions = [
        [(column, row[column]) for column in metaDatabase.columns]
        for _, row in metaDatabase.iterrows()
    ]

    freqItemsets = fuzzy_fpgrowth(fuzzyTransactions, min_supp, num_alpha)
    return fuzzy_association_rules(freqItemsets, num_alpha, metric=metric, min_threshold=min_conf)
