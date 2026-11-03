# -*- coding: utf-8 -*-
"""
Significacion estadistica de reglas DIFUSAS, regla a regla, con un test de permutacion.

Los tests de ``ARM.significance`` (Fisher, chi-cuadrado) necesitan frecuencias enteras y no valen con grados de
pertenencia. Aqui, para cada regla ``A => B``:

* se calcula el vector de pertenencia del antecedente ``a`` y del consecuente ``b`` en cada transaccion (el minimo, la
  t-norma, de los grados de sus items; un item ausente vale 0);
* el estadistico es la **confianza por cardinal escalar** ``sum(min(a, b)) / sum(a)`` -- no la confianza agregada por
  alpha-cortes (``FConf``) de ``FIM.FARE``, pero la misma idea y la misma ordenacion en la practica;
* la hipotesis nula es que ``b`` no depende de ``a``: se baraja ``b`` entre las transacciones (conserva la distribucion de
  cada lado y rompe su relacion) y el p-valor es la fraccion de barajados cuyo estadistico alcanza el observado,
  ``(1 + #barajados >= observado) / (1 + n_permutaciones)`` (Phipson y Smyth, 2010), contraste unilateral.

El p-valor se corrige por pruebas multiples con ``adjust_pvalues``. Con ``n_permutaciones`` barajados el p-valor minimo es
``1 / (n_permutaciones + 1)``: para que una correccion de Bonferroni con ``m`` pruebas pueda rechazar a nivel ``alpha``
hacen falta al menos ``m / alpha`` permutaciones.

@author Carlos Fernandez-Basso
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .significance import adjust_pvalues

FuzzyTransactions = Union[pd.DataFrame, Sequence[Iterable[Tuple[str, float]]]]

_TOL = 1e-12


def _degrees(transactions: FuzzyTransactions, items: Iterable[str]) -> Dict[str, np.ndarray]:
    """Vector de grados (uno por transaccion, 0 si falta) de cada item de `items`; el mayor grado si se repite."""
    wanted = set(items)
    if isinstance(transactions, pd.DataFrame):
        values = transactions.to_numpy(dtype=float)
        position = {str(name): k for k, name in enumerate(transactions.columns)}
        missing = wanted - set(position)
        columns = {item: values[:, position[item]].copy() for item in wanted if item in position}
    else:
        rows = list(transactions)
        columns = {item: np.zeros(len(rows)) for item in wanted}
        missing = set()
        for row, transaction in enumerate(rows):
            for item, degree in transaction:
                if item in columns:
                    columns[item][row] = max(columns[item][row], float(degree))
    if missing:
        raise ValueError("los items {} de las reglas no estan en las transacciones".format(sorted(missing)))
    for item, column in columns.items():
        if np.any((column < 0) | (column > 1)) or np.any(np.isnan(column)):
            raise ValueError("los grados de pertenencia deben estar en [0, 1] (item '{}')".format(item))
    return columns


def _side(columns: Dict[str, np.ndarray], items: Iterable[str], rows: int) -> np.ndarray:
    result = np.ones(rows)
    for item in items:
        result = np.minimum(result, columns[item])
    return result


def fuzzy_rule_significance(rules: pd.DataFrame, transactions: FuzzyTransactions, n_permutations: int = 999,
                            random_state: Union[None, int, np.random.Generator] = None) -> pd.DataFrame:
    """
    Anade a las reglas difusas el estadistico ``fuzzy_confidence`` y el ``p_value`` del test de permutacion.

    Argumentos:
        rules (pandas.DataFrame): reglas con 'antecedents' y 'consequents' (como las de ``FIM.FARE.fuzzy_association_rules``)
        transactions: las transacciones con las que se minaron, como lista de listas de pares (item, grado) o como tabla
            de grados (una fila por transaccion, una columna por item; ver ``preprocessing.timeseries.to_fuzzy_transactions``)
        n_permutations (int): numero de barajados (por defecto 999; mas da p-valores mas finos y mas lentos)
        random_state: semilla o ``numpy.random.Generator``

    Retorna:
        Copia de `rules` con las columnas 'fuzzy_confidence' y 'p_value'; ``attrs`` conserva las de `rules` y anade
        'n_permutations' y 'n_transactions'.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations debe ser >= 1")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    result = rules.copy()
    result.attrs.update(rules.attrs)
    rows = len(transactions)
    result.attrs["n_permutations"] = n_permutations
    result.attrs["n_transactions"] = rows
    if not len(rules):
        result["fuzzy_confidence"] = pd.Series(dtype=float)
        result["p_value"] = pd.Series(dtype=float)
        return result
    if rows == 0:
        raise ValueError("no hay transacciones")
    columns = _degrees(transactions, [item for sides in zip(rules["antecedents"], rules["consequents"])
                                      for side in sides for item in side])
    permutations = np.argsort(rng.random((n_permutations, rows)), axis=1)
    confidences, p_values = [], []
    for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
        a, b = _side(columns, antecedent, rows), _side(columns, consequent, rows)
        total = a.sum()
        if total <= 0:
            confidences.append(0.0)
            p_values.append(1.0)
            continue
        observed = np.minimum(a, b).sum()
        shuffled = np.minimum(a[None, :], b[permutations]).sum(axis=1)
        confidences.append(observed / total)
        p_values.append((1 + int((shuffled >= observed - _TOL).sum())) / (n_permutations + 1))
    result["fuzzy_confidence"] = confidences
    result["p_value"] = p_values
    return result


def fuzzy_significant_rules(rules: pd.DataFrame, transactions: FuzzyTransactions, alpha: float = 0.05,
                            correction: str = "fdr_bh", n_permutations: int = 999, n_tests: Optional[int] = None,
                            random_state: Union[None, int, np.random.Generator] = None) -> pd.DataFrame:
    """
    Las reglas difusas significativas tras corregir por pruebas multiples (ver `fuzzy_rule_significance`).

    `n_tests` es el numero de hipotesis de la familia; por defecto ``rules.attrs['n_candidates']`` (que anota
    ``FIM.FARE.fuzzy_association_rules``: las particiones antecedente/consecuente examinadas) o, si no esta, el numero de
    reglas dadas, que SUBESTIMA la familia. Devuelve las reglas con 'p_adjusted' <= alpha, ordenadas por p-valor ajustado.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar en (0, 1), recibido: {}".format(alpha))
    scored = fuzzy_rule_significance(rules, transactions, n_permutations, random_state)
    family = n_tests if n_tests is not None else rules.attrs.get("n_candidates")
    family = max(int(family), len(scored)) if family is not None else len(scored)
    scored["p_adjusted"] = adjust_pvalues(scored["p_value"].to_numpy(), correction, n_tests=family)
    kept = scored[scored["p_adjusted"] <= alpha].sort_values(["p_adjusted", "p_value"], kind="mergesort")
    result = kept.reset_index(drop=True)
    result.attrs.update(scored.attrs)
    result.attrs["n_tests"] = family
    return result
