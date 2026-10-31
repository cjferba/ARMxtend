# -*- coding: utf-8 -*-
"""
Significacion estadistica de reglas de asociacion: de "esta regla supera los
umbrales" a "esta regla es mas fuerte de lo que daria el azar".

Minar muchas reglas con umbrales de soporte y confianza produce reglas
espurias -- cuantas mas se prueban, mas aparecen por casualidad -- y sobre
datos temporales autocorrelacionados todavia mas. Este modulo ofrece tres
herramientas:

* TESTS por regla sobre la tabla 2x2 de sus frecuencias (``rule_significance``):
  test exacto de Fisher (cola de la hipergeometrica) o chi-cuadrado de Pearson.
  Suponen transacciones independientes.
* CORRECCION por pruebas multiples (``adjust_pvalues``, ``significant_rules``):
  Bonferroni, Holm (Holm, 1979), Benjamini-Hochberg (1995) y
  Benjamini-Yekutieli (2001). El numero de hipotesis es el de reglas
  candidatas que se han probado, no solo el de las que se muestran
  (``candidate_rule_count``; ``association_rules`` y ``temporal_association_rules``
  lo anotan en ``rules.attrs["n_candidates"]``).
* TESTS DE PERMUTACION (``permutation_null``, ``permutation_p_values``): se
  re-mina sobre datos barajados que conservan la frecuencia de cada item pero
  rompen las relaciones entre ellos, y cada regla se compara con la distribucion
  del MAXIMO de la medida entre todas las reglas del dato barajado
  (Westfall & Young, 1993), lo que controla la tasa de error por familia
  teniendo en cuenta que las reglas se seleccionaron por sus propios valores
  (Webb, 2007). Con series temporales se baraja con desplazamientos circulares
  (``null="shift"``), que conservan la autocorrelacion de cada serie.

Ademas, ``confidence_intervals`` da intervalos de Wilson o bootstrap para el
soporte, la confianza y el lift.

Los tests por regla exigen frecuencias enteras (reglas crisp); para reglas
difusas use los tests de permutacion, que valen con cualquier minero.

@author Carlos Fernandez-Basso
"""
import math
from itertools import combinations
from statistics import NormalDist
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


Miner = Callable[[pd.DataFrame], pd.DataFrame]

CORRECTIONS = ("bonferroni", "holm", "fdr_bh", "fdr_by")
TESTS = ("fisher", "chi2")
ALTERNATIVES = ("greater", "less", "two-sided")
NULLS = ("shift", "permute")

_CHUNK = 4096


# ---------------------------------------------------------------- tests 2x2

def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_upper(k: int, draws: int, successes: int, population: int) -> float:
    """
    P(X >= k) para X ~ Hipergeometrica(poblacion, exitos, extracciones): la cola
    superior del test exacto de Fisher. Se suma desde `k` hacia arriba con la
    recurrencia entre terminos consecutivos, en bloques, y se para en cuanto los
    terminos, ya decrecientes, dejan de contar: el coste depende de la desviacion
    tipica de X y no del tamano de la poblacion.
    """
    low = max(0, draws + successes - population)
    high = min(draws, successes)
    if k <= low:
        return 1.0
    if k > high:
        return 0.0

    def log_ratio(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """log de pmf(x + 1) / pmf(x)."""
        return (np.log((successes - x) * (draws - x))
                - np.log((x + 1.0) * (population - successes - draws + x + 1.0)))

    logPmf = float(_log_choose(successes, k) + _log_choose(population - successes, draws - k)
                   - _log_choose(population, draws))
    total = 0.0
    start = k
    while True:
        steps = min(_CHUNK, high - start + 1)
        logs = logPmf + np.concatenate([[0.0], np.cumsum(log_ratio(start + np.arange(steps - 1, dtype=float)))])
        terms = np.exp(logs)
        total += float(terms.sum())
        start += steps
        if start > high:
            break
        logPmf = float(logs[-1] + log_ratio(float(start - 1)))
        if logPmf < logs[-1] and terms[-1] < 1e-18 * total:
            break
    return min(total, 1.0)


def _fisher_pvalue(n_ac: int, n_a: int, n_c: int, n: int, alternative: str) -> float:
    greater = _hypergeom_upper(n_ac, n_a, n_c, n)
    if alternative == "greater":
        return greater
    # P(X <= k) = P(Y >= n_a - k) con Y = extracciones - X ~ Hipergeometrica(n, n - n_c, n_a)
    less = _hypergeom_upper(n_a - n_ac, n_a, n - n_c, n)
    if alternative == "less":
        return less
    return min(1.0, 2.0 * min(greater, less))


def _chi2_pvalue(n_ac: int, n_a: int, n_c: int, n: int, alternative: str) -> float:
    rowA, rowNotA = n_a, n - n_a
    colC, colNotC = n_c, n - n_c
    if min(rowA, rowNotA, colC, colNotC) == 0:
        return 1.0
    expected = n_a * n_c / n
    deviation = n_ac - expected
    statistic = float(n) ** 3 * deviation ** 2 / (float(rowA) * rowNotA * colC * colNotC)   # n (ad - bc)^2 / (R1 R2 C1 C2)
    two = math.erfc(math.sqrt(statistic / 2.0))
    if alternative == "two-sided":
        return two
    if deviation == 0:
        return 1.0
    towards = deviation > 0
    return two / 2.0 if (towards == (alternative == "greater")) else 1.0 - two / 2.0


def _rule_counts(rules: pd.DataFrame, n_transactions: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = n_transactions if n_transactions is not None else rules.attrs.get("n_transactions")
    if n is None:
        raise ValueError("falta el numero de transacciones: pasa `n_transactions` (las reglas de "
                         "temporal_association_rules ya lo traen en rules.attrs)")
    n = int(n)
    if n < 1:
        raise ValueError("n_transactions debe ser >= 1, recibido: {}".format(n))
    missing = [column for column in ("support", "antecedent support", "consequent support") if column not in rules.columns]
    if missing:
        raise ValueError("las reglas necesitan las columnas {}: este test es para reglas crisp "
                         "(para reglas difusas use permutation_null)".format(missing))
    scaled = [rules[column].to_numpy(dtype=float) * n for column in ("support", "antecedent support", "consequent support")]
    counts = [np.rint(values) for values in scaled]
    if any(np.any(np.abs(values - rounded) > 1e-3) for values, rounded in zip(scaled, counts)):
        raise ValueError("los soportes no son multiplos de 1/{}: n_transactions no coincide con las "
                         "transacciones de las que salen las reglas (o son reglas difusas)".format(n))
    nAC, nA, nC = (rounded.astype(np.int64) for rounded in counts)
    if np.any(nAC > np.minimum(nA, nC)) or np.any(nA > n) or np.any(nC > n) or np.any(nAC < np.maximum(0, nA + nC - n)):
        raise ValueError("las frecuencias de las reglas son incoherentes con n_transactions={}".format(n))
    return nAC, nA, nC, n


def rule_significance(rules: pd.DataFrame, n_transactions: Optional[int] = None, method: str = "fisher",
                      alternative: str = "greater") -> pd.DataFrame:
    """
    Anade a cada regla el p-valor de su asociacion.

    Para cada regla ``A => C`` se contrasta la hipotesis nula de que A y C son
    independientes con la tabla 2x2 (A / no A) x (C / no C) reconstruida a partir
    de los soportes y del numero de transacciones.

    Argumentos:
        rules (pandas.DataFrame): reglas crisp, con las columnas 'support',
            'antecedent support' y 'consequent support' (la salida de
            ``ARM.association_rules`` o de ``ARM.temporal_association_rules``)
        n_transactions (int, opcional): numero de transacciones de las que salen
            los soportes; por defecto ``rules.attrs["n_transactions"]`` (lo anotan
            las reglas temporales; en las demas hay que darlo). Si los soportes no
            son multiplos de 1/n se lanza ``ValueError``
        method (str): 'fisher' (test exacto, por defecto) o 'chi2' (Pearson, sin
            correccion de continuidad; aproxima bien con frecuencias esperadas >= 5)
        alternative (str): 'greater' (asociacion positiva: A favorece C, lo que
            interesa en una regla; por defecto), 'less' o 'two-sided' (el doble
            del menor p-valor unilateral)

    Retorna:
        Copia de `rules` con la columna 'p_value'. Supone transacciones
        independientes; con series temporales use ``permutation_null``.
    """
    if method not in TESTS:
        raise ValueError("method debe ser uno de {}, recibido: '{}'".format(TESTS, method))
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative debe ser uno de {}, recibido: '{}'".format(ALTERNATIVES, alternative))
    nAC, nA, nC, n = _rule_counts(rules, n_transactions)
    test = _fisher_pvalue if method == "fisher" else _chi2_pvalue
    result = rules.copy()
    result["p_value"] = [test(int(ac), int(a), int(c), n, alternative) for ac, a, c in zip(nAC, nA, nC)]
    result.attrs["n_transactions"] = n
    return result


# ------------------------------------------------------- pruebas multiples

def adjust_pvalues(p_values: Union[Iterable[float], np.ndarray], method: str = "fdr_bh",
                   n_tests: Optional[int] = None) -> np.ndarray:
    """
    Corrige p-valores por pruebas multiples.

    Argumentos:
        p_values: los p-valores de las hipotesis contrastadas
        method (str): 'bonferroni' y 'holm' controlan la probabilidad de algun falso
            positivo (FWER); 'fdr_bh' (Benjamini-Hochberg, para tests independientes o
            con dependencia positiva) y 'fdr_by' (Benjamini-Yekutieli, valida con
            cualquier dependencia) controlan la proporcion esperada de falsos descubrimientos (FDR)
        n_tests (int, opcional): numero total de hipotesis de la familia, si es mayor
            que ``len(p_values)``: las no listadas se tratan como no rechazadas
            (p-valor 1); por defecto, ``len(p_values)``

    Retorna:
        numpy.ndarray con los p-valores ajustados, en el orden de entrada y en [0, 1].
    """
    if method not in CORRECTIONS:
        raise ValueError("method debe ser uno de {}, recibido: '{}'".format(CORRECTIONS, method))
    p = np.asarray(list(p_values) if not isinstance(p_values, np.ndarray) else p_values, dtype=float)
    count = len(p)
    total = count if n_tests is None else int(n_tests)
    if total < count:
        raise ValueError("n_tests ({}) no puede ser menor que el numero de p-valores ({})".format(total, count))
    if count == 0:
        return p.copy()
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("los p-valores deben estar en [0, 1]")
    order = np.argsort(p, kind="mergesort")
    sortedP = p[order]
    ranks = np.arange(1, count + 1, dtype=float)
    if method == "bonferroni":
        adjusted = sortedP * total
    elif method == "holm":
        adjusted = np.maximum.accumulate((total - ranks + 1.0) * sortedP)
    else:
        factor = 1.0 if method == "fdr_bh" else float(np.sum(1.0 / np.arange(1, total + 1)))
        adjusted = np.minimum.accumulate((sortedP * total * factor / ranks)[::-1])[::-1]
    result = np.empty(count)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def candidate_rule_count(itemsets: pd.DataFrame) -> int:
    """
    Numero de reglas candidatas que salen de unos itemsets frecuentes: cada itemset
    de tamano k >= 2 se parte de ``2^k - 2`` formas en antecedente y consecuente.
    Es el tamano natural de la familia de hipotesis para `adjust_pvalues` (``n_tests``).

    Argumentos:
        itemsets (pandas.DataFrame): con la columna 'itemsets' (como la entrada de
            ``ARM.association_rules``)
    """
    return int(sum(2 ** len(items) - 2 for items in itemsets["itemsets"] if len(items) >= 2))


def significant_rules(rules: pd.DataFrame, alpha: float = 0.05, correction: str = "fdr_bh",
                      n_transactions: Optional[int] = None, method: str = "fisher", alternative: str = "greater",
                      n_tests: Optional[int] = None) -> pd.DataFrame:
    """
    Se queda con las reglas significativas tras corregir por pruebas multiples.

    Argumentos:
        rules (pandas.DataFrame): reglas crisp (ver `rule_significance`)
        alpha (float): nivel de significacion, en (0, 1): FWER o FDR segun `correction`
        correction (str): ver `adjust_pvalues`
        n_transactions, method, alternative: ver `rule_significance`
        n_tests (int, opcional): numero de hipotesis de la familia. Por defecto
            ``rules.attrs["n_candidates"]`` (el numero de reglas candidatas que anotan
            ``association_rules`` y ``temporal_association_rules``) o, si no esta, el
            numero de reglas dadas -- que SUBESTIMA la familia, porque las reglas se
            eligieron por superar umbrales; use `candidate_rule_count` o un test de permutacion

    Retorna:
        pandas.DataFrame con las reglas cuyo 'p_adjusted' <= alpha, con las columnas
        'p_value' y 'p_adjusted', ordenado por p-valor ajustado.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar en (0, 1), recibido: {}".format(alpha))
    family = n_tests if n_tests is not None else rules.attrs.get("n_candidates")
    scored = rule_significance(rules, n_transactions, method, alternative)
    family = max(int(family), len(scored)) if family is not None else len(scored)
    scored["p_adjusted"] = adjust_pvalues(scored["p_value"].to_numpy(), correction, n_tests=family)
    kept = scored[scored["p_adjusted"] <= alpha].sort_values(["p_adjusted", "p_value"], kind="mergesort")
    result = kept.reset_index(drop=True)
    result.attrs.update(scored.attrs)
    result.attrs["n_tests"] = family
    return result


def significant_miner(miner: Miner, alpha: float = 0.05, correction: str = "fdr_bh", method: str = "fisher",
                      alternative: str = "greater") -> Miner:
    """
    Envuelve un minero ``tabla -> reglas`` para que solo devuelva las reglas
    significativas (ver `significant_rules`). Sirve como ``miner=`` de
    ``ARM.window_rules`` y de `permutation_null`. El numero de transacciones es
    ``rules.attrs["n_transactions"]`` si el minero lo anota (las reglas temporales)
    y, si no, el numero de filas de la tabla.
    """
    def significant(table: pd.DataFrame) -> pd.DataFrame:
        rules = miner(table)
        if not len(rules):
            return rules
        n = rules.attrs.get("n_transactions", len(table))
        return significant_rules(rules, alpha, correction, n_transactions=n, method=method, alternative=alternative)
    return significant


# ---------------------------------------------------------- intervalos

def confidence_intervals(rules: pd.DataFrame, n_transactions: Optional[int] = None, level: float = 0.95,
                         method: str = "wilson", n_resamples: int = 1000, random_state: Optional[int] = 0) -> pd.DataFrame:
    """
    Intervalos de confianza para el soporte, la confianza y el lift de cada regla.

    Argumentos:
        rules (pandas.DataFrame): reglas crisp (ver `rule_significance`)
        n_transactions (int, opcional): numero de transacciones (ver `rule_significance`)
        level (float): nivel del intervalo, en (0, 1)
        method (str): 'wilson' -- intervalo de Wilson de una proporcion: el soporte sobre
            n transacciones y la confianza sobre las transacciones del antecedente; no da
            intervalo del lift (NaN) --; o 'bootstrap' -- percentiles de `n_resamples`
            remuestreos de las transacciones con reemplazo, que dan tambien el lift. Para
            una regla basta remuestrear sus cuatro celdas 2x2 (una multinomial), asi que
            no hace falta la tabla de transacciones
        n_resamples (int), random_state (int): solo para 'bootstrap'

    Retorna:
        Copia de `rules` con las columnas 'support_low', 'support_high', 'confidence_low',
        'confidence_high', 'lift_low' y 'lift_high'. Ambos metodos suponen transacciones
        independientes: con series autocorrelacionadas los intervalos quedan demasiado estrechos.
    """
    if not 0 < level < 1:
        raise ValueError("level debe estar en (0, 1), recibido: {}".format(level))
    if method not in ("wilson", "bootstrap"):
        raise ValueError("method debe ser 'wilson' o 'bootstrap', recibido: '{}'".format(method))
    nAC, nA, nC, n = _rule_counts(rules, n_transactions)
    result = rules.copy()
    columns = ["support_low", "support_high", "confidence_low", "confidence_high", "lift_low", "lift_high"]
    values = np.full((len(rules), 6), np.nan)
    if method == "wilson":
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        for row in range(len(rules)):
            values[row, 0:2] = _wilson(int(nAC[row]), n, z)
            if nA[row] > 0:
                values[row, 2:4] = _wilson(int(nAC[row]), int(nA[row]), z)
    else:
        rng = np.random.default_rng(random_state)
        low, high = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        for row in range(len(rules)):
            cells = np.array([nAC[row], nA[row] - nAC[row], nC[row] - nAC[row], n - nA[row] - nC[row] + nAC[row]], dtype=float)
            draws = rng.multinomial(n, cells / n, size=n_resamples).astype(float)
            ac, a, c = draws[:, 0], draws[:, 0] + draws[:, 1], draws[:, 0] + draws[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                confidence = np.where(a > 0, ac / a, np.nan)
                lift = np.where((a > 0) & (c > 0), confidence / (c / n), np.nan)
            for offset, sample in ((0, ac / n), (2, confidence), (4, lift)):
                if not np.all(np.isnan(sample)):
                    values[row, offset:offset + 2] = np.nanquantile(sample, [low, high])
    for index, column in enumerate(columns):
        result[column] = values[:, index]
    return result


def _wilson(successes: int, trials: int, z: float) -> Tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------- permutaciones

def shuffled_items(items: pd.DataFrame, null: str = "shift", random_state: Union[None, int, np.random.Generator] = None) -> pd.DataFrame:
    """
    Una version de `items` sin las relaciones entre columnas, para usarla como dato nulo.

    Argumentos:
        items (pandas.DataFrame): tabla de items (booleana o de grados), una fila por instante
        null (str): 'permute' -- cada columna se permuta al azar por separado: valida con
            transacciones independientes --; o 'shift' (por defecto) -- cada columna se
            desplaza circularmente un numero aleatorio de filas: conserva la autocorrelacion y
            los ciclos de cada serie y solo rompe su alineacion con las demas, que es lo que
            hay que hacer con series temporales
        random_state: semilla o ``numpy.random.Generator``
    """
    if null not in NULLS:
        raise ValueError("null debe ser uno de {}, recibido: '{}'".format(NULLS, null))
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    values = items.to_numpy()
    rows = len(values)
    shuffled = np.empty_like(values)
    for column in range(values.shape[1]):
        if null == "permute":
            shuffled[:, column] = values[rng.permutation(rows), column]
        else:
            shuffled[:, column] = np.roll(values[:, column], int(rng.integers(1, rows)) if rows > 1 else 0)
    return pd.DataFrame(shuffled, index=items.index, columns=items.columns)


def permutation_null(items: pd.DataFrame, miner: Miner, measure: str = "p_value", n_permutations: int = 100,
                     null: str = "shift", random_state: Optional[int] = 0, n_transactions: Optional[int] = None,
                     test: str = "fisher") -> pd.DataFrame:
    """
    Distribucion nula del mejor valor de una medida: re-mina `n_permutations` veces
    sobre datos barajados (`shuffled_items`) y guarda, de cada uno, el numero de reglas
    encontradas y la MEJOR medida entre ellas (el maximo, o el minimo si es un p-valor).

    Argumentos:
        items (pandas.DataFrame): la tabla de items de la que salieron las reglas
        miner (Callable): ``tabla -> DataFrame de reglas`` con los mismos parametros con
            que se minaron las reglas reales (p.ej. ``crisp_rule_miner(0.05)``, una
            funcion que llame a ``temporal_association_rules`` o a un minero difuso)
        measure (str): estadistico a seguir. 'p_value' (por defecto): el MENOR p-valor
            (`rule_significance`) entre las reglas del dato barajado -- el metodo minP de
            Westfall y Young, que tiene en cuenta que las reglas de poco soporte alcanzan
            medidas extremas por casualidad; solo para reglas crisp --. Cualquier otra
            columna de las reglas ('lift', 'confidence', 'zhangs_metric', ...): su MAXIMO;
            es la unica opcion con reglas difusas
        n_permutations (int): numero de datos barajados (el p-valor minimo posible es ``1/(n+1)``)
        null (str): 'shift' o 'permute' (ver `shuffled_items`)
        random_state (int): semilla
        n_transactions (int, opcional): con 'p_value', numero de transacciones si el minero
            no lo anota en ``rules.attrs`` (por defecto, el numero de filas)
        test (str): con 'p_value', 'fisher' o 'chi2'

    Retorna:
        pandas.DataFrame con una fila por permutacion: 'permutation', 'n_rules' y
        'min_p_value' o 'max_<measure>' (NaN si no salio ninguna regla).
    """
    if n_permutations < 1:
        raise ValueError("n_permutations debe ser >= 1")
    pvalues = measure == "p_value"
    column = "min_p_value" if pvalues else "max_" + measure
    rng = np.random.default_rng(random_state)
    rows = []
    for permutation in range(n_permutations):
        table = shuffled_items(items, null, rng)
        rules = miner(table)
        best = float("nan")
        if len(rules):
            if pvalues:
                n = rules.attrs.get("n_transactions", n_transactions if n_transactions is not None else len(table))
                best = float(rule_significance(rules, n, test)["p_value"].min())
            elif measure in rules.columns:
                best = float(rules[measure].max())
            else:
                raise ValueError("measure '{}' no es una columna de las reglas: {}".format(measure, sorted(rules.columns)))
        rows.append({"permutation": permutation, "n_rules": len(rules), column: best})
    result = pd.DataFrame(rows, columns=["permutation", "n_rules", column])
    result.attrs.update({"measure": measure, "test": test})
    return result


def permutation_p_values(rules: pd.DataFrame, null_distribution: pd.DataFrame,
                         n_transactions: Optional[int] = None) -> pd.DataFrame:
    """
    p-valor de permutacion de cada regla frente a la distribucion nula de la mejor medida
    (`permutation_null`): la fraccion de datos barajados cuya mejor regla es al menos tan
    buena como la suya, ``(1 + #{mejor_nulo mejor o igual}) / (1 + n_permutaciones)``. Como
    compara con la mejor de TODAS las reglas del dato barajado, ya esta ajustado por pruebas
    multiples (FWER) y por haber seleccionado las reglas de los propios datos.

    Argumentos:
        rules (pandas.DataFrame): las reglas reales, minadas con el mismo minero
        null_distribution (pandas.DataFrame): resultado de `permutation_null`
        n_transactions (int, opcional): con la medida 'p_value', numero de transacciones si
            las reglas no tienen aun la columna 'p_value' ni lo anotan en ``rules.attrs``

    Retorna:
        Copia de `rules` con la columna 'p_permutation' (y 'p_value' si se uso esa medida).
    """
    measure = null_distribution.attrs.get("measure")
    if measure is None:
        raise ValueError("`null_distribution` no viene de permutation_null (falta attrs['measure'])")
    result = rules.copy()
    if measure == "p_value":
        if "p_value" not in result.columns:
            result = rule_significance(result, n_transactions, null_distribution.attrs.get("test", "fisher"))
        null = null_distribution["min_p_value"].to_numpy(dtype=float)
        exceed = (null[None, :] <= result["p_value"].to_numpy(dtype=float)[:, None] * (1 + 1e-9) + 1e-300).sum(axis=1)
    else:
        if measure not in result.columns:
            raise ValueError("measure '{}' no es una columna de las reglas".format(measure))
        null = null_distribution["max_" + measure].to_numpy(dtype=float)
        exceed = (null[None, :] >= result[measure].to_numpy(dtype=float)[:, None] - 1e-12).sum(axis=1)
    result["p_permutation"] = (1.0 + exceed) / (1.0 + len(null))
    return result


def count_candidates(itemsets: Iterable[Iterable[str]], is_valid: Callable[[frozenset, frozenset], bool],
                     eligible_consequent: Callable[[str], bool]) -> int:
    """
    Cuenta las reglas candidatas ``A => C`` de unos itemsets que cumplen una condicion
    estructural: el consecuente solo usa items con ``eligible_consequent`` y el par
    (antecedente, consecuente) pasa ``is_valid``. Lo usa ``temporal_association_rules``
    para dar el tamano de su familia de hipotesis.
    """
    total = 0
    for itemset in itemsets:
        items = frozenset(itemset)
        usable = [item for item in sorted(items) if eligible_consequent(item)]
        for size in range(1, min(len(usable), len(items) - 1) + 1):
            for chosen in combinations(usable, size):
                consequent = frozenset(chosen)
                if is_valid(items - consequent, consequent):
                    total += 1
    return total


__all__ = [
    "rule_significance", "adjust_pvalues", "candidate_rule_count", "significant_rules", "significant_miner",
    "confidence_intervals", "shuffled_items", "permutation_null", "permutation_p_values",
]
