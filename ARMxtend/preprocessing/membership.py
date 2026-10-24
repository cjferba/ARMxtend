# -*- coding: utf-8 -*-
"""
Funciones de pertenencia y particiones difusas de un atributo numerico:

* triangular (particion de Ruspini): `triangular_partition`
* trapezoidal: `trapezoidal_membership` / `trapezoidal_partition`
* gaussiana: `gaussian_membership` / `gaussian_partition`
* calculo AUTOMATICO de los puntos de la particion a partir de los datos,
  por reparto uniforme, por cuantiles o por k-means 1-D: `automatic_peaks`,
  `automatic_partition`

Todas las funciones son puras (numpy) y devuelven una matriz de forma
(n_valores, n_etiquetas) con grados de pertenencia en [0, 1]; son la entrada
de ``preprocessing.FuzzyLib`` y, a traves de el, de los algoritmos difusos de
ARMxtend (``FFIM.fuzzy_fpgrowth``, ``FIM.FARE``, ...).

@author Carlos Fernandez-Basso
"""
import numpy as np

METHODS = ("uniform", "quantile", "kmeans")
MEMBERSHIPS = ("triangular", "trapezoidal", "gaussian")

_LABELS = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["low", "medium_low", "medium_high", "high"],
    5: ["very_low", "low", "medium", "high", "very_high"],
}

# FWHM = 2 sqrt(2 ln 2) sigma: con sigma = separacion / FWHM_FACTOR, dos
# gaussianas vecinas se cruzan exactamente en grado 0.5 en el punto medio.
_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def default_labels(n_labels):
    """Etiquetas por defecto de una particion de `n_labels` conjuntos ('low', 'medium', 'high', ...)."""
    return list(_LABELS.get(n_labels, ["L{}".format(i + 1) for i in range(n_labels)]))


def triangular_partition(values, peaks):
    """
    Particion difusa triangular de Ruspini: dado un conjunto de picos
    ordenados ascendentemente, devuelve el grado de pertenencia de cada
    valor a cada uno de los `len(peaks)` conjuntos difusos definidos, de
    forma que los grados de todos los conjuntos suman siempre 1 para
    cualquier valor.

    Los dos conjuntos extremos son "hombros" (mantienen pertenencia total
    fuera de su ultimo tramo de pendiente); los conjuntos intermedios son
    triangulos con vertice en su pico correspondiente.

    Argumentos:
        values (array-like): valores numericos a fuzzificar
        peaks (Sequence[float]): picos, ordenados ascendentemente, de cada
            conjunto difuso (uno por etiqueta)

    Retorna:
        numpy.ndarray de forma (len(values), len(peaks))
    """
    values = np.asarray(values, dtype=float)
    peaks = np.asarray(sorted(peaks), dtype=float)
    n = len(peaks)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 picos/etiquetas para fuzzificar un atributo")
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("Los picos de la particion triangular deben ser estrictamente crecientes: {}".format(peaks))

    memberships = np.zeros((len(values), n))

    # Hombro izquierdo (primera etiqueta)
    memberships[:, 0] = np.clip((peaks[1] - values) / (peaks[1] - peaks[0]), 0.0, 1.0)
    memberships[values <= peaks[0], 0] = 1.0

    # Hombro derecho (ultima etiqueta)
    memberships[:, -1] = np.clip((values - peaks[-2]) / (peaks[-1] - peaks[-2]), 0.0, 1.0)
    memberships[values >= peaks[-1], -1] = 1.0

    # Etiquetas intermedias: triangulo con vertice en peaks[i]
    for i in range(1, n - 1):
        left = np.clip((values - peaks[i - 1]) / (peaks[i] - peaks[i - 1]), 0.0, 1.0)
        right = np.clip((peaks[i + 1] - values) / (peaks[i + 1] - peaks[i]), 0.0, 1.0)
        memberships[:, i] = np.minimum(left, right)

    return memberships


def trapezoidal_membership(values, a, b, c, d):
    """
    Funcion de pertenencia trapezoidal: 0 hasta `a`, sube linealmente hasta
    1 en `b`, vale 1 entre `b` y `c`, y baja linealmente hasta 0 en `d`.
    Admite lados verticales (a == b, c == d) y "hombros" infinitos
    (a = b = -inf: pertenencia 1 a la izquierda de `c`; c = d = +inf:
    pertenencia 1 a la derecha de `b`).

    Argumentos:
        values (array-like): valores a evaluar
        a, b, c, d (float): vertices del trapecio, con a <= b <= c <= d

    Retorna:
        numpy.ndarray de grados de pertenencia en [0, 1]
    """
    if not (a <= b <= c <= d):
        raise ValueError("Un trapecio requiere a <= b <= c <= d, recibido: {}".format((a, b, c, d)))
    values = np.asarray(values, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        rising = (values - a) / (b - a) if b > a else (values >= b).astype(float)
        falling = (d - values) / (d - c) if d > c else (values <= c).astype(float)
    return np.clip(np.minimum(rising, falling), 0.0, 1.0)


def trapezoidal_partition(values, params):
    """
    Grados de pertenencia a cada trapecio de `params`.

    Argumentos:
        values (array-like): valores a fuzzificar
        params (Sequence[Tuple[float, float, float, float]]): un
            (a, b, c, d) por etiqueta (ver `trapezoidal_membership`)

    Retorna:
        numpy.ndarray de forma (len(values), len(params))
    """
    return np.column_stack([trapezoidal_membership(values, *vertices) for vertices in params])


def gaussian_membership(values, center, sigma):
    """exp(-(x - center)^2 / (2 sigma^2)). Vale 1 en `center`; sigma > 0."""
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0, recibido: {}".format(sigma))
    values = np.asarray(values, dtype=float)
    return np.exp(-0.5 * ((values - center) / sigma) ** 2)


def gaussian_partition(values, params):
    """
    Grados de pertenencia a cada gaussiana de `params`.

    Argumentos:
        values (array-like): valores a fuzzificar
        params (Sequence[Tuple[float, float]]): un (centro, sigma) por etiqueta

    Retorna:
        numpy.ndarray de forma (len(values), len(params))
    """
    return np.column_stack([gaussian_membership(values, center, sigma) for center, sigma in params])


def _kmeans_1d(values, initial_centers, max_iter=100):
    centers = np.asarray(initial_centers, dtype=float).copy()
    for _ in range(max_iter):
        assignment = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = np.array([values[assignment == k].mean() if np.any(assignment == k) else centers[k]
                            for k in range(len(centers))])
        if np.allclose(updated, centers):
            break
        centers = updated
    return np.sort(centers)


def automatic_peaks(values, n_labels=3, method="quantile"):
    """
    Calcula `n_labels` picos para una particion de los datos.

    Argumentos:
        values (array-like): valores del atributo (los NaN se ignoran)
        n_labels (int): numero de conjuntos difusos, >= 2
        method (str):
            'uniform'  -> equiespaciados entre el minimo y el maximo
            'quantile' -> cuantiles equiespaciados (min, ..., max): cada
                          conjunto cubre aproximadamente el mismo numero
                          de datos
            'kmeans'   -> centros de k-means 1-D, inicializado en los
                          cuantiles (determinista)

    Retorna:
        numpy.ndarray de `n_labels` picos estrictamente crecientes

    Lanza ValueError si los datos no tienen suficientes valores distintos
    para `n_labels` picos distintos.
    """
    if method not in METHODS:
        raise ValueError("method debe ser uno de {}, recibido: '{}'".format(METHODS, method))
    if n_labels < 2:
        raise ValueError("n_labels debe ser >= 2, recibido: {}".format(n_labels))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("No hay valores con los que calcular la particion")

    if method == "uniform":
        peaks = np.linspace(values.min(), values.max(), n_labels)
    else:
        peaks = np.quantile(values, np.linspace(0.0, 1.0, n_labels))
        if method == "kmeans":
            peaks = _kmeans_1d(values, peaks)

    if np.any(np.diff(peaks) <= 0):
        raise ValueError(
            "Los datos no tienen suficiente variabilidad para {} etiquetas con el metodo '{}' "
            "(picos repetidos: {}); pruebe con 'uniform' o menos etiquetas".format(n_labels, method, peaks))
    return peaks


def automatic_partition(values, n_labels=3, method="quantile", membership="triangular"):
    """
    Particion difusa calculada automaticamente a partir de los datos.

    Argumentos:
        values (array-like): valores del atributo
        n_labels, method: ver `automatic_peaks`
        membership (str):
            'triangular'  -> particion de Ruspini con vertices en los picos
            'trapezoidal' -> particion de Ruspini trapezoidal: en cada
                             hueco entre dos picos consecutivos, la
                             transicion ocupa su mitad central y cada
                             conjunto tiene un nucleo (pertenencia 1) de
                             un cuarto del hueco a cada lado de su pico
            'gaussian'    -> una gaussiana por pico, con sigma tal que dos
                             vecinas se cruzan en 0.5 (no suma 1)

    Retorna:
        (memberships, params): la matriz (len(values), n_labels) de grados y
        los parametros de cada conjunto ('triangular': los picos;
        'trapezoidal': tuplas (a, b, c, d); 'gaussian': tuplas
        (centro, sigma)), reutilizables con `FuzzyLib.Fuzzification`.
    """
    if membership not in MEMBERSHIPS:
        raise ValueError("membership debe ser uno de {}, recibido: '{}'".format(MEMBERSHIPS, membership))
    peaks = automatic_peaks(values, n_labels, method)
    gaps = np.diff(peaks)

    if membership == "triangular":
        return triangular_partition(values, peaks), [float(p) for p in peaks]

    if membership == "gaussian":
        # separacion local: media de los huecos vecinos
        spacing = np.array([gaps[0]] + [(gaps[i - 1] + gaps[i]) / 2.0 for i in range(1, n_labels - 1)] + [gaps[-1]])
        params = [(float(c), float(s / _FWHM_FACTOR)) for c, s in zip(peaks, spacing)]
        return gaussian_partition(values, params), params

    params = []
    for i in range(n_labels):
        a = -np.inf if i == 0 else peaks[i - 1] + gaps[i - 1] / 4.0
        b = -np.inf if i == 0 else peaks[i] - gaps[i - 1] / 4.0
        c = np.inf if i == n_labels - 1 else peaks[i] + gaps[i] / 4.0
        d = np.inf if i == n_labels - 1 else peaks[i + 1] - gaps[i] / 4.0
        params.append((float(a), float(b), float(c), float(d)))
    return trapezoidal_partition(values, params), params
