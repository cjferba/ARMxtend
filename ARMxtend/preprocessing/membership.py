# -*- coding: utf-8 -*-
"""
Membership functions and fuzzy partitions of a numeric attribute:

* triangular (Ruspini partition): `triangular_partition`
* trapezoidal: `trapezoidal_membership` / `trapezoidal_partition`
* Gaussian: `gaussian_membership` / `gaussian_partition`
* AUTOMATIC computation of the points of the partition from the data, by uniform spacing, by quantiles or by 1-D
  k-means: `automatic_peaks`, `automatic_partition`

All the functions are pure (numpy) and return a matrix of shape (n_values, n_labels) with membership degrees in
[0, 1]; they are the input of ``preprocessing.FuzzyLib`` and, through it, of the fuzzy algorithms of ARMxtend
(``FFIM.fuzzy_fpgrowth``, ``FIM.FARE``, ...).

@author Carlos Fernandez-Basso
"""
from typing import List, Sequence, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike

METHODS = ("uniform", "quantile", "kmeans")
MEMBERSHIPS = ("triangular", "trapezoidal", "gaussian")

_LABELS = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["low", "medium_low", "medium_high", "high"],
    5: ["very_low", "low", "medium", "high", "very_high"],
}

# FWHM = 2 sqrt(2 ln 2) sigma: with sigma = spacing / FWHM_FACTOR, two
# neighboring Gaussians cross exactly at degree 0.5 at the midpoint.
_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def default_labels(n_labels: int) -> List[str]:
    """Default labels of a partition of `n_labels` sets ('low', 'medium', 'high', ...)."""
    return list(_LABELS.get(n_labels, ["L{}".format(i + 1) for i in range(n_labels)]))


def triangular_partition(values: ArrayLike, peaks: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Ruspini triangular fuzzy partition: given a set of peaks sorted in ascending order, it returns the membership
    degree of each value in each of the `len(peaks)` defined fuzzy sets, so that the degrees of all the sets always
    add up to 1 for any value.

    The two extreme sets are "shoulders" (they keep full membership outside their last sloping section); the
    intermediate sets are triangles with their vertex at the corresponding peak.

    Args:
        values (array-like): numeric values to fuzzify
        peaks (Sequence[float]): peaks, sorted in ascending order, of each fuzzy set (one per label)

    Returns:
        numpy.ndarray of shape (len(values), len(peaks))
    """
    values = np.asarray(values, dtype=float)
    peaks = np.asarray(sorted(peaks), dtype=float)
    n = len(peaks)
    if n < 2:
        raise ValueError("At least 2 peaks/labels are needed to fuzzify an attribute")
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("The peaks of the triangular partition must be strictly increasing: {}".format(peaks))

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


def trapezoidal_membership(values: ArrayLike, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Trapezoidal membership function: 0 up to `a`, rises linearly to 1 at `b`, is 1 between `b` and `c`, and falls
    linearly to 0 at `d`. It admits vertical sides (a == b, c == d) and infinite "shoulders" (a = b = -inf: membership 1
    to the left of `c`; c = d = +inf: membership 1 to the right of `b`).

    Args:
        values (array-like): values to evaluate
        a, b, c, d (float): vertices of the trapezoid, with a <= b <= c <= d

    Returns:
        numpy.ndarray of membership degrees in [0, 1]
    """
    if not (a <= b <= c <= d):
        raise ValueError("A trapezoid requires a <= b <= c <= d, got: {}".format((a, b, c, d)))
    values = np.asarray(values, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        rising = (values - a) / (b - a) if b > a else (values >= b).astype(float)
        falling = (d - values) / (d - c) if d > c else (values <= c).astype(float)
    return np.clip(np.minimum(rising, falling), 0.0, 1.0)


def trapezoidal_partition(values: ArrayLike, params: Sequence[Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Membership degrees in each trapezoid of `params`.

    Args:
        values (array-like): values to fuzzify
        params (Sequence[Tuple[float, float, float, float]]): one (a, b, c, d) per label (see
            `trapezoidal_membership`)

    Returns:
        numpy.ndarray of shape (len(values), len(params))
    """
    return np.column_stack([trapezoidal_membership(values, *vertices) for vertices in params])


def gaussian_membership(values: ArrayLike, center: float, sigma: float) -> np.ndarray:
    """exp(-(x - center)^2 / (2 sigma^2)). Vale 1 en `center`; sigma > 0."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0, got: {}".format(sigma))
    values = np.asarray(values, dtype=float)
    return np.exp(-0.5 * ((values - center) / sigma) ** 2)


def gaussian_partition(values: ArrayLike, params: Sequence[Tuple[float, float]]) -> np.ndarray:
    """
    Membership degrees in each Gaussian of `params`.

    Args:
        values (array-like): values to fuzzify
        params (Sequence[Tuple[float, float]]): one (center, sigma) per label

    Returns:
        numpy.ndarray of shape (len(values), len(params))
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


def automatic_peaks(values: ArrayLike, n_labels: int = 3, method: str = "quantile") -> np.ndarray:
    """
    Compute `n_labels` peaks for a partition of the data.

    Args:
        values (array-like): values of the attribute (NaNs are ignored)
        n_labels (int): number of fuzzy sets, >= 2
        method (str):
            'uniform'  -> equally spaced between the minimum and the maximum
            'quantile' -> equally spaced quantiles (min, ..., max): each set covers roughly the same number of data
            'kmeans'   -> centers of 1-D k-means, initialized at the quantiles (deterministic)

    Returns:
        numpy.ndarray of `n_labels` strictly increasing peaks

    Raises ValueError if the data do not have enough distinct values for `n_labels` distinct peaks.
    """
    if method not in METHODS:
        raise ValueError("method must be one of {}, got: '{}'".format(METHODS, method))
    if n_labels < 2:
        raise ValueError("n_labels must be >= 2, got: {}".format(n_labels))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("There are no values to compute the partition with")

    if method == "uniform":
        peaks = np.linspace(values.min(), values.max(), n_labels)
    else:
        peaks = np.quantile(values, np.linspace(0.0, 1.0, n_labels))
        if method == "kmeans":
            peaks = _kmeans_1d(values, peaks)

    if np.any(np.diff(peaks) <= 0):
        raise ValueError(
            "The data do not have enough variability for {} labels with method '{}' "
            "(repeated peaks: {}); try 'uniform' or fewer labels".format(n_labels, method, peaks))
    return peaks


def automatic_partition(values: ArrayLike, n_labels: int = 3, method: str = "quantile",
                        membership: str = "triangular") -> Tuple[np.ndarray, list]:
    """
    Fuzzy partition computed automatically from the data.

    Args:
        values (array-like): values of the attribute
        n_labels, method: see `automatic_peaks`
        membership (str):
            'triangular'  -> Ruspini partition with vertices at the peaks
            'trapezoidal' -> trapezoidal Ruspini partition: in each gap between two consecutive peaks, the transition
                             occupies its central half and each set has a core (membership 1) of a quarter of the gap
                             on each side of its peak
            'gaussian'    -> one Gaussian per peak, with sigma such that two neighbors cross at 0.5 (does not add up
                             to 1)

    Returns:
        (memberships, params): the (len(values), n_labels) matrix of degrees and the parameters of each set
        ('triangular': the peaks; 'trapezoidal': tuples (a, b, c, d); 'gaussian': tuples (center, sigma)), reusable
        with `FuzzyLib.Fuzzification`.
    """
    if membership not in MEMBERSHIPS:
        raise ValueError("membership must be one of {}, got: '{}'".format(MEMBERSHIPS, membership))
    peaks = automatic_peaks(values, n_labels, method)
    gaps = np.diff(peaks)

    if membership == "triangular":
        return triangular_partition(values, peaks), [float(p) for p in peaks]

    if membership == "gaussian":
        # local spacing: mean of the neighboring gaps
        spacing = np.array([gaps[0]] + [(gaps[i - 1] + gaps[i]) / 2.0 for i in range(1, n_labels - 1)] + [gaps[-1]])
        gaussians = [(float(c), float(s / _FWHM_FACTOR)) for c, s in zip(peaks, spacing)]
        return gaussian_partition(values, gaussians), gaussians

    trapezoids = []
    for i in range(n_labels):
        a = -np.inf if i == 0 else peaks[i - 1] + gaps[i - 1] / 4.0
        b = -np.inf if i == 0 else peaks[i] - gaps[i - 1] / 4.0
        c = np.inf if i == n_labels - 1 else peaks[i] + gaps[i] / 4.0
        d = np.inf if i == n_labels - 1 else peaks[i + 1] - gaps[i] / 4.0
        trapezoids.append((float(a), float(b), float(c), float(d)))
    return trapezoidal_partition(values, trapezoids), trapezoids
