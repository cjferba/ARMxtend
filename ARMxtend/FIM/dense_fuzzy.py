# -*- coding: utf-8 -*-
"""
DENSE fuzzy frequent-itemset miner (single machine). Same result as ``FFIM.fuzzy_fpgrowth`` -- the bit-list of
relative support per alpha-cut of every itemset whose aggregated support FSupp reaches ``min_supp`` -- but computed
with a level-wise Apriori over packed bit vectors, one per item and alpha-cut, instead of running FP-Growth once per
cut and re-scanning the transactions for every candidate.

FSupp (a weighted mean of the supports per level, see ``FIM._shared.weighted_alpha_aggregate``) is anti-monotone
because the supports per level are, so the usual Apriori pruning is valid on it directly.

@author Carlos Fernandez-Basso
"""
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from ._shared import alpha_levels, alpha_weights, itemset_to_key
from ._vertical import _POPCOUNT
from .dense import _CHUNK_BYTES, _join, _prune


def dense_fuzzy_itemsets(transactions: Iterable[Iterable[Tuple[Any, float]]], min_supp: float,
                         num_alpha: int) -> Dict[str, np.ndarray]:
    """
    Fuzzy frequent itemsets (FSupp >= `min_supp`) by Apriori over bit vectors.

    Args:
        transactions: one transaction per element, each a list of (item, membership degree in [0, 1]) pairs
        min_supp (float): minimum relative aggregated support FSupp, in (0, 1]
        num_alpha (int): number of alpha-cuts

    Returns:
        dict {itemset_key: numpy.ndarray(num_alpha)} with the bit-list of relative support of each frequent itemset
        in every alpha-cut, as ``FFIM.fuzzy_fpgrowth`` returns it.
    """
    if not 0 < min_supp <= 1:
        raise ValueError("min_supp must be in (0, 1]")
    levels = alpha_levels(num_alpha)
    weights = alpha_weights(levels)
    baskets = [{str(item): float(degree) for item, degree in transaction} for transaction in transactions]
    total = len(baskets)
    items = sorted({item for basket in baskets for item in basket})
    if total == 0 or not items:
        return {}
    position = {item: k for k, item in enumerate(items)}
    rows = np.fromiter((r for r, basket in enumerate(baskets) for _ in basket), dtype=np.intp)
    cols = np.fromiter((position[item] for basket in baskets for item in basket), dtype=np.intp)
    degrees = np.fromiter((degree for basket in baskets for degree in basket.values()), dtype=float)
    if len(degrees) and (degrees.min() < 0.0 or degrees.max() > 1.0):
        raise ValueError("The fuzzy membership degree must be in [0, 1]")
    matrix = np.zeros((len(items), total), dtype=float)
    matrix[cols, rows] = degrees
    packed = np.packbits(matrix[:, None, :] >= levels[None, :, None], axis=2)      # (items, levels, words)
    minimum = min_supp - 1e-12

    def support_vectors(masks: np.ndarray) -> np.ndarray:
        return _POPCOUNT[masks].sum(axis=2, dtype=np.int64) / float(total)           # (itemsets, levels)

    singles = support_vectors(packed)
    keep = np.flatnonzero(singles @ weights >= minimum)
    result: Dict[str, np.ndarray] = {items[k]: singles[k] for k in keep}
    masks = packed[keep]
    level = keep.astype(np.intp).reshape(-1, 1)
    size = 1
    while len(level) > 1:
        first, second, candidates = _join(level)
        first, second, candidates = _prune(level, first, second, candidates, size)
        if not len(candidates):
            break
        width = masks.shape[1] * masks.shape[2]
        step = max(1, _CHUNK_BYTES // max(1, width))
        new_masks = np.empty((len(candidates),) + masks.shape[1:], dtype=np.uint8)
        vectors = np.empty((len(candidates), len(levels)), dtype=float)
        for begin in range(0, len(candidates), step):
            block = slice(begin, begin + step)
            new_masks[block] = masks[first[block]] & masks[second[block]]
            vectors[block] = support_vectors(new_masks[block])
        frequent = np.flatnonzero(vectors @ weights >= minimum)
        if not len(frequent):
            break
        level, masks = candidates[frequent], new_masks[frequent]
        size += 1
        for k in frequent:
            result[itemset_to_key(items[i] for i in candidates[k])] = vectors[k]
    return result
