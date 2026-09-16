"""Random-key encoding and decoding between continuous space [0, 1]^n and customer permutations."""

from __future__ import annotations
from typing import List, Sequence, Union
import numpy as np


def random_keys_to_permutation(keys: Union[Sequence[float], np.ndarray]) -> List[int]:
    """
    Convert random key vector in [0, 1]^n into a giant tour permutation of customer IDs [1..n].
    Uses stable argsort.
    
    Example:
    keys = [0.72, 0.15, 0.91, 0.32, 0.48]
    indices: [1, 3, 4, 0, 2] -> customer IDs: [2, 4, 5, 1, 3]
    """
    arr = np.asarray(keys, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Keys must be a 1D array, got shape {arr.shape}.")
    if len(arr) == 0:
        return []
    
    sorted_indices = np.argsort(arr, kind="stable")
    # Customer IDs are 1-indexed (1..n)
    return [int(idx + 1) for idx in sorted_indices]


def permutation_to_random_keys(permutation: Sequence[int]) -> np.ndarray:
    """
    Convert a customer permutation of [1..n] back into a normalized random-key vector in [0, 1]^n.
    For customer at position `pos` (0 <= pos < n):
        key = pos / (n - 1)  (or 0.5 if n == 1)
        
    The key is placed at index (customer_id - 1) so that argsort(keys) + 1 reproduces the permutation.
    """
    n = len(permutation)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        c = permutation[0]
        keys = np.zeros(1, dtype=np.float64)
        keys[0] = 0.5
        return keys

    keys = np.zeros(n, dtype=np.float64)
    for pos, customer_id in enumerate(permutation):
        idx = customer_id - 1
        if 0 <= idx < n:
            keys[idx] = float(pos) / float(n - 1)
        else:
            raise ValueError(f"Customer id {customer_id} out of bounds for permutation of length {n}.")

    return keys
