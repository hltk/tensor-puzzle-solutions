"""Puzzle 15 - bincount

Compute bincount, count number of times an entry was seen.
https://numpy.org/doc/stable/reference/generated/numpy.bincount.html

Every entry of a is less than j.
"""

from torchtyping import TensorType as TT

from puzzles.base import arange
from puzzles.p01_ones import ones


def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1


def bincount(a: TT["i"], j: int) -> TT["j"]:
    return (a == arange(j)[:, None])*1 @ ones(a.shape[0])
