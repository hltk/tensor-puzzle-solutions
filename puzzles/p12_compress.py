"""Puzzle 12 - compress

Compute compress, keep only masked entries (left-aligned).
https://numpy.org/doc/stable/reference/generated/numpy.compress.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange
from puzzles.p07_cumsum import cumsum


def compress_spec(g, v, out):
    j = 0
    for i in range(len(g)):
        if g[i]:
            out[j] = v[i]
            j += 1


def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    return ( (cumsum(g * 1) * g) -1 == arange(i)[:, None] )*1 @ v
