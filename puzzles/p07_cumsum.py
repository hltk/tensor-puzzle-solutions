"""Puzzle 7 - cumsum

Compute cumsum, the cumulative sum.
https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
"""

from torchtyping import TensorType as TT

from puzzles.p06_triu import triu


def cumsum_spec(a, out):
    total = 0
    for i in range(len(out)):
        out[i] = total + a[i]
        total += a[i]


def cumsum(a: TT["i"]) -> TT["i"]:
    return a @ triu(a.shape[0])
