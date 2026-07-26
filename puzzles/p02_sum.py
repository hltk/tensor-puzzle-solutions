"""Puzzle 2 - sum

Compute sum, the sum of a vector.
https://numpy.org/doc/stable/reference/generated/numpy.sum.html
"""

from torchtyping import TensorType as TT

from puzzles.p01_ones import ones


def sum_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        out[0] += a[i]


def sum(a: TT["i"]) -> TT[1]:
    return a[None,:] @ ones(a.shape[0])
