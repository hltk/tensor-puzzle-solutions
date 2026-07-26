"""Puzzle 9 - vstack

Compute vstack, the matrix of two vectors.
https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange, where


def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]


def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    return where(arange(2)[:, None] == 0, a, b)
