"""Puzzle 4 - diag

Compute diag, the diagonal vector of a square matrix.
https://numpy.org/doc/stable/reference/generated/numpy.diag.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]


def diag(a: TT["i", "i"]) -> TT["i"]:
    return a[arange(a.shape[0]), arange(a.shape[0])]
