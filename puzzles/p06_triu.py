"""Puzzle 6 - triu

Compute triu, the upper triangular matrix.
https://numpy.org/doc/stable/reference/generated/numpy.triu.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0


def triu(j: int) -> TT["j", "j"]:
    return 1*( arange(j) >= arange(j)[:, None] )
