"""Puzzle 17 - flatten

Compute flatten.
https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def flatten_spec(a, out):
    k = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            out[k] = a[i][j]
            k += 1


def flatten(a: TT["i", "j"], i:int, j:int) -> TT["i * j"]:
    return a[arange(i * j) // j, arange(i * j) % j]
