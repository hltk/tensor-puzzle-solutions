"""Puzzle 1 - ones

Compute ones, the vector of all ones.
https://numpy.org/doc/stable/reference/generated/numpy.ones.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1


def ones(i: int) -> TT["i"]:
    return arange(i) ** 0
