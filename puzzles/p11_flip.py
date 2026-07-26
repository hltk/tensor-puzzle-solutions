"""Puzzle 11 - flip

Compute flip, the reversed vector.
https://numpy.org/doc/stable/reference/generated/numpy.flip.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]


def flip(a: TT["i"], i: int) -> TT["i"]:
    return a[i - arange(i) - 1]
