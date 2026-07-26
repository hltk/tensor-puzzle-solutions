"""Puzzle 8 - diff

Compute diff, the running difference.
https://numpy.org/doc/stable/reference/generated/numpy.diff.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]


def diff(a: TT["i"], i: int) -> TT["i"]:
    return a - (arange(i) > 0) * a[arange(i) - 1]
