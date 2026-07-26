"""Puzzle 18 - linspace

Compute linspace.
https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def linspace_spec(i, j, out):
    for k in range(len(out)):
        out[k] = float(i + (j - i) * k / max(1, len(out) - 1))


def linspace(i: TT[1], j: TT[1], n: int) -> TT["n", float]:
    return arange(n) * (j - i) / max(1, n - 1) + i
