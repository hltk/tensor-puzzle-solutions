"""Puzzle 24 - argsort (bonus)

Compute argsort, the indices that would sort a vector, ascending.
https://numpy.org/doc/stable/reference/generated/numpy.argsort.html

Equal values keep their original relative order.
"""

from torchtyping import TensorType as TT

from puzzles.base import arange
from puzzles.p06_triu import triu


def argsort_spec(a, out):
    order = sorted(range(len(a)), key=lambda i: (a[i], i))
    for k in range(len(out)):
        out[k] = order[k]


def argsort(a: TT["i"]) -> TT["i"]:
    x = a[:, None] > a
    y = (a[:, None] == a) * (1 - triu(a.shape[0]))

    return ((x + y) @ a ** 0 == arange(a.shape[0])[:, None])*1 @ arange(a.shape[0])
