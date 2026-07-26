"""Puzzle 13 - pad_to

Compute pad_to, eliminate or add 0s to change size of vector.
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]


def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    return  ( arange(i) == arange(j)[:, None] )*1 @ a
