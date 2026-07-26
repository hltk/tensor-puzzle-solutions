"""Puzzle 5 - eye

Compute eye, the identity matrix.
https://numpy.org/doc/stable/reference/generated/numpy.eye.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1


def eye(j: int) -> TT["j", "j"]:
    return 1*( arange(j) == arange(j)[:, None] )
