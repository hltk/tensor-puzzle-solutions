"""Puzzle 3 - outer

Compute outer, the outer product of two vectors.
https://numpy.org/doc/stable/reference/generated/numpy.outer.html
"""

from torchtyping import TensorType as TT


def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]


def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    return a[:,None] @ b[None, :]
