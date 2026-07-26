"""Puzzle 10 - roll

Compute roll, the vector shifted 1 circular position.
https://numpy.org/doc/stable/reference/generated/numpy.roll.html
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]


def roll(a: TT["i"], i: int) -> TT["i"]:
    return a[(arange(i) + 1) % i]
