"""Puzzle 19 - heaviside

Compute heaviside.
https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
"""

from torchtyping import TensorType as TT


def heaviside_spec(a, b, out):
    for k in range(len(out)):
        if a[k] == 0:
            out[k] = b[k]
        else:
            out[k] = int(a[k] > 0)


def heaviside(a: TT["i"], b: TT["i"]) -> TT["i"]:
    return (a == 0) * b + (a > 0)
