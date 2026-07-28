"""Puzzle 22 - argmax (bonus)

Compute argmax, the index of the largest element of a vector.
https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

Ties resolve to the first occurrence, as in np.argmax and torch.argmax.
"""

from torchtyping import TensorType as TT

from puzzles.p02_sum import sum
from puzzles.p07_cumsum import cumsum


def argmax_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        if a[i] > a[out[0]]:
            out[0] = i


def argmax(a: TT["i"]) -> TT[1]:
    return sum((cumsum(((a[:, None] < a) * 1 @ a ** 0 == 0) * 1) == 0) * 1)
