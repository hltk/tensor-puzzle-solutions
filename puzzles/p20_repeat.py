"""Puzzle 20 - repeat

Compute repeat (1d).
https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html

d holds a single value, the number of output rows.
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def repeat_spec(a, d, out):
    for i in range(d[0]):
        for k in range(len(a)):
            out[i][k] = a[k]


def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    return (arange(d) * 0)[:, None] + a
