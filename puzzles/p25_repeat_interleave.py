"""Puzzle 25 - repeat_interleave (bonus)

Compute repeat_interleave, repeating each element of v by its own count.
https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html

The counts are non-negative and sum to the output length i.
"""

from torchtyping import TensorType as TT

from puzzles.base import arange
from puzzles.p01_ones import ones
from puzzles.p02_sum import sum
from puzzles.p07_cumsum import cumsum


def repeat_interleave_spec(v, c, out):
    t = 0
    for k in range(len(v)):
        for _ in range(c[k]):
            out[t] = v[k]
            t += 1


def repeat_interleave(v: TT["j"], c: TT["j"], i: int) -> TT["i"]:
    return v[(cumsum(c) <= arange(sum(c))[:, None])*1 @ ones(c.shape[0])]
