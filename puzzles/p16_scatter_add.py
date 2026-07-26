"""Puzzle 16 - scatter_add

Compute scatter_add, add together values that link to the same location.
https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html

Every entry of link is less than j.
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def scatter_add_spec(values, link, out):
    for j in range(len(values)):
        out[link[j]] += values[j]


def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    return (link == arange(j)[:, None])*1 @ values
