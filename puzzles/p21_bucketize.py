"""Puzzle 21 - bucketize

Compute bucketize.
https://pytorch.org/docs/stable/generated/torch.bucketize.html

boundaries is sorted and non-negative.
"""

from torchtyping import TensorType as TT

from puzzles.p01_ones import ones


def bucketize_spec(v, boundaries, out):
    for i, val in enumerate(v):
        out[i] = 0
        for j in range(len(boundaries)-1):
            if val >= boundaries[j]:
                out[i] = j + 1
        if val >= boundaries[-1]:
            out[i] = len(boundaries)


def bucketize(v: TT["i"], boundaries: TT["j"]) -> TT["i"]:
    return (v[:, None] >= boundaries)*1 @ ones(boundaries.shape[0])
