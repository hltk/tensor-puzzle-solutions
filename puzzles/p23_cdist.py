"""Puzzle 23 - cdist (bonus)

Compute cdist, the matrix of pairwise distances between two sets of points.
https://pytorch.org/docs/stable/generated/torch.cdist.html

Squared distances, since sqrt is not in the toolbox.
"""

from torchtyping import TensorType as TT

from puzzles.p01_ones import ones


def cdist_spec(a, b, out):
    for i in range(len(a)):
        for j in range(len(b)):
            total = 0
            for k in range(len(a[i])):
                total += (a[i][k] - b[j][k]) ** 2
            out[i][j] = total


def cdist(a: TT["i", "k"], b: TT["j", "k"]) -> TT["i", "j"]:
    return (a[:, None] - b) ** 2 @ ones(a.shape[1])
