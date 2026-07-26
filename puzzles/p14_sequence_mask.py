"""Puzzle 14 - sequence_mask

Compute sequence_mask, pad out to length per batch.
https://www.tensorflow.org/api_docs/python/tf/sequence_mask

Every entry of length is less than the number of columns.
"""

from torchtyping import TensorType as TT

from puzzles.base import arange


def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0


def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    return values * (arange(values.shape[1]) < length[:, None])
