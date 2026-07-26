"""The two helpers the puzzles start with.

arange replaces a for-loop, where replaces an if-statement.
"""

import torch


def arange(i: int):
    return torch.tensor(range(i))


def where(q, a, b):
    return (q * a) + (~q) * b
