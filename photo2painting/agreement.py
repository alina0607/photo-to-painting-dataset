"""Is a labelled standard reliable? Agreement between two labellings of the same items.

Before a classifier is judged against hand labels, the labels themselves are judged:
the same items are labelled twice (by the same annotator, in a shuffled order, or by
two annotators) and the two answers compared. Raw agreement is misleading when one
answer is common, so it is set against the agreement two independent annotators with
the same answer frequencies would reach by chance, and summarised as Cohen's kappa:

    kappa = (observed - chance) / (1 - chance)

1 is perfect agreement, 0 is chance level.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Agreement:
    """Agreement between two labellings.

    Attributes:
        n: the number of items labelled twice.
        same: how many got the same answer both times.
        observed: ``same / n``.
        chance: the agreement expected from the answer frequencies alone.
        kappa: Cohen's kappa.
    """

    n: int
    same: int
    observed: float
    chance: float
    kappa: float


def agreement(first: Sequence[str], second: Sequence[str]) -> Agreement:
    """Agreement between two labellings of the same items, in the same order.

    Raises:
        ValueError: if the two lists differ in length or are empty.
    """
    if len(first) != len(second) or not first:
        raise ValueError("two non-empty labellings of the same items are needed")
    n = len(first)
    same = sum(a == b for a, b in zip(first, second, strict=True))
    c1, c2 = Counter(first), Counter(second)
    chance = sum(c1[k] * c2[k] for k in c1) / (n * n)
    observed = same / n
    kappa = 1.0 if chance == 1 else (observed - chance) / (1 - chance)
    return Agreement(n=n, same=same, observed=observed, chance=chance, kappa=kappa)
