import pytest
from sklearn.metrics import cohen_kappa_score

from photo2painting.agreement import agreement


def test_perfect_agreement():
    a = agreement(["x", "y", "x"], ["x", "y", "x"])
    assert a.same == 3
    assert a.kappa == pytest.approx(1.0)


def test_kappa_matches_sklearn():
    first = list("AABBCCAABCAB")
    second = list("ABBBCAACBCAA")
    a = agreement(first, second)
    assert a.kappa == pytest.approx(cohen_kappa_score(first, second))


def test_chance_agreement_from_the_answer_frequencies():
    # both annotators answer A half the time: chance agreement is 0.5
    a = agreement(list("AABB"), list("ABAB"))
    assert a.chance == pytest.approx(0.5)
    assert a.observed == pytest.approx(0.5)
    assert a.kappa == pytest.approx(0.0)


def test_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same items"):
        agreement(["a"], ["a", "b"])
