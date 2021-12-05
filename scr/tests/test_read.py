import numpy as np
import pytest
from baseline import read_cancer_dataset, read_spam_dataset


@pytest.mark.parametrize("path_to_csv", ["../data/cancer.csv"])
def test_zeros_c(path_to_csv):
    X, y = read_cancer_dataset(path_to_csv)
    assert not np.all(X == 0)

@pytest.mark.parametrize("path_to_csv", ["../data/spam.csv"])
def test_zeros_s(path_to_csv):
    X, y = read_spam_dataset(path_to_csv)
    assert not np.all(X == 0)

@pytest.mark.parametrize("path_to_csv", ["../data/cancer.csv"])
def test_target(path_to_csv):
    X, y = read_cancer_dataset(path_to_csv)
    assert (np.unique(y) == np.array([0, 1])).all()
