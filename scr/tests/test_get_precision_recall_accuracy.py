import numpy as np
import pytest
from baseline import get_precision_recall_accuracy
@pytest.mark.parametrize("y_true", [np.random.randint(0, 2, 20) for _ in range(10)])
@pytest.mark.parametrize("y_pred", [np.random.randint(0, 2, 20) for _ in range(10)])

def test_pr_rec_acc(y_true, y_pred):
    pr, rec, acc = get_precision_recall_accuracy(y_pred, y_true)
    assert np.all(pr <= 1)
    assert np.all(pr >= 0)
    assert np.all(rec <= 1)
    assert np.all(rec >= 0)
    assert np.all(acc <= 1)
    assert np.all(acc >= 0)
