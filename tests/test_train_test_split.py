import numpy as np
import pytest
from scr_.baseline import train_test_split
@pytest.mark.parametrize("X", [np.random.uniform(-200, 200, (20, 10))])
@pytest.mark.parametrize("y", [np.random.randint(0, 5, 20) for _ in range(10)])
@pytest.mark.parametrize("ratio",
                         [np.random.uniform(0, 1) for _ in range(10)])
def test_shape(X, y, ratio):
    X_train, y_train, X_test, y_test = train_test_split(X, y, ratio)
    assert int(X.shape[0] * ratio) == X_train.shape[0]
    assert int(y.shape[0] * ratio) == y_train.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
