from baseline import train_test_split
from baseline import StandardScaler
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as true_Stdsc


@pytest.mark.parametrize("X", [np.random.uniform(-200, 200, (20, 10)) for _ in range(5)])
@pytest.mark.parametrize("y", [np.random.randint(0, 20, 20) for _ in range(5)])
def test_mean(X, y):
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
    scale = StandardScaler()
    true_scale = true_Stdsc()
    assert np.all(scale.fit_transform(X_train) == true_scale.fit_transform(X_train))

