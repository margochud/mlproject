
from sklearn.neighbors import KNeighborsClassifier
#from baseline import train_test_split
from sklearn.model_selection import train_test_split
from baseline import KNearest
from baseline import StandardScaler
import numpy as np
import pytest

@pytest.mark.parametrize("X", [np.random.uniform(-200, 200, (20, 10)) for _ in range(5)])
@pytest.mark.parametrize("y", [np.random.randint(0, 2, 20) for _ in range(5)])
@pytest.mark.parametrize("k", [np.random.randint(0, 5) for _ in range(5)])

def test_KNearest(X, y, k):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    knn = KNearest(n_neighbors=k, leaf_size=20)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    true_knn = KNeighborsClassifier(n_neighbors=k, leaf_size=20)
    true_knn.fit(X_train, y_train)
    knn_pred_true = true_knn.predict(X_test)

    assert np.all(knn_pred == knn_pred_true)
