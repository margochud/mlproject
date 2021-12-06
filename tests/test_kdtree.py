import numpy as np
import pytest
from scr_.baseline import KDTree
from sklearn.neighbors  import KDTree as skl_KDTree
@pytest.mark.parametrize("X", [np.random.uniform(-200, 200, (20, 10)) for _ in range(7)])
@pytest.mark.parametrize("y", [np.random.randint(0, 5, 20) for _ in range(7)])
@pytest.mark.parametrize("leafsize", [np.random.randint(0, 40) for _ in range(7)])
def test_istree(X, y, leafsize):
    tree = KDTree(X, leaf_size=leafsize)
    true_tree = skl_KDTree(X, leaf_size=leafsize)
    pr = tree.query(X)
    pr_skl = true_tree.query(X, k=4, return_distance=False)
    assert np.all(pr == pr_skl)