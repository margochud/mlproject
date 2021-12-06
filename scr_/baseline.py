import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas as pd
from IPython import get_ipython
from typing import NoReturn, Tuple, List


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).

   """
    df = pd.read_csv(path_to_csv)  # считываем
    df.loc[df['label'] == "M", 'label'] = 1  # меняем
    df.loc[df['label'] == "B", 'label'] = 0
    df = df.sample(frac=1)  # делаем shuffle
    Y = df["label"]  # отдельно берем таргет
    X = df.drop('label', axis=1)  # отдельно берем фичи
    return np.array(X), np.array(Y)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    df = pd.read_csv(path_to_csv)  # считываем
    df = df.sample(frac=1)  # делаем shuffle
    Y = df["label"]  # отдельно берем таргет
    X = df.drop('label', axis=1)  # отдельно берем фичи
    return np.array(X), np.array(Y)


def train_test_split(X: np.array, y: np.array, ratio: float
                     ) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    l = int(ratio * X.shape[0])
    X_train = X[:l]  # берем строки с 0 по l, столбцы все те же
    X_test = X[l:]
    y_train = y[:l]
    y_test = y[l:]
    return X_train, y_train, X_test, y_test




def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array
                                  ) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    true = sum(y_pred == y_true)  # все правильные делить на все неправильные - accuracy
    all_dots = y_pred.shape[0]
    accuracy = true / all_dots

    precision, recall = np.array([]), np.array([])
    # для того, чтобы посчитать метрики для каждого класса, нам нужно
    # посчитать для них TP, FN, FP
    # еще надо понять, для чего считать, то есть нам нужен массив с классами
    classes = np.unique(y_true)
    for item in classes:
        TP = sum((y_pred == item) * (y_true == item))
        FP = sum((y_pred == item) * (y_true != item))
        FN = sum((y_pred != item) * (y_true == item))
        precision = np.append(precision, [TP / (TP + FP)])
        recall = np.append(recall, [TP / (TP + FN)])
    return precision, recall, accuracy


def plot_precision_recall(X_train, y_train, X_test, y_test, path, max_k=30):
    ks = list(range(1, max_k + 1))
    classes = len(np.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k in ks:
        classifier = KNearest(k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in range(classes):
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(ys) - 0.01, np.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{ylabel}.png')

    plot(ks, recalls, "Recall")
    plot(ks, precisions, "Precision")
    plot(ks, [accuracies], "Accuracy", legend=False)

class Node:
    def __init__(self, value=None, axis=None, data=None):
        self.value = value
        self.axis = axis
        self.data = data
        self.left = None
        self.right = None

    def single_query(self, root, point, k):

        if root.left is None and root.right is None:

            neigh_dist = np.sqrt(np.sum((root.data[:, 1:] - point) ** 2, axis=1))
            neigh_index = np.argsort(neigh_dist)
            index = root.data[neigh_index][:k]
            return neigh_dist[neigh_index], index

        else:
            axis = root.axis - 1
            if root.value > point[axis]:
                neigh_dist, index = self.single_query(root.left, point, k)
                opposite = root.right

            else:
                neigh_dist, index = self.single_query(root.right, point, k)
                opposite = root.left

            if neigh_dist[-1] >= np.sqrt(np.sum(point[axis] - root.value) ** 2) or len(index) < k:
                opposite_dist, opposite_index = self.single_query(opposite, point, k)
                return merge(opposite_index, opposite_dist, index, neigh_dist, k)

            return neigh_dist, index


def merge(opposite_ind, opposite_dist, index, neigh_dist, k):
    i = j = 0
    index_merged = []
    dist_merged = []
    while (i < len(opposite_ind)) and (j < len(index)) and (i + j < k):
        if opposite_dist[i] <= neigh_dist[j]:
            index_merged.append(opposite_ind[i])
            dist_merged.append(opposite_dist[i])
            i += 1
        else:
            index_merged.append(index[j])
            dist_merged.append(neigh_dist[j])
            j += 1
    delta = k - i - j
    index_merged.extend(opposite_ind[i: i + delta])
    dist_merged.extend(opposite_dist[i: i + delta])
    index_merged.extend(index[j: j + delta])
    dist_merged.extend(neigh_dist[j: j + delta])
    return dist_merged, index_merged


class KDTree:

    def __init__(self, X, leaf_size=40):
        self.X = np.hstack([np.arange(X.shape[0]).reshape(-1, 1), X])
        self.dim = X[0].size

        self.leaf_size = leaf_size
        self.root = self.build_tree(self.X, depth=0)

    def build_tree(self, X, depth=0):

        axis = (depth % self.dim) + 1
        median = np.median(X[:, axis])
        left, right = X[X[:, axis] < median], X[X[:, axis] >= median]

        if left.shape[0] < self.leaf_size or right.shape[0] < self.leaf_size:
            return Node(data=X)

        root = Node(value=median, axis=axis)

        root.left = self.build_tree(left, depth + 1)
        root.right = self.build_tree(right, depth + 1)

        return root

    def index_extraction(self, point, k=4):
        one_point_ans = []
        point_neigh = self.root.single_query(self.root, point, k=k)
        for index in point_neigh[1]:
            one_point_ans.append(int(index[0]))
        return one_point_ans

    def query(self, X, k=4):
        res = []
        for point in X:
            ans = self.index_extraction(point, k=k)
            res.append(ans)
        return res


# ### Задание 5  (3 балла)
# Осталось реализовать сам классификатор. Реализуйте его, используя KD-дерево.
#
# Метод `__init__` принимает на вход количество соседей, по которым предсказывается класс, и размер листьев KD-дерева.
#
# Метод `fit` должен по набору данных и меток строить классификатор.
#
# Метод `predict_proba` должен предсказывать веротности классов для заданного набора данных основываясь на классах соседей

# In[17]:


# сделаем класс как в sklearn
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):  # считаем среднее и стандартное отклонение
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, X):  # пересчитываем
        X = (X - self.mean) / self.std
        return X

    def fit_transform(self, X):  # вместе
        X = self.fit(X).transform(X)
        return X


# In[18]:


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.tree = KDTree(X, self.leaf_size)
        self.y = y

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """

        neigh = self.tree.query(X, self.n_neighbors)  # вытащили соседей
        classes = np.unique(self.y)  # посчитали кол-во классов
        pred = self.y[neigh]  # классы индексов

        proba = np.zeros(shape=(X.shape[0], classes.shape[0]))

        for i, item in enumerate(classes):  # считаем, где сколько классов
            proba[:, i] = np.sum(pred == item, axis=1)

        return proba / self.n_neighbors

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(self.predict_proba(X), axis=1)





