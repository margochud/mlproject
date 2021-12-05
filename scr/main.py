from baseline import read_cancer_dataset, train_test_split, KNearest
import numpy as np

if __name__ == '__main__':
    X, y = read_cancer_dataset("../data/cancer.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    clf = KNearest(n_neighbors=4)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(pred)