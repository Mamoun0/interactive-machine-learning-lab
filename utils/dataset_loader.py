from sklearn.datasets import load_iris


def load_iris_dataset():

    iris = load_iris()

    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names

    return X, y, feature_names, target_names