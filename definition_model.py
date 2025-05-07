import random
from collections import Counter
import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()  # 將 DataFrame 轉換為 NumPy 陣列

        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iterations):
            error = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()  # 將 DataFrame 轉換為 NumPy 陣列
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def predict_shap(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self.net_input(X)

    def __call__(self, X):
        # 讓模型成為可調用的，返回預測結果
        return self.predict(X)


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X_train = X.values
        else:
            self.X_train = X
        if isinstance(y, pd.Series):
            self.y_train = y.values
        else:
            self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.euclidean_distance(
            x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class KNN_manhattan:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X_train = X.values
        else:
            self.X_train = X
        if isinstance(y, pd.Series):
            self.y_train = y.values
        else:
            self.y_train = y

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.manhattan_distance(
            x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class KNN_chebyshev:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X_train = X.values
        else:
            self.X_train = X
        if isinstance(y, pd.Series):
            self.y_train = y.values
        else:
            self.y_train = y

    def chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2))

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.chebyshev_distance(
            x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class NaiveDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.feature_importance = None  # 儲存特徵重要性

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        self.compute_feature_importance(X)  # 計算特徵重要性

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None  # Return None for both idx and thr if no suitable split is found

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - \
                    sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - \
                    sum((num_right[x] / (m - i)) **
                        2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr  # Return the best split values or None

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def compute_feature_importance(self, X):
        feature_importance = np.zeros(self.n_features_)

        def dfs(node, depth=0):
            if node is None:
                return
            feature_importance[node.feature_index] += depth
            dfs(node.left, depth + 1)
            dfs(node.right, depth + 1)
        dfs(self.tree_)
        self.feature_importance = feature_importance  # 儲存特徵重要性

    def top_n_important_features(self, X, n=5):
        sorted_feature_indices = np.argsort(self.feature_importance)[::-1][:n]
        feature_names = X.columns
        top_n_features = [(feature_names[i], self.feature_importance[i])
                          for i in sorted_feature_indices]
        return top_n_features


class PrunedDecisionTree(NaiveDecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2):
        super().__init__(max_depth=max_depth)
        self.min_samples_split = min_samples_split

    def _grow_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split:
            return super()._grow_tree(X, y, depth)  # 剪枝条件

        # 剩下的代码与父类一致
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return self.new_method(node)

    def new_method(self, node):
        return node

    def predict_shap(self, X):
        # A new method for prediction that SHAP will use
        return [self._predict(inputs) for inputs in X]


def split_data(X, y, train_ratio=0.8, random_seed=42):
    random.seed(random_seed)

    data_size = len(X)
    index = list(range(data_size))
    random.shuffle(index)

    train_size = int(data_size*train_ratio)
    test_size = data_size - train_size

    X_train = X.iloc[index[:train_size]]
    y_train = y.iloc[index[:train_size]]
    X_test = X.iloc[index[train_size:]]
    y_test = y.iloc[index[train_size:]]

    return X_train, X_test, y_train, y_test
