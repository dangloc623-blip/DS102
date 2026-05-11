import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None,min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        num_samples, _ = X.shape

        num_labels = [np.sum(y == i) for i in range(self.n_classes_)]

        predicted_class = np.argmax(num_labels)

        node = {
            "type": "leaf",
            "class": predicted_class
        }

        if (
            depth < (self.max_depth if self.max_depth is not None else np.inf)
            and num_samples >= self.min_samples_split
            and len(np.unique(y)) > 1
        ):
            best_feat, best_thresh = self._best_split(X, y)

            if best_feat is not None:
                left_idx = X[:, best_feat] < best_thresh
                right_idx = ~left_idx

                if (
                    np.sum(left_idx) >= self.min_samples_leaf and
                    np.sum(right_idx) >= self.min_samples_leaf
                ):
                    node = {
                        "type": "node",
                        "feature": best_feat,
                        "threshold": best_thresh,
                        "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
                        "right": self._build_tree(X[right_idx], y[right_idx], depth + 1)
                    }
        return node

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        probs = np.bincount(y, minlength=self.n_classes_) / m
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        best_gini = float("inf")
        best_feat, best_thresh = None, None

        for feat in range(n):
            sorted_idx = np.argsort(X[:, feat])
            X_sorted = X[sorted_idx, feat]
            y_sorted = y[sorted_idx]

            for i in range(1, m):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                thresh = (X_sorted[i] + X_sorted[i - 1]) / 2

                left_y = y_sorted[:i]
                right_y = y_sorted[i:]

                gini_left = self._gini(left_y)
                gini_right = self._gini(right_y)

                weighted_gini = (
                    (len(left_y) * gini_left + len(right_y) * gini_right) / m
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _predict_sample(self, x, node):
        if node["type"] == "leaf":
            return node["class"]

        if x[node["feature"]] < node["threshold"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])