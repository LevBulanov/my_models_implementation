import numpy as np
import math

from tree_models.tree import MyDecisionTreeRegressor


class MyGradientBoostingClassifier():
    def __init__(self, max_depth=6, n_estimators=100, max_features=0.7, subsample=0.7, learning_rate=0.1,
                 random_state=None):
        self.max_depth = max_depth
        self.n_est = n_estimators
        self.max_features = max_features
        self.lr = learning_rate
        self.random_gen = np.random.default_rng(random_state)

        if not isinstance(subsample, float):
            raise ValueError("subsample must be float")
        elif subsample > 1 or subsample <= 0:
            raise ValueError("subsample must be in (0, 1.0]")
        else:
            self.subsample = subsample

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(int)

        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("y must be numeric")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('X must be numeric')

        if len(X.shape) == 1:
            X = X[:, None]

        self.n_features = X.shape[1]
        self.n = X.shape[0]
        self.n_per_tree = max(1, min(self.n, int(self.n * self.subsample)))

        if self.max_features == 'sqrt':
            self.features_per_tree = max(1, int(math.sqrt(self.n_features)))
        elif isinstance(self.max_features, int):
            self.features_per_tree = self.max_features
        elif isinstance(self.max_features, float):
            self.features_per_tree = max(1, int(self.max_features * self.n_features))
        elif self.max_features is None:
            self.features_per_tree = self.n_features
        else:
            raise ValueError("unsupported max_features, use 'sqrt', int, float or None")

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        if not self.check_labels_(y):
            raise ValueError('Y must be from 0 to k-1')

        if self.n_classes > 2:
            self.mode = 'multiclass'
            self.boost_trees_multiclass_(X, y)
        elif self.n_classes == 2:
            self.boost_trees_binary_(X, y)
            self.mode = 'binary'
        else:
            raise ValueError("Must be at least 2 classes")

    def boost_trees_binary_(self, X, y):
        self.boosting = []
        mean_y = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
        self.F_start = np.log(mean_y / (1 - mean_y))
        F = np.full_like(y, self.F_start, dtype=float)
        for _ in range(self.n_est):
            features_indxs = self.features_sample_()
            indxs = self.indexes_sample_()

            X_subsample = X[np.ix_(indxs, features_indxs)]
            y_subsample = y[indxs]

            resids = y_subsample - self.sigmoid_(F[indxs])

            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_subsample, resids)

            F += self.lr * tree.predict(X[:, features_indxs])

            self.boosting.append((tree, features_indxs))

    def boost_trees_multiclass_(self, X, y):
        self.boosting = []

        ohe_y = self.ohe_(y)
        mean_y = np.mean(ohe_y, axis=0)
        self.F_start = np.log(np.clip(mean_y, 1e-12, None))
        F = np.tile(self.F_start, (self.n, 1))

        for _ in range(self.n_est):

            features_indxs = self.features_sample_()
            indxs = self.indexes_sample_()

            X_subsample = X[np.ix_(indxs, features_indxs)]
            ohe_y_subsample = ohe_y[indxs]

            resids = ohe_y_subsample - self.softmax_(F[indxs])

            class_trees = []
            for i in range(len(self.classes_)):
                model = MyDecisionTreeRegressor(max_depth=self.max_depth)
                model.fit(X_subsample, resids[:, i])

                F[:, i] += self.lr * model.predict(X[:, features_indxs])

                class_trees.append(model)

            self.boosting.append((class_trees, features_indxs))

    def predict(self, X):
        if not self.boosting:
            raise ValueError("Call fit first")
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        if not self.boosting:
            raise ValueError("Call fit first")
        X = np.asarray(X)

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('X must be numeric')

        if X.ndim == 1:
            if self.n_features == 1:
                X = X[:, None]
            elif X.shape[0] == self.n_features:
                X = X[None, :]
            else:
                raise ValueError("X shape does not match n_features")

        if X.shape[1] != self.n_features:
            raise ValueError("X shape does not match n_features")

        if self.mode == 'binary':
            probs = self.predict_proba_binary_(X)
        else:
            probs = self.predict_proba_multiclass_(X)

        return probs

    def predict_proba_binary_(self, X):
        logits = np.full(X.shape[0], fill_value=self.F_start, dtype=float)

        for tree, feature_indexes in self.boosting:
            tree_preds = tree.predict(X[:, feature_indexes]) * self.lr
            logits += tree_preds

        probs = np.zeros((X.shape[0], 2), dtype=float)
        probs[:, 1] = self.sigmoid_(logits)
        probs[:, 0] = 1 - probs[:, 1]
        return probs

    def predict_proba_multiclass_(self, X):
        probs = np.tile(self.F_start, (X.shape[0], 1))
        for trees, feature_indexes in self.boosting:
            for i, tree in enumerate(trees):
                tree_preds = tree.predict(X[:, feature_indexes]) * self.lr
                probs[:, i] += tree_preds

        return self.softmax_(probs)

    def ohe_(self, y):
        N = y.shape[0]
        k = len(self.classes_)
        return np.eye(k, dtype=int)[y]

    def indexes_sample_(self):
        size = max(2, min(self.n_per_tree, self.n))
        if self.n == 1:
            size = 1
        indexes = self.random_gen.choice(a=self.n, size=size, replace=False)
        return indexes

    def features_sample_(self):
        size = max(1, min(self.features_per_tree, self.n_features))
        indxes = self.random_gen.choice(a=self.n_features, size=size, replace=False)
        return indxes

    def check_labels_(self, labels):
        unique_labels = np.unique(labels)
        return (len(unique_labels) == self.n_classes and
                unique_labels[0] == 0 and
                unique_labels[-1] == self.n_classes - 1)

    @staticmethod
    def sigmoid_(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax_(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)