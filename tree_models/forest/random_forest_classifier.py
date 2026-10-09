import numpy as np
import math

from tree_models.tree import MyDecisionTreeClassifier

class MyRandomForestClassifier():
    def __init__(self, n_estimators=100, max_depth=8, max_features='sqrt', random_state=None):
        self.n_est = n_estimators
        self.max_depth = max_depth
        self.random_gen = np.random.default_rng(random_state)

        if max_features != 'sqrt' and not isinstance(max_features, int):
            raise ValueError("unsupported max_features, use 'sqrt' or int")

        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)

        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain ints only")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('X must be numeric')

        if len(X.shape) == 1:
            X = X[:, None]

        self.n_features = X.shape[1]
        self.n = X.shape[0]

        if self.max_features == 'sqrt':
            self.features_per_tree = round(math.sqrt(self.n_features))
        elif isinstance(self.max_features, int):
            self.features_per_tree = self.max_features
        else:
            raise ValueError("unsupported max_features, use 'sqrt' or int")

        self.forest = []
        for _ in range(self.n_est):
            indexes = self.indexes_bootstrap_()
            feature_indexes = self.features_bootstrap_()

            X_subsample = X[np.ix_(indexes, feature_indexes)]
            y_subsample = y[indexes]

            model = MyDecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(X_subsample, y_subsample)

            self.forest.append((model, feature_indexes))

    def predict(self, X):
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

        preds = np.stack([model.predict(X[:, features_indexes])
                          for model, features_indexes in self.forest],
                         axis=1)
        mods = np.array([np.bincount(row.astype(int)).argmax() for row in preds])

        return mods

    def predict_proba(self, X):
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

        probs_list = []
        for model, feat_idx in self.forest:
            p = model.predict_proba(X[:, feat_idx])
            aligned = np.zeros((X.shape[0], len(self.classes_)))
            col_idx = np.searchsorted(self.classes_, model.classes_)
            aligned[:, col_idx] = p
            probs_list.append(aligned)
        probs = np.stack(probs_list, axis=0)
        return np.mean(probs, axis=0)

    def indexes_bootstrap_(self):
        indexes = self.random_gen.choice(a=self.n, size=self.n, replace=True)
        return indexes

    def features_bootstrap_(self):
        size = max(1, min(self.features_per_tree, self.n_features))
        return self.random_gen.choice(a=self.n_features, size=size, replace=False)
