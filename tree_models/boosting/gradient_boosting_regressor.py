import numpy as np
import math

from tree_models.tree import MyDecisionTreeRegressor


class MyGradientBoostingRegressor():
    def __init__(self, max_depth=6, n_estimators=100, max_features=0.7, subsample=0.7, learning_rate=0.1, random_state=None):
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
        y = np.asarray(y).ravel()

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
        
        self.boost_trees_(X, y)
        
    def boost_trees_(self, X, y):
        self.boosting = []
        self.F_start = np.mean(y)
        F = np.full_like(y, self.F_start, dtype=float)
        for i in range(self.n_est):
            features_indxs = self.features_sample_()
            indxs = self.indexes_sample_()

            X_subsample = X[np.ix_(indxs, features_indxs)]
            y_subsample = y[indxs]

            resids = y_subsample - F[indxs]

            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_subsample, resids)

            F = F + self.lr * tree.predict(X[:, features_indxs])

            self.boosting.append((tree, features_indxs))

    def predict(self, X):   
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
            
        preds = np.full(X.shape[0], self.F_start, dtype=float)

        for tree, feature_indexes in self.boosting:
            tree_preds = tree.predict(X[:, feature_indexes]) * self.lr
            preds += tree_preds

        return preds
        
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
    




# if __name__ == "__main__":
#     X = np.random.uniform(0, 50, size=(20))

#     X_prov = np.random.uniform(0, 50, size=(10))
#     y = np.random.randint(0, 2, size=(20))

#     model = GradientBoostingClassifier(max_depth=6)

#     model.fit(X, y)

#     preds = model.predict_proba(X_prov)

#     print('hi')