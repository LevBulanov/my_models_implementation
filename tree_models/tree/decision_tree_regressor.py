from tree_models.node import Node

import math

import numpy as np

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=7):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if len(X.shape) == 2:
            self.n_features = X.shape[1]
        else:
            self.n_features = 1
            X = X[:, None]

        full_indexes = np.arange(X.shape[0])
        self.root = Node(full_indexes)
        self.root.impurity_before = self.calc_mse_(y)

        self.grow_tree_(X, y, self.root, 0)

    def predict(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            if self.n_features == 1:
                X = X[:, None]
            elif X.shape[0] == self.n_features:
                X = X[None, :]
            else:
                raise ValueError("X shape does not match n_features")

        preds = np.fromiter((self.predict_one_(x) for x in X), dtype=float)
        return preds

    def find_leaf_(self, x):
        c_node = self.root
        while not c_node.is_leaf:
            c_thresh = c_node.split_threshold
            c_feature_index = c_node.feature_index
            if x[c_feature_index] <= c_thresh:
                c_node = c_node.left
            else:
                c_node = c_node.right
        return c_node

    def predict_one_(self, x):
        leaf = self.find_leaf_(x)
        return leaf.predicted_value

    def grow_tree_(self, X, y, root: Node, depth):
        if depth >= self.max_depth or math.isclose(root.impurity_before, 0.0):
            self.change_to_leaf_(root, y)
            return
        
        gain, feature_index, tresh = self.find_best_split_(root, X, y)

        if gain > 1e-6:
            root.feature_index = feature_index
            root.split_threshold = tresh
            self.create_children_(feature_index, tresh, X, y, root, root.node_indxs)
        else:
            self.change_to_leaf_(root, y)
            return

        self.grow_tree_(X, y, root.left, depth + 1)
        self.grow_tree_(X, y, root.right, depth + 1)
        

    def make_split_one_feature_(self, X, y):
        N = X.shape[0]
        order = np.argsort(X, kind='mergesort')
        Xs = X[order]
        ys = y[order].astype(np.float64)

        S = np.cumsum(ys)
        S2 = np.cumsum(ys * ys)

        S_tot = S[-1]
        S2_tot = S2[-1]

        SSE_parent = S2_tot - (S_tot ** 2) / N
        MSE_parent = SSE_parent / N

        best_gain = -np.inf
        best_thresh = None
        best_imp = None

        for i in range(N - 1):
            if Xs[i] == Xs[i+1]:
                continue

            NL = i + 1
            NR = N - NL

            S_L = S[i]
            S2_L = S2[i]
            SSE_L = S2_L - (S_L ** 2) / NL

            S_R = S_tot - S_L
            S2_R = S2_tot - S2_L
            SSE_R = S2_R - (S_R ** 2) / NR

            MSE_split = (SSE_L + SSE_R) / N
            gain = MSE_parent - MSE_split

            if gain > best_gain:
                best_gain = gain
                best_imp = MSE_split
                best_thresh = 0.5 * (Xs[i] + Xs[i+1])

        return best_thresh, best_imp, best_gain

    def find_best_split_(self, node: Node, X, y):
        
        cur_indxs = node.node_indxs
        cur_data = X[cur_indxs]
        cur_targets = y[cur_indxs]

        best_gain = -np.inf
        best_tresh = None
        best_feature_index = None

        for i in range(self.n_features):
            feature_values = cur_data[:, i]
            tresh, _, gain = self.make_split_one_feature_(feature_values, cur_targets)

            if gain == -np.inf:
                continue
            
            if gain > best_gain:
                best_gain = gain
                best_tresh = tresh
                best_feature_index = i

        return best_gain, best_feature_index, best_tresh


    def change_to_leaf_(self, node: Node, y):
        node.is_leaf = True
        node_targets = y[node.node_indxs]
        node.predicted_value = np.mean(node_targets)
        node.leaf_impurity = node.impurity_before

    def create_children_(self, feature_index, tresh, X, y, parent:Node, parent_indxs):
        left_mask = X[parent_indxs][:, feature_index] <= tresh

        left_indxs = parent_indxs[left_mask]
        right_indxs = parent_indxs[~left_mask]
        
        left_node = Node(left_indxs)
        left_node.impurity_before = self.calc_mse_(y[left_indxs])

        right_node = Node(right_indxs)
        right_node.impurity_before = self.calc_mse_(y[right_indxs])

        parent.left = left_node
        parent.right = right_node
        
        
    def calc_mse_(self, y):
        N = len(y)
        mean_y = np.mean(y)
        mse = np.sum(y - mean_y) / N
        return mse