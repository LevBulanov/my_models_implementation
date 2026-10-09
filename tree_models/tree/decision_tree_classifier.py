from tree_models.node import Node

import math

import numpy as np

class MyDecisionTreeClassifier():
    def __init__(self, max_depth=7):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        y = self.encode_targets_(y)
        self.n_classes = np.max(y) + 1
        
        if len(X.shape) == 2:
            self.n_features = X.shape[1]
        else:
            self.n_features = 1
            X = X[:, None]

        full_indexes = np.arange(X.shape[0])
        self.root = Node(full_indexes)
        self.root.impurity_before = self.calc_gini_(y)

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

        preds = np.fromiter((self.predict_one_(x) for x in X), dtype=int)
        return self.decode_preds(preds)

    def predict_proba(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            if self.n_features == 1:
                X = X[:, None]
            elif X.shape[0] == self.n_features:
                X = X[None, :]
            else:
                raise ValueError("X shape does not match n_features")

        probs = np.vstack([self.predict_proba_one_(x) for x in X])
        return probs
    
    def encode_targets_(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("Ожидается одномерный массив меток (1D).")

        self.label_to_id = {}
        self.id_to_label = []

        encoded = np.empty(y.shape[0], dtype=np.int64)
        next_id = 0
        for i, label in enumerate(y):
            if label not in self.label_to_id:
                self.label_to_id[label] = next_id
                self.id_to_label.append(label)
                next_id += 1
            encoded[i] = self.label_to_id[label]

        self._id_to_label_arr = np.array(self.id_to_label, dtype=object)

        return encoded
    
    def decode_preds(self, y_encoded):
        y_encoded = np.asarray(y_encoded)
        if y_encoded.size > 0:
            min_id = int(y_encoded.min())
            max_id = int(y_encoded.max())
            if min_id < 0 or max_id >= len(self._id_to_label_arr):
                raise ValueError("Найдены кодированные метки вне допустимого диапазона.")

        return self._id_to_label_arr[y_encoded]


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

    def predict_proba_one_(self, x):
        leaf = self.find_leaf_(x)
        N = np.sum(leaf.class_counts)
        probs = leaf.class_counts / N
        return probs

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
        if self.n_classes is None:
            self.n_classes = np.max(y) + 1
        
        N = X.shape[0]

        order = np.argsort(X, kind='mergesort')
        X_sorted = X[order]
        y_sorted = y[order]

        total = np.bincount(y, minlength=self.n_classes)
        left = np.zeros(self.n_classes, dtype=int)
        right = total.copy()

        s_left = 0
        s_right = (right ** 2).sum()

        G_parent = 1 - s_right / (N ** 2)

        best_gain = -np.inf
        best_thresh = None
        best_imp = None

        for i in range(N - 1):
            c = y_sorted[i]
            s_left += 2 * left[c] + 1
            s_right -= 2 * right[c] - 1

            left[c] += 1
            right[c] -= 1

            if X_sorted[i] == X_sorted[i + 1]:
                continue

            NL = i + 1
            NR = N - NL

            G_left = 1 - s_left / (NL ** 2) if NL > 0 else 0
            G_right = 1 - s_right / (NR ** 2) if NR > 0 else 0
            G_split = (NL * G_left + NR * G_right) / N

            gain = G_parent - G_split
            if gain > best_gain:
                best_gain = gain
                best_imp = G_split
                best_thresh = 0.5 * (X_sorted[i] + X_sorted[i+1])
        
        return best_thresh, best_imp, best_gain


    def find_best_split_(self, node: Node, X, y):
        
        cur_indxs = node.node_indxs
        cur_data = X[cur_indxs]
        cur_labels = y[cur_indxs]

        best_gain = -np.inf
        best_tresh = None
        best_feature_index = None

        for i in range(self.n_features):
            feature_values = cur_data[:, i]
            tresh, imp, gain = self.make_split_one_feature_(feature_values, cur_labels)

            if gain == -np.inf:
                continue
            
            if gain > best_gain:
                best_gain = gain
                best_tresh = tresh
                best_feature_index = i

        return best_gain, best_feature_index, best_tresh


    def change_to_leaf_(self, node: Node, y):
        node.is_leaf = True

        class_counts = np.zeros(self.n_classes, dtype=int)
        unique, counts = np.unique(y[node.node_indxs], return_counts=True)
        for i, unique_v in enumerate(unique):
            class_counts[unique_v] = counts[i]
        node.class_counts = class_counts

        node_lables = y[node.node_indxs]
        node.predicted_value = self.mode_(node_lables)
        node.leaf_impurity = node.impurity_before

    def create_children_(self, feature_index, tresh, X, y, parent:Node, parent_indxs):
        left_mask = X[parent_indxs][:, feature_index] <= tresh

        left_indxs = parent_indxs[left_mask]
        right_indxs = parent_indxs[~left_mask]
        
        left_node = Node(left_indxs)
        left_node.impurity_before = self.calc_gini_(y[left_indxs])

        right_node = Node(right_indxs)
        right_node.impurity_before = self.calc_gini_(y[right_indxs])

        parent.left = left_node
        parent.right = right_node
        
        
    def calc_gini_(self, y):
        N = len(y)
        total = np.bincount(y, minlength=self.n_classes)
        return 1 - np.sum(total ** 2) / N ** 2
    
    def mode_(self, y):
        return np.bincount(y).argmax()