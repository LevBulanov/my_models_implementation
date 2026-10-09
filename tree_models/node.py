class Node():
    def __init__(self, node_indxs, leaf=False):
        self.node_indxs = node_indxs
        if not leaf:
            self.is_leaf = leaf
            self.left = None
            self.right = None
            self.split_threshold = None
            self.impurity_before = None
            self.feature_index = None
        else:
            self.is_leaf = leaf
            self.predicted_value = None
            self.leaf_impurity = None
            self.class_counts = None