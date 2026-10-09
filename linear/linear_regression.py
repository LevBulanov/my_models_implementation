import numpy as np
import random

class MyLinearRegression:
    def __init__(self, method='SGD', seed=21, epoch=4, learning_rate=0.01, tol=1e-03):
        self.method = method
        self.epoch = epoch
        self.lr = learning_rate
        self.tol = tol

        np.random.seed(seed)
        random.seed(seed)
    
    def SGD(self):

        num_epoch = 0
        while num_epoch < self.epoch:
            prev_w = self.w.copy()
            indices = np.random.permutation(self.X.shape[0])
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
            for i in range(X_shuffled.shape[0]):
                self.w = self.w - self.lr * self.calc_grad(X_shuffled[i:i+1], y_shuffled[i:i+1], self.w)
            if np.linalg.norm(self.w - prev_w) < self.tol:
                print("Tol reached")
                return
            num_epoch += 1
        print("Num_epoch reached")

    @staticmethod
    def calc_grad(X_i:np.array, y_i, w):
        grad = -2 * X_i.T * (y_i - X_i @ w)
        return grad
        

    def fit(self, X, y):
        X = np.array(X)
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = np.array(y).reshape(-1)
        shape_of_w = X.shape[1]
        self.w = np.random.rand(shape_of_w + 1, 1)
        self.SGD()
        return self
    
    def predict(self, X):
        X = np.array(X)
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b.dot(self.w).ravel()