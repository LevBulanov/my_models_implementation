import numpy as np

class MyMLP:
    def __init__(self, n_hidden=100, activation="relu", lr=0.001, n_epoch=3, 
                 batch_size=32, optimizer='Adam'):
        self.n_hidden: int = n_hidden
        self.activation: str = activation
        self.lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        input_layer_w_shape = (self.n_hidden, X.shape[1]) 
        input_layer_bias_shape = self.n_hidden

        hidden_layer_w_shape = (1, self.n_hidden)
        hidden_layer_bias_shape = 1

        self.input_layer_weights = self.he_normal_initialization(input_layer_w_shape)
        self.input_layer_bias = np.zeros(input_layer_bias_shape)

        self.hidden_layer_weights = self.he_normal_initialization(hidden_layer_w_shape)
        self.hidden_layer_bias = np.zeros(hidden_layer_bias_shape)

        if self.optimizer == 'Adam':
            self.mW1 = np.zeros_like(self.input_layer_weights)
            self.vW1 = np.zeros_like(self.input_layer_weights)
            self.mb1 = np.zeros_like(self.input_layer_bias)
            self.vb1 = np.zeros_like(self.input_layer_bias)

            self.mW2 = np.zeros_like(self.hidden_layer_weights)
            self.vW2 = np.zeros_like(self.hidden_layer_weights)
            self.mb2 = np.zeros_like(self.hidden_layer_bias)
            self.vb2 = np.zeros_like(self.hidden_layer_bias)
            self.t = 0

        for _ in range(self.n_epoch):
            indices = np.random.permutation(X.shape[0])
            shuffled_X = X[indices]
            shuffled_y = y[indices]
            
            for i in range(0, X.shape[0], self.batch_size):
                batch_X = shuffled_X[i:i + self.batch_size]
                batch_y = shuffled_y[i:i + self.batch_size]
                
                self.forward(batch_X)
                self.backward(batch_X, batch_y)
                if self.optimizer == 'Adam':
                    self.Adam_step()
                else:
                    self.SGD_step()

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        probs = self.forward(X)
        return (probs >= 0.5).astype(float).ravel()
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self.forward(X).ravel()
        
    def forward(self, X):
        # (batch, n_features) @ (n_hidden, n_features)^T -> (batch, n_hidden)
        self.z_1 = X @ self.input_layer_weights.T + self.input_layer_bias
       
        if self.activation == 'relu':
            self.a_1 = self.relu(self.z_1)
        elif self.activation == 'sigmoid':
            self.a_1 = self.sigmoid(self.z_1)
        elif self.activation == 'cosine':
            self.a_1 = self.cosine(self.z_1)
        else:
            raise ValueError("Unknown activation")

        self.z_2 = self.a_1 @ self.hidden_layer_weights.T + self.hidden_layer_bias
        self.a_2 = self.sigmoid(self.z_2)

        return self.a_2

    def backward(self, X, y):
        batch_size = X.shape[0]

        dZ_2 = self.a_2 - y
        
        self.dW_2 = dZ_2.T @ self.a_1 / batch_size
        self.db_2 = dZ_2.mean(axis=0)

        dA_1 = dZ_2 @ self.hidden_layer_weights

        if self.activation == 'relu':
            dZ_1 = dA_1 * self.relu_grad(self.z_1)
        elif self.activation == 'sigmoid':
            dZ_1 = dA_1 * self.sigmoid_grad(self.a_1)
        elif self.activation == 'cosine':
            dZ_1 = dA_1 * self.cosine_grad(self.z_1)
        else:
            raise ValueError("Unknown activation")

        self.dW_1 = dZ_1.T @ X / batch_size
        self.db_1 = dZ_1.mean(axis=0)

    def SGD_step(self):
        self.input_layer_weights -= self.lr * self.dW_1
        self.input_layer_bias -= self.lr * self.db_1

        self.hidden_layer_weights -= self.lr * self.dW_2
        self.hidden_layer_bias -= self.lr * self.db_2

    def Adam_step(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

        # ---- W1 ----
        self.mW1 = beta1 * self.mW1 + (1 - beta1) * self.dW_1
        self.vW1 = beta2 * self.vW1 + (1 - beta2) * (self.dW_1 ** 2)

        mW1_hat = self.mW1 / (1 - beta1 ** self.t)
        vW1_hat = self.vW1 / (1 - beta2 ** self.t)

        self.input_layer_weights -= self.lr * mW1_hat / (np.sqrt(vW1_hat) + eps)

        # ---- b1 ----
        self.mb1 = beta1 * self.mb1 + (1 - beta1) * self.db_1
        self.vb1 = beta2 * self.vb1 + (1 - beta2) * (self.db_1 ** 2)

        mb1_hat = self.mb1 / (1 - beta1 ** self.t)
        vb1_hat = self.vb1 / (1 - beta2 ** self.t)

        self.input_layer_bias -= self.lr * mb1_hat / (np.sqrt(vb1_hat) + eps)

        # ---- W2 ----
        self.mW2 = beta1 * self.mW2 + (1 - beta1) * self.dW_2
        self.vW2 = beta2 * self.vW2 + (1 - beta2) * (self.dW_2 ** 2)

        mW2_hat = self.mW2 / (1 - beta1 ** self.t)
        vW2_hat = self.vW2 / (1 - beta2 ** self.t)

        self.hidden_layer_weights -= self.lr * mW2_hat / (np.sqrt(vW2_hat) + eps)

        # ---- b2 ----
        self.mb2 = beta1 * self.mb2 + (1 - beta1) * self.db_2
        self.vb2 = beta2 * self.vb2 + (1 - beta2) * (self.db_2 ** 2)

        mb2_hat = self.mb2 / (1 - beta1 ** self.t)
        vb2_hat = self.vb2 / (1 - beta2 ** self.t)

        self.hidden_layer_bias -= self.lr * mb2_hat / (np.sqrt(vb2_hat) + eps)



    def he_normal_initialization(self, shape):
        n_in = shape[1] # Number of input units (fan-in)
        std_dev = np.sqrt(2.0 / n_in)
        weights = np.random.normal(loc=0.0, scale=std_dev, size=shape)
        return weights
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_grad(a):
        return a * (1 - a)
    
    @staticmethod
    def cosine(x):
        return np.cos(x)
    
    @staticmethod
    def cosine_grad(z):
        return -np.sin(z)