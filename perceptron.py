import numpy as np


class Perceptron:
    def __init__(self, n_features, loss_type='cross_entropy', lambda_l2=0.0):
        self.w = np.random.randn(n_features, 1) * 0.01
        self.b = 0.0
        self.loss_type = loss_type
        self.lambda_l2 = lambda_l2
        self.loss_history = {'train': [], 'val': []}

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b

        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred, X=None):
        eps = 1e-15

        if self.loss_type == 'cross_entropy':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_type == 'hinge':
            y_true_hinge = 2 * y_true - 1
            z = np.dot(X, self.w) + self.b
            loss = np.mean(np.maximum(0, 1 - y_true_hinge * z))
        else:
            raise ValueError("Unknown loss_type")
        
        if self.lambda_l2 > 0:
            loss += 0.5 * self.lambda_l2 * np.sum(self.w ** 2)
            
        return loss

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, momentum=0.0):
        m = X_train.shape[0]
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        v_w = np.zeros_like(self.w)
        v_b = 0.0

        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                if self.loss_type == 'cross_entropy':
                    y_pred = self.forward(X_batch)
                    error = y_pred - y_batch
                elif self.loss_type == 'hinge':
                    y_batch_hinge = 2 * y_batch - 1
                    z = np.dot(X_batch, self.w) + self.b
                    margin = y_batch_hinge * z
                    mask = (margin < 1).astype(float)
                    error = -y_batch_hinge * mask

                dw = np.dot(X_batch.T, error) / X_batch.shape[0]
                db = np.mean(error)

                if self.lambda_l2 > 0:
                    dw += self.lambda_l2 * self.w

                if momentum > 0:
                    v_w = momentum * v_w + lr * dw
                    v_b = momentum * v_b + lr * db
                    self.w -= v_w
                    self.b -= v_b
                else:
                    self.w -= lr * dw
                    self.b -= lr * db

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)
            train_loss = self.compute_loss(y_train, train_pred, X_train)
            val_loss = self.compute_loss(y_val, val_pred, X_val)

            self.loss_history['train'].append(train_loss)
            self.loss_history['val'].append(val_loss)

        return self

    def predict(self, X, threshold=0.5):
        probs = self.forward(X)
        return (probs >= threshold).astype(int).flatten()

    def predict_proba(self, X):
        return self.forward(X).flatten()