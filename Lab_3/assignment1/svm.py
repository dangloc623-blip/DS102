import numpy as np

class SVM:
    def __init__(self, C: float = 30.0, learning_rate: float = 0.0001, n_iters: int = 100):
        self.C = C
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_iters):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for idx in indices:
                x_i = x[idx]
                y_i = y[idx]

                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * y_i * x_i)
                    self.b -= self.lr * (-self.C * y_i)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_hat = np.dot(x, self.w) + self.b
        return np.sign(y_hat)

    def hinge_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        delta = 1 - y * y_hat
        delta[delta < 0] = 0

        regularization_loss = 0.5 * np.dot(self.w.T, self.w)
        hinge_loss_val = self.C * np.sum(delta)

        return regularization_loss + hinge_loss_val