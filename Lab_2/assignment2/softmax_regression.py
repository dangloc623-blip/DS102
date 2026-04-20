import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.W = None  
        self.losses = []

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def one_hot_encode(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        N = y.shape[0]
        Y_one_hot = np.zeros((N, num_classes))
        Y_one_hot[np.arange(N), y] = 1
        return Y_one_hot

    def loss_fn(self, Y_one_hot: np.ndarray, Y_hat: np.ndarray) -> float:
        loss = -np.sum(Y_one_hot * np.log(Y_hat + 1e-15)) / Y_one_hot.shape[0]
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape

        num_classes = len(np.unique(y))
        
        self.W = np.zeros((d, num_classes), dtype=np.float64)
        
        Y_one_hot = self.one_hot_encode(y, num_classes)
        
        pbar = tqdm(range(self.epoch), desc="Training Softmax")
        for e in pbar:
            Y_hat = self.predict_proba(X)
            
            delta_Y = Y_hat - Y_one_hot

            gradient = (X.T @ delta_Y) / N
            self.W = self.W - self.lr * gradient

            l = self.loss_fn(Y_one_hot, Y_hat)
            self.losses.append(l)

            pbar.set_postfix({'loss': f"{l:.4f}"})

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray) -> dict:
        precision = precision_score(y, y_pred, average='macro', zero_division=0)
        recall = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W
        return self.softmax(Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y_hat_proba = self.predict_proba(X)
        return np.argmax(Y_hat_proba, axis=1)