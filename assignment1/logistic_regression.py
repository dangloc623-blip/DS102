import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = (1 - y) * np.log(1 - y_hat + 1e-15) + y * np.log(y_hat + 1e-15)
        return -l.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d, ), dtype=np.float64)
        pbar = tqdm(range(self.epoch), desc="Training")
        
        for e in pbar:
            y_hat = self.predict_proba(X)
            delta_y = (y_hat - y)
            gradient = (X.T @ delta_y) / N
            self.w = self.w - self.lr * gradient
            l = self.loss_fn(y, y_hat)
            self.losses.append(l)
            pbar.set_postfix({'loss': f"{l:.4f}"})
    
    def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        precision = precision_score(y, y_hat, average='binary', zero_division=0)
        recall = recall_score(y, y_hat, average='binary', zero_division=0)
        f1 = f1_score(y, y_hat, average='binary', zero_division=0)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        y_hat_proba = self.predict_proba(X)
        return (y_hat_proba >= threshold).astype(int)
    

