import numpy as np
import idx2numpy as idx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

test_image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte")
test_label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte")

train_image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-images-idx3-ubyte\train-images.idx3-ubyte")
train_label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-labels-idx1-ubyte\train-labels.idx1-ubyte")

X_train = train_image.reshape(train_image.shape[0], -1) / 255.0
y_train = train_label

X_test = test_image.reshape(test_image.shape[0], -1) / 255.0
y_test = test_label

model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")