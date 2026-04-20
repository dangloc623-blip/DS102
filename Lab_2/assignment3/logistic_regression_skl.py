import numpy as np
import idx2numpy as idx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

test_image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte")
test_label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte")

train_image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-images-idx3-ubyte\train-images.idx3-ubyte")
train_label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-labels-idx1-ubyte\train-labels.idx1-ubyte")

def filter_data(data, condition):
    images, labels = data
    new_images = images[labels == condition]
    new_labels = labels[labels == condition]  
    return new_images, new_labels


X_train_0, y_train_0 = filter_data((train_image, train_label), 0)
X_train_1, y_train_1 = filter_data((train_image, train_label), 1)
X_test_0, y_test_0 = filter_data((test_image, test_label), 0)
X_test_1, y_test_1 = filter_data((test_image, test_label), 1)

X_train = np.concatenate((X_train_0, X_train_1), axis=0)
y_train = np.concatenate((y_train_0, y_train_1), axis=0)
X_test = np.concatenate((X_test_0, X_test_1), axis=0)
y_test = np.concatenate((y_test_0, y_test_1), axis=0)

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")