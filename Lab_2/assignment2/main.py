import numpy as np
import matplotlib as mlb
import idx2numpy as idx
from softmax_regression import SoftmaxRegression

test_image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte")
test_label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte")

train_image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-images-idx3-ubyte\train-images.idx3-ubyte")
train_label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-labels-idx1-ubyte\train-labels.idx1-ubyte") 
X_train = train_image.reshape(train_image.shape[0], -1) / 255.0
y_train = train_label

X_test = test_image.reshape(test_image.shape[0], -1) / 255.0
y_test = test_label

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

model = SoftmaxRegression(epoch=1000, lr=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics = model.evaluate(y_test, y_pred)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1']:.4f}")