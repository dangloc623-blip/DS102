import numpy as np
import matplotlib as mlb
import idx2numpy as idx
from logistic_regression import LogisticRegression
test_label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte")
test_image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte")
test_data = (test_image, test_label)

image = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-images-idx3-ubyte\train-images.idx3-ubyte")
label = idx.convert_from_file(r"C:\Users\DUC LOC\OneDrive\Máy tính\train-labels-idx1-ubyte\train-labels.idx1-ubyte")
train_data = (image, label)


def filter_data(data, condition):
    images, labels = data
    new_images = images[labels == condition]
    new_labels = labels[labels == condition]  
    return new_images, new_labels

train_data_0, train_label_0 = filter_data(train_data, 0)
train_data_1, train_label_1 = filter_data(train_data, 1)

test_data_0, test_label_0 = filter_data(test_data, 0)
test_data_1, test_label_1 = filter_data(test_data, 1)

X_train = np.concatenate((train_data_0, train_data_1), axis=0)
y_train = np.concatenate((train_label_0, train_label_1), axis=0)

X_test = np.concatenate((test_data_0, test_data_1), axis=0)
y_test = np.concatenate((test_label_0, test_label_1), axis=0)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


model = LogisticRegression(epoch=1000, lr=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = model.evaluate(y_test, y_pred)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1']:.4f}")

