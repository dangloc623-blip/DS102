import os
import cv2 as cv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from svm import SVM

base_dir = r"D:\chest_xray"

def load_data(split: str = "train"):
    normal_dir = os.path.join(base_dir, split, "NORMAL")
    pneumonia_dir = os.path.join(base_dir, split, "PNEUMONIA")
    data = []
    labels = []

    def process_folder(folder_path, label_val):
        for img_file in os.listdir(folder_path):
            if img_file.endswith(('.jpeg', '.png', '.jpg')):
                img_path = os.path.join(folder_path, img_file)
                img_array = np.fromfile(img_path, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)

                if image is not None:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    image = cv.resize(image, (128, 128), interpolation=cv.INTER_AREA)
                    data.append(image.flatten())
                    labels.append(label_val)

    print(f"load du lieu {split}")
    process_folder(normal_dir, -1)
    process_folder(pneumonia_dir, 1)
    
    return np.array(data), np.array(labels)


x_train, y_train = load_data("train")
x_val, y_val = load_data("val")
x_test, y_test = load_data("test")

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

model = SVM(C=1.0, learning_rate=0.0001, n_iters=100)
model.fit(x_train, y_train)

y_pred_val = model.predict(x_val)
f1_val = f1_score(y_val, y_pred_val)
print(f"F1 Score (Validation): {f1_val:.4f}")

y_pred_test = model.predict(x_test)

precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f"Precision (Test): {precision:.4f}")
print(f"Recall (Test):    {recall:.4f}")
print(f"F1 Score (Test):  {f1:.4f}")

print(classification_report(y_test, y_pred_test, target_names=['NORMAL (-1)', 'PNEUMONIA (1)']))