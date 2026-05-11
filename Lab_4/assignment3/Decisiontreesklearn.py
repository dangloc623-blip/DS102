import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

url = r"D:\Dataset\wine+quality\winequality-red.csv"
df = pd.read_csv(url, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)

f1 = f1_score(y_test, y_pred, average='weighted')

print(f"[Decision Tree - Sklearn] Weighted F1 Score: {f1:.4f}")

