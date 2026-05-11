import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from Decisiontree import DecisionTreeClassifier

url = r"D:\Dataset\wine+quality\winequality-red.csv"
df = pd.read_csv(url, sep=";")

X = df.drop("quality", axis=1).values
y_raw = df["quality"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Evaluation: ")
print(f"F1 Score: {f1:.4f}")