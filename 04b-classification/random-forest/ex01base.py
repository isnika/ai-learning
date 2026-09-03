from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris(as_frame=True) #yeu cau tra dl dang pandas dataframe/series thay vi numpy arr
df = iris.frame

print("First 10 lines: ")
print(df.head(10))
print("\nSecond 10 lines: ")
print(df.tail(10))
print("Datasets shape: ")
print(df.shape)
print("Datasets information: ")
print(df.info())

X = df.drop("target", axis=1)
y = df["target"]

print("X shape: ", X.shape)
print("y shape: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100, #tao 100 decision in random forest
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual: ")
print(y_test.values)
print("Predicted: ")
print(y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

import pandas as pd

importance = pd.DataFrame({
    "feature": iris.feature_names,
    "importance": model.feature_importances_
})

print(importance)