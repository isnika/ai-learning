from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import  XGBClassifier

data = load_iris(as_frame=True)
df = data.frame

print("First 10 line: ")
print(df.head(10))
print("\nSecond 10 line: ")
print(df.tail(10))

print("\nDatasets shape: ")
print(df.shape)
print("\nDatasets imformation: ")
print(df.info())

X = df.drop(["target"], axis=1)
y = df["target"]

print("\nX shape: ")
print(X.shape)
print("\ny shape: ")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape: ")
print(X_train.shape)
print("\nTest shape: ")
print(X_test.shape)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual: ")
print(y_test.values)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ")
print(accuracy)
report = classification_report(y_test, y_pred)
print("Classification report: ")
print(report)