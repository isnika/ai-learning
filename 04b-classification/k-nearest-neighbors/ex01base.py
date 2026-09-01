'''
dùng Iris Dataset
Iris Dataset là một dataset kinh điển trong Machine Learning
Có 150 mẫu hoa Iris, gồm 4 đặc trưng đầu vào và 1 nhãn (target).
sepal: dai khoa
petal: canh hoa
'''

from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
print("Fisrt 5 lines")
print(df.head())
print("Datasets shape:")
print(df.shape)
print("Datasets information:")
print(df.info())

X = df.drop("target", axis=1)
y = df["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #tra ve NumPy array
X_test_scaled = scaler.transform(X_test)

print("X_train shape:", X_train_scaled.shape)
print(X_train_scaled[:10]) #numpy

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

from sklearn.model_selection import train_test_split

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test)
print("Actual: ")
print(y_test.values)
print("Predicted: ")
print(y_pred)

#Danh gia model Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))


