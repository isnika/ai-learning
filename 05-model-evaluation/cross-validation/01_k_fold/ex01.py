import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

#2. Train/test split ( Giu nguyen tap test de danh gia cuoi cung)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)
#3. Chuan hoa du lieu Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#4. Ap dung K-fold cross validation tren tap Train de danh gia mo hinh
kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
knn = KNeighborsClassifier(
    n_neighbors=5,
)
cv_scores = cross_val_score(
    knn,
    X_train_scaled,
    y_train,
    cv = kf,
    scoring = 'accuracy',
)

print("Diem accuracy tung fold: ", cv_scores)
print(f"Accuracy trung binh (K-Fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}) ")

#5. Tìm k diem tot nhat bang K_fold ( thu nhieu gia tri k)

print("----- Tim K toi uu ------")
best_k, best_score = 1,0
for k in range( 1,21):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn_k,
        X_train_scaled,
        y_train,
        cv = kf,
        scoring = 'accuracy',
    )
    mean_score = scores.mean()
    print(f"k={k}, accuracy trung binh = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\n>>> k tot nhat: {best_k} voi accuracy CV = {best_score:.4f}")

#6.train lai mo hinh voi k tot nhat tren toan bo tap train
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)

#7. Danh gia tren tap Test (du lieu chua tung thay)
y_pred = final_knn.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print("----- Ket qua cuoi cung tren tap test -----")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Bao cao chi tiet: ")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
print("Ma tran nham lan (Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


