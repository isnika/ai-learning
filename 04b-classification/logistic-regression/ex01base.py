'''
BẢN CODE THỨ 2 Ở DƯỚI SẼ ĐẦY ĐỦ HƠN BẢN 1 ĐẦU
Bài tậo cơ bản thực hành
Dự đoán sinh viên có đậi hay không đậu dựa trên số giờ học

Sự khác biệt giữa
predict()
--> y_pred = model.predict(X_test) ==> output dạng [ 0 1 1 0] class
predict_proba()
--> y_prob = model.predict_proba(X_test)
==> output [[0.82 0.18]
 [0.12 0.88]
 [0.20 0.80]
 [0.91 0.09]]

scikit-learn định nghĩa predict_proba() là xác suất của từng class, và trong binary case cột thứ hai là \(P(y=1|X)\).

Muốn lấy probability của class 1:
y_prob = model.predict_proba(X_test)[:, 1]
'''

import numpy as np

X=np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import  LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pre = model.predict(X_test)
print("Actual: ")
print(y_test)
print("Predicted: ")
print(y_pre)

# Đánh giá Accuracy ( độ chính xác)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pre)
print("Accuracy: ", accuracy)  #Nhưng chưa được kết luận model tốt chỉ dựa vào Accuracy.

#Confusion Matrix ( Ma tran nham lan)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pre)
print("Confusion matrix: ")
print(cm)

#Precision
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pre)
print("Precision: ", precision)

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pre)
print("Recall: ", recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pre)
print("F1: ", f1)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pre))

'''
BẢN CODE THỨ 2 NÀY ĐẦY ĐỦ HƠN CHO CODE MACHINE L 

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# 1. Dataset
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1
])



# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# 3. Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create Model
model = LogisticRegression(
    random_state=42
)

# 5. Training
model.fit(
    X_train_scaled,
    y_train
)

# 6. Prediction

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 7. Evaluation

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)


print("Actual:")
print(y_test)

print("\nPredicted:")
print(y_pred)

print("\nProbability:")
print(y_prob)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print("ROC-AUC  :", auc)

print("\nCoefficient:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)
'''
