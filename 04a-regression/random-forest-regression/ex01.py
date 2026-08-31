'''

Đề bài

Sử dụng California Housing Dataset để xây dựng mô hình Random Forest Regression nhằm dự đoán giá nhà dựa trên các đặc điểm của khu vực.
'''
import math

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

print("First 5 lines: ")
print(df.head())
print("Datasets shape: ")
print(df.shape)
print("Datasets information: ")
print(df.info())

X = df.drop(["MedHouseVal"], axis=1)
y = df["MedHouseVal"]

print("X shape: ", X.shape)
print("y shape: ", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100, #dùng để quy định số lượng Decision Tree được tạo ra trong Random Forest
    random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual: " )
print(y_test.values) # bth y_test la 1 pandas series, if use values, se bo chi so index va chi lay gia tri ben trong
print("Predicted: " )
print(y_pred) #y_pred thường là NumPy array duoc tra ve tu y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("----- DANH GIA MODEL -----")
print("mse: ", mse)
print("mae: ", mae)
print("rmse: ", rmse)
print("r2: ", r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

#actual and predicted
plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
)
#perfect prediction line: y =x
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")

plt.title("Random Forest Regression - California Housing Dataset")
plt.grid(True)
plt.savefig("ex01_random_forest_model.png", dpi=300, bbox_inches="tight")
plt.show()