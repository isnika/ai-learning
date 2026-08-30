'''
Đề tài: Dự đoán giá nhà California bằng Decision Tree Regression
'''

import numpy as np
import pandas as pd

# write dataset
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

print("First 5 lines: ")
print(df.head())
print("Dataset shape: ")
print(df.shape)
print("Dataset information: ")
print(df.info())

#split dataset

X= df.drop(["MedHouseVal"], axis=1)
y = df["MedHouseVal"]

print("X shape: ", X.shape)
print("y shape: ", y.shape)
print("\nFeature columns: ")
print(df.columns)
print("\n Target columns: ")
print("MedHouseVal")

# division train / tesst
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state = 42
)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

#decision tree
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth = 3, # do sau toi da ( dang tesst toi so nay thi thay cang cao du doan cang xuong(20))
    random_state=42
)

#train model
model.fit(X_train, y_train)
#du doan
y_pred = model.predict(X_test)
print("Actual: ", y_test.values)
print("Predicted: ", y_pred)

# danh gia model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("---- DANH GIA MODEL ----")
print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", rmse)
print("R2: ", r2)


# ve cay
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree Regression - California Housing")
plt.savefig(
    "ex01_decision_tree_regression.png",
    dpi=300,
    bbox_inches="tight")
plt.show()

#ve chart
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Value")

# Đường dự đoán hoàn hảo
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.savefig("ex01_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()

#xem feature nap anh huong nhieu den du doan

importance = model.feature_importances_

plt.figure(figsize=(10, 6))

plt.bar(X.columns, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance - Decision Tree")

plt.xticks(rotation=45)

plt.savefig("ex01_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()