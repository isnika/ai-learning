'''
Xây dựng mô hình Polynomial Regression để dự đoán giá nhà tại California dựa trên các đặc điểm của khu vực.
'''
#data use
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load dataset
data = fetch_california_housing(as_frame=True) #yêu cầu Scikit-learn trả dữ liệu dưới dạng Pandas DataFrame/Series thay vì NumPy array.
df = data.frame
print("First 5 lines:")
print(df.head())
print("Dataset shape:")
print(df.shape)
print("Dataset imformation: ")
print(df.info())

#xac dinh  X, y tu data sets
X = df.drop(["MedHouseVal"], axis=1)
y = df["MedHouseVal"]

print("\nX shape: ", X.shape)
print("\ny shape: ", y.shape)

print("\nFeatures: ")
print(X.columns)

print("\nTarget: ")
print("MedHouseval")

# division train, tesst
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% for test
    random_state=42
)
print("\nTrain shape: ", X_train.shape)
print("\nTest shape: ", X_test.shape)
print("\nTrain target: ", y_train.shape)
print("\nTest target: ", y_test.shape)

# polynomial regression
poly = PolynomialFeatures ( degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("\nOriginal features:", X_train.shape[1])
print("Polynomial features:", X_train_poly.shape[1])


#train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

#predict
y_pred = model.predict(X_test_poly) #dùng model Polynomial Regression đã train để dự đoán giá trị y trên tập test.

print("Actual: ")
print(y_test.values)
print("Predicted: ")
print(y_pred)

#comment model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("----- Danh gia model -----")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R2: ", r2)

#compare
print("----- Compare -----")
for actual, predicted in zip(
        y_test.head(10),
        y_pred[:10]
):
    print(f"Actual: {actual: .4f}"
          f"| Predicted: {predicted: .4f}")

# draw a chart
plt.figure(figsize=(8,6))

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

plt.title("Polynomial Regression - California Housing")
plt.grid(True)
plt.savefig(
    "ex02-polynomial-regression.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
'''
DANH GIA MODEL

'''