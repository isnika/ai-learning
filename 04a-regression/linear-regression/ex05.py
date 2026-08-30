'''
Dự đoán giá nhà bằng Linear Regression
Đề bài
Sử dụng California Housing Dataset để xây dựng mô hình
Linear Regression dự đoán giá nhà dựa trên các thông tin về khu vực dân cư.
https://scikit-learn.org/stable/datasets/real_world.html?: fetch_california_housing()
Yêu cầu
1. Load và khám phá dataset.
2. Xác định Feature (X) và Target (y).
3. Chia dữ liệu thành:
80% Training
20% Testing
random_state=42
4. Xây dựng và huấn luyện mô hình Linear Regression.
5. In Weight và Bias.
6. Dự đoán giá nhà trên tập Test.
7. Đánh giá model bằng:
MAE
MSE
RMSE
R²
8. Chọn một mẫu trong tập Test và so sánh:
Giá thực tế
Giá dự đoán
Sai lệch
9. Nhận xét model dựa trên các chỉ số đánh giá.
Kết quả cần đạt: Hoàn thành quy trình:
Dataset → X/y → Train/Test → Linear Regression → Prediction → MAE/MSE/RMSE/R² → Đánh giá Model.
1. Import thư viện
2. Load dataset
3. Khám phá dữ liệu
4. Tạo X, y
5. Chia Train/Test
6. Tạo Linear Regression
7. Train model
8. Lấy Weight và Bias
9. Prediction
10. MAE
11. MSE
12. RMSE
13. R²
14. Dự đoán một mẫu
15. Phân tích kết quả
'''

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np
import pandas as pd

#load dataset = tap du lieu
data = fetch_california_housing(as_frame=True) #data là một object chứa dataset California Housing

df = data.frame #date.frame là toàn bộ dataset ở dạng Pandas DataFrame.
print("First 5 lines: ")
print(df.head())
print("Dataset shape: ")
print(df.shape)
print("Column names: ")
print(df.columns)
print("Dataset imformation: ")
print(df.info())

#xac dinh X va y từ toàn bộ dữ liệu bảng
X = df.drop("MedHouseVal", axis=1) # xóa cột Med...
# drop là xóa ==> Ở đây xóa cột "MedHouseVal" khỏi df
#pandas quy ước axis = 0 (hàng, row), axis = 1 (cột, column)
y = df["MedHouseVal"] # lấy riêng cột Med... = target

print("\nX shape: ", X.shape) # cho biết kích thước dữ liệu
print("y shape: ", y.shape)
print("\nFeatures: ") #Dùng để xem X đang có những Feature nào.
print(X.columns)

print("\nTargets: ") #Dòng này chỉ đơn giản là in ra tên Target.
print("MedHouseVal")

#chia 80 train/ 20 test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # 20% dữ liệu dành cho test ==> mac dinh train = 80%
    random_state=42
)
print("\n ----- Dataset shape -----")
print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

#tao model
model = LinearRegression()
#train model
model.fit(X_train, y_train)
#tinh intercept and coefificient or weight and bias
print("\n ----- Weight (Coefficients) -----")
for feature, weight in zip(X.columns, model.coef_):
    print(f"{feature}: {weight:.6f}")

print("\n ----- Bias (Intercept) -----")
print(f"{model.intercept_:.6f}")

#du doan prediction

y_pred = model.predict(X_test)
print("----- 10 first predictions ----- ")
result = pd.DataFrame(
    {
        "Actual": y_test.values[:10],
        "Predicted": y_pred[:10],
    }
)

print(result)

#danh gia model
print("----- Model Review -----")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

#dua doan 1 mau
sample = X_test.iloc[[0]]

actual_value = y_test.iloc[0]

predicted_value = model.predict(sample)[0]

error = actual_value - predicted_value

print("----- Predicting 1 sample ----- ")
print("\nHouse information: ")

for feature in X.columns:
    print(f"{feature}: {sample.iloc[0][feature]}")
print(f"\nGia thuc te: {actual_value: .6f}")
print(f"Gia du doan: {predicted_value: .6f}" )
print(f"Sai lech: {error: .6f}")

#NHAN XET
print("\n----- COMMENT ----- ")
print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

print("\nTarget của dataset có đơn vị là $100,000.")
print("Vì vậy MAE/RMSE cũng đang được tính theo đơn vị $100,000.")



'''
Đánh giá tổng quan model

Dataset: 20,640 mẫu
Feature: 8
Train: 16,512 mẫu (80%)
Test: 4,128 mẫu (20%)

20,640 dữ liệu
      ↓
X = 8 Features
y = MedHouseVal
      ↓
80% Train / 20% Test
      ↓
Linear Regression
      ↓
y_pred
      ↓
Đánh giá

Kết quả: ( Tính theo target có đơn vị 100.000 USB)
MAE  = 0.5332 #Trung bình dự đoán của model lệch khoảng 53.320 USD so với giá thực tế.
==> Mức sai số không đáng kể
MSE  = 0.5559 #Trung bình của (giá thật - giá dự đoán)²
==> MSE phạt mạnh những dự đoán sai nhiều.
==> Không đáng giá vì chủ yếu dùng để đánh giá và so sánh model.
RMSE = 0.7456 #Mức sai số dự đoán điển hình theo RMSE khoảng 74.560 USD.
==>  RMSE > MAE ==> Điều này cho thấy trong dữ liệu có một số trường hợp model dự đoán sai khá xa, 
vì RMSE nhạy với các sai số lớn.
R²   = 0.5758 #Đây là chỉ số chú ý nhất.
==> Model Linear Regression giải thích được khoảng 57,58% sự biến thiên của giá nhà trong tập test.
==> Còn khoảng:
100% - 57.58%
= 42.42%chưa được model giải thích.c

Mức độ đánh giá:
R² = 1.0      → rất tốt
R² gần 1      → tốt
R² ≈ 0.58     → trung bình/khá
R² gần 0      → khả năng giải thích thấp
R² < 0        → model rất kém

'''




