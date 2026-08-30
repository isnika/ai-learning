from math import degrees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt



#write dataset
df = pd.read_csv("cafe_sales.csv")
print(df.head()) #xem 5 dong dau tien muon x dong = head(x)
print(df.info()) #kiem tra cau truc data set

#split X, y (so khach, doanh thu)
X = df[["customers"]] #X PHẢI LÀ DATA FRAME 2 CHIỀU ( NHẤN MẠNH)
y = df["revenue"]

#division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# creat polynomial features
poly = PolynomialFeatures(degree = 2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(X_train_poly)
'''
in ra:
1
customers
customers²

1
x   = 75
x²  = 75² = 5625
'''
#train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

#predict
y_pred = model.predict(X_test_poly)
print("Actual: ")
print(y_test.values)
print("Predicted: ")
print(y_pred)

#comment model
mse =mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse= math.sqrt(mse)
r2 =r2_score(y_test, y_pred)

print("----- DANH GIA MODEL -----")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

#draw a chart
X_curve = np.linspace(
    X["customers"].min(), #lấy số khách nhỏ nhất
    X["customers"].max(), #lấy số khách lớn nhất
    100 #tạo 100 giá trị nằm đề từ 10 đến 80
).reshape(-1, 1) # biến dữ liệu thành dạng 2 chiều:

X_curve_poly = poly.transform(X_curve) # transform() biến 100 giá trị khách hàng
                                       # thành dạng Polynomial mà model hiểu được.
y_curve = model.predict(X_curve_poly) # lấy khách và đón doanh thu tương ứng

plt.scatter(X,y) #chính là các dấu "chấm" trên hình in ra, mỗi chấm đại diện cho dữ liệu thực tế (trong datadet)
plt.plot(X_curve,y_curve) #chính là đường cong trên hình, là duẽ liệu dự đoán từ thuật toán polynomial regression
plt.xlabel("Customers") # đặt tên cho mục ngang
plt.ylabel("Revenue") # đặc tên cho trục dọc
plt.title("Polynomial Regression - Cafe Revenue") # đặt tiêu đề cho biểu đồ
plt.savefig("ex01_polynomial_regression.png", dpi=300, bbox_inches="tight")
plt.show() # hiênr thị biểu đồ

#predict 90 customers
new_data = np.array[90]

new_data_poly = poly.transform(new_data)
prediction = model.predict(new_data_poly)
print("\nPredicted Revenue for 90 customers: ", prediction[0], "VND")



'''
MAE : 31.41574165352826
=> lệch khoảng 31.42 đơn vị so với giá trị thực tế (MAE càng gần 0 càng tốt.)
MSE : 1314.7850047100978
RMSE: 36.25996421275258
=> RMSE > MAE ==> Điều này cho thấy model có một số dự đoán sai nhiều hơn bình thường, 
nhưng không quá nghiêm trọng.
R2  : 0.9971516789326038
=> Polynomial Regression giải thích được khoảng 99.72% sự biến thiên của biến mục tiêu.
'''







