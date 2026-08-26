'''
Bài 18 — Linear Regression bằng sklearn
Cho dataset:
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]
Sử dụng:
LinearRegression
để:
Train model.
In w.
In b.
Dự đoán x = 6.
Tính MSE: Sai số bình trương trung bình
Tính R²: Model độ giải thích biến thiên của y
'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#Date set
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

#Tạo model
model = LinearRegression()
#train model
model.fit(X, y)
# in w
print("Giá trị w = ", model.coef_[0])
#in b
print("Giá trị b = ", model.intercept_)
#Dự đoán x =6
y_pred = model.predict([[6]])
print("Dự đoán y-hat khi x = 6: ",y_pred[0])

#Dự đoán toàn bộ dữ liệu
y_predict = model.predict(X)

#Tính MSE
mse = mean_squared_error(y, y_predict)
print("MSE = ", mse)
#Tính r^2: R^2 = 1 -((sum(y real - y-pred)^2 / (sum(y real - y trung bình)^2)
r2 = r2_score(y, y_predict)
print("R^2 = ", r2)
