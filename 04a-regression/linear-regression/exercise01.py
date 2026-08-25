'''
Bài 16 — Linear Regression không dùng sklearn
Cho:
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
Hãy tìm một phương trình:
y=wx+b
sao cho mô hình dự đoán gần với dữ liệu nhất.
Sau đó viết chương trình:
Nhập x
↓
Model
↓
In ra y dự đoán

thi vien du dung: from statistics import LinearRegression

'''
feature = [1,2,3,4,5]
target = [2,4,6,8,10]

# find w and b
n = len(feature)

w = (n * sum(feature[i] * target[i] for i in range(n)) - sum(feature) * sum (target)) / \
    (n * sum(feature[i] ** 2 for i in range(n)) - sum(feature) ** 2)

b = (sum(target) - w * sum(feature)) / n

print("w= ", w)
print("b= ", b)

# nhap x
x_feature = float (input("Enter x: "))
#Nhap model: y = wx + b
y_predict = w * x_feature + b
#In ket qua
print("y-hat (Giá trị y dự đoán (Predict là: ", y_predict)
# Cho thấy tỉ lệ loss = 0 --> Model dự đoán chính xác