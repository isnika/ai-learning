'''
Bài 19 — Bài thực tế cho quán cafe
Tạo dataset:
Số khách	Quảng cáo (k)	Khuyến mãi	Doanh thu (k)
50	100	0	1800
70	200	1	2600
90	300	1	3500
60	100	0	2100
100	400	1	4200
120	500	1	5000
80	200	0	3000
110	400	1	4600
40	100	0	1500
130	500	1	5400
Trong đó:
X₁ = số khách
X₂ = tiền quảng cáo
X₃ = khuyến mãi
y  = doanh thu
Yêu cầu
Bước 1: Chia X và y.
Bước 2: Chia:
80% train
20% test
Bước 3: Train Linear Regression.
Bước 4: In:
w₁
w₂
w₃
bias
Bước 5: Dự đoán doanh thu trên test.
Bước 6: Tính:
MAE
MSE
RMSE
R²
Bước 7: Giả sử ngày mai:
150 khách
500k quảng cáo
Có khuyến mãi
→ Dự đoán doanh thu.
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

'''
X₁ = số khách
X₂ = tiền quảng cáo
X₃ = khuyến mãi
y  = doanh thu
'''
#data set
X = [
    [50,100,0],
    [70,200,1],
    [90,300,1],
    [60,100,0],
    [100,400,1],
    [120,400,1],
    [80,200,0],
    [110,400,0],
    [40,100,0],
    [130,500,1],
]

y = [1800, 2600, 3500, 2100, 4200, 5000, 3000,4600,1500, 5400]
# / 80% train/20%test
# train_test_split() sẽ xáo trộn dữ liệu trước khi chia. rồi mới lấy 80% / 20%.
# su dung thuat toan: pseudo-random.
'''
random_state giống như cách xào bài được cố định.
42 → cách xào A
10 → cách xào B
99 → cách xào C

random_state
     ↓
quy tắc tạo số
     ↓
chuỗi số "ngẫu nhiên"

'''
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # 20% dữ liệu dành cho test ==> mac dinh train = 80%
    random_state=42 # giúp việc xáo trộn này lặp lại được, 42 là seed để khởi tạo bộ sinh số giả ngẫu nhiên.
)

#train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

#in w1, w2, w3, bias
print("w1 = ", model.coef_[0])
print("w2 = ", model.coef_[1])
print("w3 = ", model.coef_[2])
print("bias (b) = ", model.intercept_)

# Dự đoán doanh thu trên test.
y_pred = model.predict(X_test)
print("\nDoanh thu that: ", y_test)
print("\nDoanh thu du doan: ", y_pred)

#tinh mse, mae, rmse, r^2
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE = ", mae)
print("MSE = ", mse)
print("RMSE = ", rmse)
print("R2 = ", r2)

#du doan ngay mai
'''
Giả sử ngày mai:
150 khách
500k quảng cáo
Có khuyến mãi
→ Dự đoán doanh thu.
'''
tomorrow = [[150, 500, 1]]
revenue = model.predict(tomorrow)
print("\nDoanh thu du doan ngay mai: ", revenue)

'''
Ket qua: 

w1 =  36.52173913043502
w2 =  2.1521739130434336
w3 =  -58.695652173913004
bias (b) =  -294.5652173913136
==> Phuong trinh la: y = 36.52x1 + 2.15x2 - 58.7x3 - 294
Trong đó:
X₁ = số khách ==> Thêm 1 khách thì doanh thu dự đoán tăng khoảng: 36.52k
X₂ = tiền quảng cáo (k) ==> Nếu quảng cáo tăng thêm 1k, model dụ đoán doanh thu tăng khoảng: 2.15k
X₃ = khuyến mãi (0 hoặc 1) ==> Tại sao âm? Không có nghĩa khuyến mãi làm doanh thu 
giảm 58.7k một cách chắc chắn
Nó có nghĩa: Sau khi đã giữ số khách và tiền quảng cáo cố định, hệ số mà model ước lượng cho biến 
khuyến mãi là khoảng -58.7k
Dataset của bạn chỉ có 10 dòng, nên hệ số có thể khá nhạy và chưa đủ để kết luận khuyến mãi thực sự làm giảm doanh thu.

Đặc biệt, dữ liệu của bạn có quan hệ giữa các biến khá mạnh: các ngày có khuyến mãi thường cũng có nhiều khách và quảng 
cáo cao. Model phải "chia" ảnh hưởng giữa các feature, nên hệ số có thể hơi khó trực giác.

Kiem tra du doan doanh thu:
Doanh thu that:  [1500, 2600]
Doanh thu du doan:  [1381.52173913 2633.69565217]
==> Sai số:
1500 → 1381.52 
2600 → 2633.70
==> Không quá lớn 

#Tính kiểm tra 
MAE =  76.08695652174049 → Sai lệch trung bình khoảng bao nhiêu?
==> Trung bình model dự đoán lệch khoảng 76k doanh thu trên tập test.

MSE =  7586.2476370514805  → Sai số bình phương trung bình (MSE không có cùng đơn vị với doanh thu)
==> 
RMSE =  87.09906794594005 Sai lệch trung bình theo đơn vị ban đầu
==> RMSE lớn hơn MAE vì RMSE phạt các sai số lớn mạnh hơn.
R2 =  0.9749214954147059 ( Khoảng 97.49%) Model giải thích dữ liệu tốt đến mức nào?
==> Model giải thích được khoảng 97.5% biến thiên của doanh thu trong tập test.
==> Khả năng giải thích dữ liệu rất cao.

Du doan doanh thu ngày mai
Doanh thu du doan ngay mai:  [6201.08695652]
y=36.52(150)+2.15(500)−58.70(1)−294.57 = 6201.09k
Tức khoảng:6201 triệu đồng 

Ket luan:
Model có hiệu suất tốt trên tập test, với R² ≈ 97.49%, MAE ≈ 76.09k và RMSE ≈ 87.10k. 
Tuy nhiên, do dataset rất nhỏ và tập test chỉ có 2 mẫu, cần thêm dữ liệu để đánh giá 
độ tin cậy của model trong thực tế.
'''
