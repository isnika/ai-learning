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
'''