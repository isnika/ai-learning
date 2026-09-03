'''
Decision Tree Classifier – Iris
Yêu cầu:
1. Load Iris Dataset
2. EDA cơ bản
3. Train/Test Split
   test_size = 0.2
   random_state = 42
4. Train DecisionTreeClassifier
5. criterion = gini
6. Predict
7. Accuracy
8. Confusion Matrix
9. Classification Report
10. Visualize Decision Tree
11. Feature Importance
12. Thử:
    max_depth = 1
    max_depth = 2
    max_depth = 3
    max_depth = 5
    max_depth = None
13. So sánh:
    Training Accuracy
    Testing Accuracy
14. Xác định:
    Underfitting
    Good fitting
    Overfitting
Làm thêm:
criterion="gini"
        VS
criterion="entropy"

note: Muoosn x200 dau "-"  ==> print ("-" *200)
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report )


#1. Load Datasets
iris = load_iris(as_frame=True)
df = iris.frame

print("----- LOAD IRIS DATASET -----")
print("First 6 lines:")
print(df.head(6))
print("\nSecond 6 lines:")
print(df.tail(6))

print("\nDatasets shape:", df.shape)
print("Datasets information:")
print(df.info())

X = df.drop("target", axis=1)
y = df["target"]
print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("\nFeature names:")
print(iris.feature_names)
print("Target names:")
print(iris.target_names)

#2. Basic EDA (exploratory data analysis = phân tích khám phá dữ liệu)
print("\n----- EDA -----")

#2.1 Test missing values _ Kiem tra du lieu bi thieu
print("\n1. Missing values: ")
print(df.isnull().sum())

#2.2 Basic statistics - Thong ke co ban
'''
Tóm tắt dataset bằng các con số như số lượng, trung bình, độ phân tán, min/max và các phân vị.
'''
print("2. Descriptive statistics")
print(df.describe())
'''
Ví dụ với sepal length:
Có thể hiểu:
Có 150 mẫu → count = 150
Chiều dài đài hoa trung bình ≈ 5.84 cm → mean
Nhỏ nhất 4.3 cm → min
Lớn nhất 7.9 cm → max
50% dữ liệu ≤ 5.8 cm → 50%
Độ phân tán khoảng 0.83 cm → std

25% → Q1 (phân vị 25)
50% → Q2 / Median (trung vị)
75% → Q3 (phân vị 75)
'''
#2.3 Check the number of each Class - Kiem tra so luong tung class
print("\n3. Class distribution:")
print(y.value_counts().sort_index())

print("\n4. Class distribution with names:")
for class_id, class_name in enumerate(iris.target_names):
    count = ( y == class_id ).sum()
    print(f"{class_id} - {class_name}: {count}")




