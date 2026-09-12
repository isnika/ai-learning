'''
# XGBOOST CLASSIFICATION
# Dataset: Breast Cancer Wisconsin
(Breast Cancer Wisconsin (Diagnostic) (WDBC), lấy từ UCI.)

Mục tiêu là dự đoán một khối u ở vú (breast mass) là:

Malignant (M) → ác tính
Benign (B) → lành tính

'''
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

from xgboost import XGBClassifier

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

print("First 5 line: ")
print(df.head())

#eda
print("\nDataset shape: ")
print(df.shape)

print("\nDataset imformation: ")
print(df.info())

X = df.drop(["target"], axis=1)
y = df["target"]

print("\nX shape: ")
print(X.shape)
print("\ny shape: ")
print(y.shape)

print("\n----- THONG TIN DATASETS -----")
print("\nStatistical summary:") # tom tat thong ke
print(df.describe())
print("\nTarget Distribution") #phan bo ket qua
print(df["target"].value_counts())
print("\nTarget Percentage")
print(
    df["target"]
    .value_counts(normalize=True) #normalize=True không trả về số lượng, mà trả về tỷ lệ.
    .mul(100) #multiply 100 Tức là nhân các tỷ lệ với 100:
    .round(2) #Làm tròn đến 2 chữ số thập phân
    )

#missing values check
missing = df.isnull().sum()

print("\nMissing values:")
print(missing)
print("\nTotal missing values:")
print(missing.sum())
missing_columns = missing[missing > 0]
if len(missing_columns) == 0:
    print("No missing values found.")
else:
    print("Columns containing missing values:")
    print(missing_columns)

#categorical features check
categorial_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("Categorical columns:", list(categorial_columns))

if len(categorial_columns) == 0:
    print("No categorical columns found.")
