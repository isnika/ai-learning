'''
# XGBOOST CLASSIFICATION
# Dataset: Breast Cancer Wisconsin
(Breast Cancer Wisconsin (Diagnostic) (WDBC), lấy từ UCI.)

Mục tiêu là dự đoán một khối u ở vú (breast mass) là:

Malignant (M) → ác tính
Benign (B) → lành tính

'''
from statistics import correlation

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

from matplotlib import pyplot as plt

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
#numerical features

numerical_columns = df.select_dtypes(
    include=["number"]
).columns
print("\nNumber of numerrical columns:")
print(len(numerical_columns))

print("\nNumerical columns:")
print(numerical_columns)

#correlation
correlation = df.corr(numeric_only=True)
print("\nCorrelation with target: ")
print(correlation["target"]
      .sort_values(ascending=False)
      )

#feature engineering
# Tạo một vài feature mới để thực hành

df["radius_texture_ratio"] = (
    df["mean radius"] /
    df["mean texture"]
)

df["area_radius_ratio"] = (
    df["mean area"] /
    df["mean radius"]
)

print("\nNew features:")
print(
    df[
        [
            "radius_texture_ratio",
            "area_radius_ratio"
        ]
    ].head()
)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify = y
)

print("\nX_train shape: ")
print(X_train.shape)
print("X_test shape")
print(X_test.shape)
print("\ny_train shape: ")
print(y_train.shape)
print("\ny_test shape: ")
print(y_test.shape)

#baseline xgboost
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

print("Baseline model trained successfully")

#baseline prediction
y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test) [:,1] #probability: xac suat

#baseline evaluation
print("\nConfusion matrix:")

baseline_accuracy = accuracy_score(y_test, y_pred)
baseline_precision = precision_score(y_test, y_pred)
baseline_recall = recall_score(y_test, y_pred)
baseline_f1 = f1_score(y_test, y_pred)

print("\nBaseline accuracy:")
print(baseline_accuracy)
print("\nBaseline precision:")
print(baseline_precision)
print("\nBaseline recall:")
print(baseline_recall)

print("\nClasification report:")
print(classification_report(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names,
)

disp.plot()
plt.title("Baseline XGBoost - Confusion matrix")
plt.savefig("confusion_matrix.png")
plt.show()

#cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_score = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
)

print("\nCV scores: ")
for i, score in enumerate(
    cv_score,
    start=1,
):
    print(f"Fold {i}: {score:.4f}")

print(f"\nMean CV accuracy: {cv_score.mean():.4f}")
print(f"Std CB accuracy: {cv_score.std():.4f}") #standard deviation

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [1, 3, 4],
}

print("\nParameter grid:")
print(param_grid)

#grid search + cross validation

grid_search = GridSearchCV(
    estimator= XGBClassifier(
        random_state=42,
    ),

    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=1,
    verbose=2,
)

grid_search.fit(X_train, y_train)

#best parameters
print("\nBest parameters:")
print(grid_search.best_params_)
print("\nBest cv accuracy:")
print(f"{grid_search.best_score_:.4f}")

#best model
best_model = grid_search.best_estimator_
print("\nBest model:")
print(best_model)

#final prediction

y_pred_final = best_model.predict(X_test)
y_proba_final = best_model.predict_proba(X_test)[:,1]

#final evaluation
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_auc = roc_auc_score(y_test, y_proba_final)

print("\nFinal accuracy:")
print(final_accuracy)
print("\nFinal precision:")
print(final_precision)
print("\nFinal recall:")
print(final_recall)
print("\nFinal F1:")
print(final_f1)
print("\nFinal ROC-AUC:")
print(final_auc)

#final classification report
cm_final = confusion_matrix(y_test, y_pred_final)

print("\nFinal Confusion Matrix:")
print(cm_final)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_final,
    display_labels=data.target_names
)

disp.plot()

plt.title(
    "Final XGBoost - Confusion Matrix"
)
plt.savefig("confusion_matrix.png")

plt.show()

#roc - auc curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_proba_final
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {final_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "XGBoost ROC Curve"
)

plt.legend()
plt.savefig("roc_curve.png")

plt.show()


 
# 27. FEATURE IMPORTANCE
 

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)

feature_importance = (
    feature_importance
    .sort_values(ascending=False)
)

print(
    feature_importance.head(15)
)


 
# 28. FEATURE IMPORTANCE VISUALIZATION
 

top_features = (
    feature_importance
    .head(15)
    .sort_values()
)

plt.figure(figsize=(8, 6))

top_features.plot(
    kind="barh"
)

plt.title(
    "Top 15 XGBoost Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()
plt.savefig("feature_importance.png")

plt.show()


 
# 29. TRAIN VS TEST
 

print("\n" + "=" * 70)
print("TRAIN VS TEST")
print("=" * 70)

train_pred = best_model.predict(
    X_train
)

test_pred = best_model.predict(
    X_test
)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

gap = (
    train_accuracy -
    test_accuracy
)

print(
    f"Train Accuracy: {train_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Accuracy Gap  : {gap:.4f}"
)


 
# 30. FINAL SUMMARY
 

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Best Parameters: {grid_search.best_params_}"
)

print(
    f"Best CV Score  : {grid_search.best_score_:.4f}"
)

print(
    f"Test Accuracy  : {final_accuracy:.4f}"
)

print(
    f"Test Precision : {final_precision:.4f}"
)

print(
    f"Test Recall    : {final_recall:.4f}"
)

print(
    f"Test F1        : {final_f1:.4f}"
)

print(
    f"Test ROC-AUC   : {final_auc:.4f}"
)

print(
    f"Train Accuracy : {train_accuracy:.4f}"
)

print(
    f"Accuracy Gap   : {gap:.4f}"
)




