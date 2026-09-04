"""
Random Forest Classification - Iris Dataset

Flow:
1. Load Dataset
2. EDA / Cleaning
3. Train/Test Split
4. Random Forest
   ├── Bootstrap Sampling
   ├── Tree 1
   ├── Tree 2
   ├── Tree 3
   └── ...
5. Aggregate Results
6. Prediction
7. Evaluation
   ├── Accuracy
   ├── Confusion Matrix
   └── Classification Report
"""

 
# 1. IMPORT LIBRARIES
 

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


 
# 2. LOAD DATASET
iris = load_iris(as_frame=True)
# Feature data
X = iris.data
# Target
y = iris.target

# 3. EDA - BASIC EXPLORATION

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(X.head())

print("\nDataset shape:")
print(X.shape)

print("\nFeature names:")
print(X.columns.tolist())

print("\nTarget values:")
print(y.unique())

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nBasic statistics:")
print(X.describe())


 
# 4. CLEANING
 

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Check missing values
print("\nMissing values:")
print(X.isnull().sum())

print("\nMissing values in target:")
print(y.isnull().sum())

# Check duplicated rows
print("\nNumber of duplicated rows:")
print(X.duplicated().sum())

# Iris dataset has no missing values,
# so no imputation is necessary.

# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nX_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


 
# 6. CREATE RANDOM FOREST MODEL
 

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)


 
# 7. TRAIN MODEL
 

model.fit(X_train, y_train)


 
# 8. INFORMATION ABOUT RANDOM FOREST
 

print("\n" + "=" * 60)
print("RANDOM FOREST INFORMATION")
print("=" * 60)

print("\nNumber of trees:")
print(model.n_estimators)

print("\nBootstrap sampling:")
print(model.bootstrap)

print("\nCriterion:")
print(model.criterion)

print("\nMax features:")
print(model.max_features)


 
# 9. PREDICTION
 

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


 
# 10. SHOW ACTUAL VS PREDICTED
 

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)

print("\nFirst 10 actual test values:")
print(y_test.values[:10])

print("\nFirst 10 predicted test values:")
print(y_test_pred[:10])


 
# 11. ACCURACY
 

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\n" + "=" * 60)
print("ACCURACY")
print("=" * 60)

print("\nTrain Accuracy:", train_accuracy)
print("Test Accuracy :", test_accuracy)


 
# 12. CONFUSION MATRIX
 

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


 
# 13. CLASSIFICATION REPORT
 

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=iris.target_names
    )
)


 
# 14. VISUALIZE CONFUSION MATRIX
 

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


 
# 15. FEATURE IMPORTANCE
 

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)


 
# 16. VISUALIZE FEATURE IMPORTANCE
 

plt.figure(figsize=(8, 5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


 
# 17. PREDICT A NEW SAMPLE
 

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(new_flower)

prediction_probability = model.predict_proba(new_flower)


print("\n" + "=" * 60)
print("NEW SAMPLE PREDICTION")
print("=" * 60)

print("\nNew flower:")
print(new_flower)

print("\nPredicted class:")
print(prediction)

print("\nPredicted species:")
print(iris.target_names[prediction[0]])

print("\nPrediction probability:")
print(prediction_probability)