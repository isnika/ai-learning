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

0 → setosa
1 → versicolor
2 → virginica
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
#2.3 Check the number of each Class - Kiem tra so luong tung class (
print("\n3. Class distribution:") #phân bố số lượng của các class
print(y.value_counts().sort_index()) # dem sl moi class, sap xep tang dan
# print(df["target"].value_counts()) # dem sl moi class, sap xep giam dan

print("\n4. Class distribution with names:")
for class_id, class_name in enumerate(iris.target_names): #enumerate() lấy cả vị trí (index) và giá trị
    '''
iris.target_names
       ↓
['setosa', 'versicolor', 'virginica']
       ↓
enumerate()
       ↓
0 → 'setosa'
1 → 'versicolor'
2 → 'virginica'
       ↓
(y == class_id)
       ↓
Đếm số lượng
       ↓
In ra class + tên + số lượng
    '''
    count = ( y == class_id ).sum() #dem so
    print(f"{class_id} - {class_name}: {count}")

#2.4  Visualization: Class distribution - Hinh anh truc phan: Phan bo lop (y)

plt.figure(figsize=(8, 5))
y.value_counts().sort_index().plot(
    kind='bar'
    )

plt.title("Iris Class distribution")
plt.xlabel("Class")
plt.ylabel("Number Of Samples")

plt.xticks(
    ticks = [0, 1, 2],
    labels = iris.target_names,
    rotation=0
)

plt.tight_layout()
plt.savefig("ex02_iris_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

#2.5 Visualization: Feature distributions HInh anh truc quan: Phan bo dac trung(x)

df[iris.feature_names].hist(
    figsize=(12, 8),
    bins=15
)

plt.suptitle("Iris Feature Distributions")
plt.tight_layout()
plt.savefig("ex02_iris_feature_distributions.png", dpi=300, bbox_inches="tight")
plt.show()

#3. Train/test split
print ("----- Train/Test Split -----")
X_train , X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

#4. train decision tree classifier
print("-----Train Decision Tree Classifier-----")
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

#5. Predict
print("----- Predict -----")

# Predict trên TRAIN
y_train_pred = model.predict(X_train)

# Predict trên TEST
y_test_pred = model.predict(X_test)

print("\nFirst 10 actual test values:")
print(y_test.values[:10])

print("First 10 predicted test values:")
print(y_test_pred[:10])

#6. Accuracy
print("----- Accuracy -----")
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score (y_test, y_test_pred)
print("\nTrain Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

#7.Confusion matrix
print("----- Confusion Matrix -----") #ma tran nham lan
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

#visualization Confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names,
)

disp.plot()
plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.savefig("ex02_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

#8. classification report
print("----- Classification Report -----")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=iris.target_names,
)
)

#9. Visualization decision tree

print("----- Visualization Decision Tree -----")

plt.figure(figsize=(18, 12))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    precision=2
)
plt.title("Decision Tree Classifier - Iris")

plt.tight_layout()
plt.savefig("ex02_decision_tree_classifier.png", dpi=300, bbox_inches="tight")
plt.show()

#10.Feature importance
print("----- Feature Importance -----")
fea_importance = pd.DataFrame(
    {
        "Feature": iris.feature_names,
        "Importance": model.feature_importances_,
    }
)
fea_importance = fea_importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nFeature Importance:")
print(fea_importance)

#Visualization feature importance
plt.figure(figsize=(10, 6))
plt.bar(
    fea_importance["Feature"],
    fea_importance["Importance"]
)

plt.title("Decision Tree Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(
    rotation=20,
    ha="right",
)
plt.tight_layout()
plt.savefig("ex02_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

#11. Thu nhieu max depth

print("----- Test Max_Depth -----")
depth = [
    1,
    2,
    3,
    5,
    None
]
result = []

for dept in depth:
    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=dept,
        random_state=42,
    )

    tree_model.fit(X_train, y_train)

    train_pred = tree_model.predict(X_train)
    test_pred = tree_model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    gap = train_acc - test_acc
    result.append(
        {
            "max_depth": dept,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "gap": gap,
        }
    )
#12. comparison table
result_df = pd.DataFrame(result)
print("----- MAX_DEPTH COMPARISON ----- ")
print( result_df.to_string( index=False ) )

#13. Analyze fitting - Phan tich su phu hop
print("----- Fitting Analysis -----")

def analyxe_fit(train_acc, test_acc):
    """ Educational heuristic.
    Underfitting: Train và Test đều thấp.
    Good fitting: Train và Test đều cao và khoảng cách nhỏ.
    Overfitting: Train rất cao nhưng Test thấp hơn đáng kể.
    """
    gap = train_acc - test_acc
    if train_acc < 0.90 and test_acc < 0.90:
        return "Underfitting"
    elif train_acc >= 0.90 and test_acc >= 0.90 and gap < 0.10:
        return "Good fitting"
    elif train_acc >= 0.95 and gap >= 0.10:
        return "Overfitting"
    else:
        return "Needs further analysis" # Can phan tich them nua
result_df["fitting"] = result_df.apply(
    lambda row: analyxe_fit(
        row["train_accuracy"],
        row["test_accuracy"],
    ),
    axis=1
)
print(
    result_df[
        [
            "max_depth",
            "train_accuracy",
            "test_accuracy",
            "gap",
            "fitting",
        ]
    ].to_string(index=False)
)

#14. VISUALIZE TRAINING VS TESTING ACCURACY
print("----- TRAINING VS TESTING ACCURACY -----")

depth_labels = result_df[
    "max_depth"
].astype(str)

plt.figure( figsize=(10, 6) )
plt.plot(
    depth_labels,
    result_df["train_accuracy"],
    marker="o",
    label="Training Accuracy" )
plt.plot(
    depth_labels,
    result_df["test_accuracy"],
    marker="o",
    label="Testing Accuracy" )
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title( "Decision Tree - Training vs Testing Accuracy" )
plt.ylim( 0.5, 1.05 )
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ex02_decision_tree_train_test accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
#15. Gini and entropy
print("----- GINI VS ENTROPY -----")

criteria = [ "gini", "entropy" ]
criterion_results = []
for criterion in criteria:
    tree = DecisionTreeClassifier(
        criterion=criterion,
        random_state=42
    )
    tree.fit( X_train, y_train )
    train_pred = tree.predict( X_train )
    test_pred = tree.predict( X_test )
    train_acc = accuracy_score( y_train, train_pred )
    test_acc = accuracy_score( y_test, test_pred )
    criterion_results.append(
        {
            "criterion": criterion,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "gap": train_acc - test_acc
        }
    )

    criterion_df = pd.DataFrame( criterion_results )
    print("\nGini vs Entropy:")
    print( criterion_df.to_string( index=False ) )

# 16. FINAL SUMMARY
print("----- FINAL SUMMARY ------")
best_index = result_df[ "test_accuracy" ].idxmax()
best_model = result_df.loc[ best_index ]

print( f""" 
Best max_depth: 
{best_model["max_depth"]} 
Training Accuracy: 
{best_model["train_accuracy"]:.4f} 
Testing Accuracy: 
{best_model["test_accuracy"]:.4f} 
Accuracy Gap: 
{best_model["gap"]:.4f} 
Fitting: 
{best_model["fitting"]} """ )

print("\nGini vs Entropy:")
print( criterion_df.to_string( index=False ) )
print("\nProgram finished successfully!")