"""
So sánh các mô hình Regression trên California Housing Dataset.

Models:
1. Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Decision Tree Regression
"""

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


 
# 1. LOAD DATASET
 

data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

print("Dataset:")
print(X.head())

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


 
# 2. TRAIN / TEST SPLIT
 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


 
# 3. FUNCTION ĐÁNH GIÁ MODEL
 

def evaluate_model(model_name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(model_name)
    print(f"{'=' * 50}")

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return mae, mse, rmse, r2


 
# 4. LINEAR REGRESSION
 

# Chỉ sử dụng 1 feature: MedInc
X_train_linear = X_train[["MedInc"]]
X_test_linear = X_test[["MedInc"]]

linear_model = LinearRegression()

linear_model.fit(
    X_train_linear,
    y_train
)

y_pred_linear = linear_model.predict(
    X_test_linear
)

linear_results = evaluate_model(
    "1. Linear Regression",
    y_test,
    y_pred_linear
)


 
# 5. MULTIPLE LINEAR REGRESSION
 

# Sử dụng tất cả 8 features
multiple_model = LinearRegression()

multiple_model.fit(
    X_train,
    y_train
)

y_pred_multiple = multiple_model.predict(
    X_test
)

multiple_results = evaluate_model(
    "2. Multiple Linear Regression",
    y_test,
    y_pred_multiple
)


 
# 6. POLYNOMIAL REGRESSION
 

# Sử dụng tất cả features
# Degree = 2
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)

y_pred_poly = poly_model.predict(
    X_test_poly
)

poly_results = evaluate_model(
    "3. Polynomial Regression",
    y_test,
    y_pred_poly
)


 
# 7. DECISION TREE REGRESSION
 

tree_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

y_pred_tree = tree_model.predict(
    X_test
)

tree_results = evaluate_model(
    "4. Decision Tree Regression",
    y_test,
    y_pred_tree
)


 
# 8. SO SÁNH CÁC MODEL

results = pd.DataFrame(
    [
        linear_results,
        multiple_results,
        poly_results,
        tree_results
    ],
    columns=[
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],
    index=[
        "Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression",
        "Decision Tree Regression"
    ]
)

print("\n\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results)


 
# 9. TÌM MODEL TỐT NHẤT

best_model = results["R2"].idxmax()
print("\nBest model based on R²:")
print(best_model)