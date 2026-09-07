from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Tập dữ liệu mất cân bằng ví dụ
X = np.random.rand(100, 5)
y = np.array([0]*90 + [1]*10)  # 90% lớp 0, 10% lớp 1

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42,
)

accuracies = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = RandomForestClassifier(
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)

    print(f"Fold {fold + 1} Phan bo lop trong tap validation ={np.bincount(y_val)}, do chinh xac = {acc:.3f}")

print(f"\n DO chinh xac trung binh: {np.mean(accuracies):.3f} +- {np.std(accuracies):.3f}")