from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_scora
import numpy as np

# Tập dữ liệu mất cân bằng ví dụ
X = np.random.rand(100, 5)
y = np.array([0]*90 + [1]*10)  # 90% lớp 0, 10% lớp 1

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42,
)

