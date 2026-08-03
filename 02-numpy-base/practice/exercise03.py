"""
Bài 3
python import numpy as np
diem = np.array([7.5, 8.0, 4.5, 9.0, 6.0, 3.5, 10.0])

Tính trung bình, độ lệch chuẩn (np.std), điểm cao nhất, thấp nhất
Tạo mảng mới diem_cong = điểm cộng thêm 0.5 cho mỗi người, nhưng không vượt quá 10
Đếm bằng NumPy (không vòng lặp) xem có bao nhiêu bạn ≥ 8.0

 """
import numpy as np
score = np.array([7.5, 8.0, 4.5, 9.0, 6.0, 3.5, 10.0])

print("Avarage score:", score.mean())

variance_population = np.std(score,ddof=0)
print("Variance:", variance_population)
variance_sample = np.std(score,ddof=1)
print("Variance sample:", variance_sample)

max_score = np.max(score)
print("Max score:", max_score)
min_score = np.min(score)
print("Min score:", min_score)

bonus_scores = np.minimum(score + 0.5, 10)

print("Bonus scores:", bonus_scores)

""" thu cong de hieu ro thuat toan:
bonus_scores = []

for s in score:
    bonus_scores.append(min(s + 0.5, 10))

bonus_scores = np.array(bonus_scores)

print(bonus_scores)
"""

count = np.sum(score >= 8.0)
#[False, True, False, True, False, False, True] t=1, f=0

print("Number of students >= 8.0:", count)

"""
import numpy as np

score = np.array([7.5, 8.0, 4.5, 9.0, 6.0, 3.5, 10.0])

count = 0

for s in score:
    if s >= 8.0:
        count += 1

print("Number of students >= 8.0:", count)
"""