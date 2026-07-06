"""
Bài 4
Tạo ma trận 2 chiều ngẫu nhiên 5x3 (5 học sinh, 3 môn) bằng np.random.randint(0, 11, size=(5,3))
Tính điểm trung bình theo từng học sinh (theo hàng) và theo từng môn (theo cột) — gợi ý dùng axis=0 / axis=1
"""

"""
    mON HOC 
hs:

scores = np.array([
    [8, 7, 10],
    [5, 9, 6],
    [10, 4, 8],
    [7, 6, 9],
    [3, 10, 5]
])
column_3 = scores[:, 2]

print(column_3)
"""



import numpy as np

# Tạo ma trận điểm ngẫu nhiên
array_score = np.random.randint(0, 11, size=(5, 3))

print("Scores:")
print(array_score)

# Trung bình theo từng học sinh (theo hàng)
average_student = np.mean(array_score, axis=1)

print("\nAverage of each student:")
print(average_student)

# Trung bình theo từng môn (theo cột)
average_subject = np.mean(array_score, axis=0)

print("\nAverage of each subject:")
print(average_subject)