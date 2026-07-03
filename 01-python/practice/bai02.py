"""
Bài 2 — Cho hoc_sinh = {"An": 8.0, "Binh": 4.5, "Chi": 9.5}

In ra tên và điểm của người có điểm cao nhất (dùng vòng lặp, không dùng max() trực tiếp trên dict)

"""

students = {"An": 8.0, "Binh": 4.5, "Chi": 9.5}
highest_name = "An"
highest_score = 8.0

for (name, score) in students.items():
    if score> highest_score:
        highest_name = name
        highest_score = score

print(highest_name, highest_score)