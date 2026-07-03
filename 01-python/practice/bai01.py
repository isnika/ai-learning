"""
Bài 1 — Cho danh sách điểm: diem = [7.5, 8.0, 4.5, 9.0, 6.0, 3.5, 10.0]

a. Viết hàm đếm số bạn đạt (≥5) và số bạn không đạt
b. Tính điểm trung bình bằng vòng lặp (chưa dùng thư viện)

"""
#a
def dem_dat_khong_dat(scores):
    passed_count = 0
    failed_count = 0
    for score in scores:
        if score >= 5:
            passed_count += 1
        else:
            failed_count += 1
    return passed_count, failed_count

passed, failed = dem_dat_khong_dat(scores)
print("Number passed:", passed, "\nNumber Failed:", failed)
#b

def tinh_trung_binh(scores):
    total = 0
    for score in scores:
        total += score
    return total / len(scores)

print("Average of scores:", tinh_trung_binh(scores))