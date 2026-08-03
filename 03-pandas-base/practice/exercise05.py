"""
Tạo file diem_hs.csv với nội dung:
ten,mon,diem
An,Toan,8.5
An,Van,7.0
Binh,Toan,4.5
Binh,Van,6.0
Chi,Toan,9.5
Chi,Van,8.0
Bài 5
import pandas as pd
df = pd.read_csv("diem_hs.csv")

In ra danh sách học sinh có điểm ≥ 8
Tính điểm trung bình mỗi môn (groupby)
Tính điểm trung bình mỗi học sinh, sắp xếp giảm dần
Thêm cột xep_loai: ≥8 là "Giỏi", 5-7.9 là "Khá", dưới 5 là "Yếu" (gợi ý: dùng apply hoặc pd.cut)
Xuất kết quả ra ket_qua.csv

"""
"""
from openpyxl import Workbook
import pandas as pd

data = [
    ["An","Toan",8.5],
    ["An","Van",7.0],
    ["Binh","Toan",4.5],
    ["Binh","Van",6.0],
    ["Chi","Toan",9.5],
    ["Chi","Van",8.0],
]
df=pd.DataFrame(data,columns=["ten","mon","diem"])
path="/mnt/data/diem_hs.csv"
df.to_csv(path,index=False)
print(path)

"""

import pandas as pd

df = pd.read_csv("diem_hs.csv")


#In ra danh sách học sinh có điểm ≥ 8
print(" Học sinh có điểm >= 8 ")
print(df[df["diem"] >= 8])

print(" Học sinh có điểm >= 8 ")

for index, row in df.iterrows():
    if row["diem"] >= 8:
        print(row["ten"], row["mon"], row["diem"])


#Tính điểm trung bình mỗi môn (groupby)
print ("ĐIỂM TRUNG BÌNH MỖI MÔN")
print(df.groupby("mon")["diem"].mean())

tong = {}
dem = {}

for i in range(len(df)):
    mon = df.loc[i, "mon"]
    diem = df.loc[i, "diem"]

    if mon not in tong:
        tong[mon] = 0
        dem[mon] = 0

    tong[mon] += diem
    dem[mon] += 1

print("Điểm trung bình mỗi môn:")
for mon in tong:
    print(mon, ":", tong[mon] / dem[mon])

#Tính điểm trung bình mỗi học sinh, sắp xếp giảm dần
tb_hs = df.groupby("ten")["diem"].mean().sort_values(ascending=False)
print(" điểm trung bình mỗi học sinh")
print(tb_hs)


tong = {}
dem = {}

# Bước 1: Tính tổng điểm và số môn của từng học sinh
for i in range(len(df)):
    ten = df.loc[i, "ten"]
    diem = df.loc[i, "diem"]

    if ten not in tong:
        tong[ten] = 0
        dem[ten] = 0

    tong[ten] += diem
    dem[ten] += 1

# Bước 2: Tính điểm trung bình
tb = {}

for ten in tong:
    tb[ten] = tong[ten] / dem[ten]

# Bước 3: Sắp xếp giảm dần (thủ công)
ds = list(tb.items())

for i in range(len(ds)):
    for j in range(i + 1, len(ds)):
        if ds[i][1] < ds[j][1]:
            ds[i], ds[j] = ds[j], ds[i]

# Bước 4: In kết quả
print("Điểm trung bình mỗi học sinh:")

for ten, diem_tb in ds:
    print(ten, ":", diem_tb)

#Thêm cột xep_loai: ≥8 là "Giỏi", 5-7.9 là "Khá", dưới 5 là "Yếu" (gợi ý: dùng apply hoặc pd.cut)
def xep_loai(diem):
    if diem >= 8:
        return "Giỏi"
    elif diem >= 5:
        return "Khá"
    else:
        return "Yếu"

# Thêm cột xếp loại
df["xep_loai"] = df["diem"].apply(xep_loai)
print(df)
df.to_csv("ket_qua.csv", index=False, encoding="utf-8-sig")

print("Đã lưu file ket_qua.csv")