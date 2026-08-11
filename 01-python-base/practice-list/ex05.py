'''
Đề bài: Nhập một danh sách và một số N. Đếm xem N xuất hiện bao nhiêu lần.
'''

numbers = list(map(int, input("Enter list numbers: ").split()))
n = int(input("Enter N: "))

count = 0

for x in numbers:
    if x == n:
        count += 1

print(f"{n} appears {count} times")