'''
Nhập một số nguyên từ bàn phím.

Nếu là số chẵn in: So chan
Nếu là số lẻ in: So le
'''

n = int(input("Nhập số để kiểm tra: "))

if n % 2 == 0:
    print("Đây là số chẵn.")
else:
    print("Đây là số lẽ.")

'''Bài 2

Nhập 5 số nguyên.

Đếm xem có bao nhiêu số chẵn và bao nhiêu số lẻ.
'''
even_number =0
odd_number =0

for i in range(5):
    n = int(input(f"Nhập số thứ {i + 1}: "))

    if n % 2 == 0:
        even_number += 1
    else:
        odd_number += 1

# In kết quả
print("Số lượng số chẵn:", even_number)
print("Số lượng số lẻ:", odd_number)

'''
Nhập một số N.

In tất cả các số chẵn từ 1 đến N.
'''
N = int(input("Nhập số N > 0: "))
for i in range(1, N+1):
    if i % 2 == 0:
        print(i)