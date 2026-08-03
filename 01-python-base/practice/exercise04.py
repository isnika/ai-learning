'''
Nhập một số nguyên.

Kiểm tra xem số đó có phải số nguyên tố không.
Số nguyên tố là số tự nhiên lớn hơn 1 và chỉ có đúng 2 ước số dương, đó là:

1
Chính nó
'''

n = int(input("Nhập số n>0 : "))

if n <= 1:
    print (" n không phải là số nguyên tố")
else:
    la_so_nguyen_to = True
    # Kiểm tra xem n có chia hết cho số nào từ 2 đến √n hay không.
    for i in range (2, int(n**0.5) + 1):
        if n%i ==0:
            la_so_nguyen_to = False
            break
    if la_so_nguyen_to:
        print("n là số nguyên tố")
    else:
        print("n không phải là số nguyên tố")

'''
Nhập số N.

In tất cả các số nguyên tố từ 2 đến N.
'''
N = int(input("Nhập số nguyên N >0: "))

for i in range(2, N+1):
    la_so_nguyen_to2 = True

    for j in range (2, i):
        if i % j == 0:
            la_so_nguyen_to2 = False
            break
    if la_so_nguyen_to2:
        print(i)
'''
Đếm có bao nhiêu số nguyên tố trong đoạn từ A đến B.
'''
A = int(input("Nhập khoảng A đầu: "))
B = int(input("Nhập khoảng B cuối: "))
dem =0

for i in range(A, B+1):
    if i<2:
        continue

    la_so_nguyen_to3 = True

    for j in range (2, i): #for j in range(2, int(i ** 0.5) + 1):
        if i % j == 0:
            la_so_nguyen_to3 = False
            break
    if la_so_nguyen_to3:
        dem +=1
print("Có", dem, "số nguyên tố.")