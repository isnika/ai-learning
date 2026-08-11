'''
Nhập một số N.

In bảng cửu chương của N.
'''

n = int(input("Enter an integer: "))
if n <= 0:
    print("Please enter a positive integer")
else:
    for i in range (1,11):
        print(f"{n} x {i} = {n*i}")


'''
In toàn bộ bảng cửu chương từ 2 đến 9.
'''

for n in range(2,10):
    print(f"Multiplicaton table of {n} ")

    for i in range(1,11):
        print(f"{n} x {i} = {n*i}")
    print()

'''
Nhập hai số A và B.

In bảng cửu chương từ A đến B.
'''

A = int(input("Enter an integer A: "))
B = int(input("Enter an integer B: "))

if A > B:
    print("Pleas enter A < B")
else:
    for n in range(A,B+1):
        print(f"Multiplicaton table of {n} ")

        for i in range(1,11):
            print(f"{n} x {i} = {n*i}")
    print()