'''
Nhập số nguyên dương N.
Tính: N!
'''

N = int(input("Enter an integer N: "))
factorial = 1

if N == 0:
    factorial =1
else:
    for i in range (1,N+1):
        factorial *= i
print(f"The factorial of {N} is {factorial}")


'''
Nhập nhiều số nguyên.

In giai thừa của từng số.
'''

numbers = list(map(int, input("Enter integers: ").split()))
print(numbers)

for n in numbers:
    factorial1 = 1

    for i in range (1,n+1):
        factorial1 *= i
    print(f"The factorial of {n} is {factorial1}")