'''
OPTIPN 2 use def
Viết chương trình menu.

===== MENU =====

1. Kiểm tra chẵn lẻ
2. Kiểm tra số nguyên tố
3. Tính giai thừa
4. In bảng cửu chương
0. Thoát
'''

def check_even_odd():
    n = int(input("Enter a number: "))
    if n % 2 == 0:
        print(f"{n} is Even")
    else:
        print(f"{n} is Odd")

def check_prime():
    n = int(input("Enter a number: "))
    if n <2:
        print(f"{n} is not a prime number")
        return
    for i in (2,n):
        if n % i == 0:
            print(f"{n} is not a prime number")
            return
    print(f"{n} is a prime number")

def calculate_factorial():
    n = int(input("Enter a number: "))
    if n <0:
        print("Can not calculate the factorial")
        return
    factorial = 1
    for i in range(1, n+1):
        factorial *=i
    print(f"The factorial of {n} is {factorial}")

def multiplication_table():
    n= int (input("Enter a number: "))
    if n<0:
        print("Can not calculate the multiplication table")
    print(f"Multiplication table of {n} is:")
    for i in range (1,11):
        print(f"{n} * {i} = {n * i}")

while True:
    print("\n ----- Menu ----- ")
    print("1. Check even or odd numbers")
    print("2. Check prime numbers")
    print("3. Calculate factorial")
    print("4. Print multiplication table")
    print("0. Exit")

    choice = int(input("Choose function: "))
    if choice == 1:
        check_even_odd()
    elif choice == 2:
        check_prime()
    elif choice == 3:
        calculate_factorial()
    elif choice == 4:
        multiplication_table()
    elif choice == 0:
        break
    else:
        print("Invalid choice")




