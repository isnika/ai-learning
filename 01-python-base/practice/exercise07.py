'''
OPTIPN 1: basic: uss for if while
Viết chương trình menu.

===== MENU =====

1. Kiểm tra chẵn lẻ
2. Kiểm tra số nguyên tố
3. Tính giai thừa
4. In bảng cửu chương
0. Thoát
'''

while True:
    print("\n ----- Menu ----- ")
    print("1. Check even or odd numbers")
    print("2. Check prime numbers")
    print("3. Calculate factorial")
    print("4. Print multiplication table")
    print("0. Exit")

    choice = input("Enter your choice: ")
    if choice == "1":
        n=int(input("Enter integer number: "))
        if n % 2 == 0:
            print(f"{n} is an even number.")
        else:
            print(f"{n} is an odd number.")

    elif choice == "2":
        n=int(input("Enter integer number: "))
        if n < 2:
            print(f"{n} is not  an prime number.")
        else:
            is_prime = True
            for i in range(2, n):
                if n % i == 0:
                    is_prime = False
                    break
            if is_prime:
                print(f"{n} is a prime number.")
            else:
                print(f"{n} is not a prime number.")

    elif choice =="3":
        n=  int (input("Enter integer number(N>=0): "))
        factorial =1

        if n == 0:
            factorial =1
            print(f"The factory of {n} is : {factorial}")
        else:
            for i in range (1,n+1):
                factorial *= i
        print(f"The factorial of {n} is {factorial}")

    elif choice =="4":
        n = int(input("Enter integer number(N>=0): "))
        if n < 0:
            print("Pleas enter a positive number.")
        else:
            print(f"Multiplication table of {n}")
            for i in range (1,11):
                print(f"{n} x {i} = {n*i}")

    elif choice =="0":
        break
    else:
        print("Please enter a valid choice.")