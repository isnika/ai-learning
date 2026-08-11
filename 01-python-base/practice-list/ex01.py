"""
Đề bài: Nhập một danh sách số nguyên. Tìm số lớn nhất và nhỏ nhất.
"""

numbers = list(map(int, input (" Input list number: ").split()))
print(numbers)

min_number = numbers[0]
max_number = numbers[0]

for n in numbers:
    if n > min_number:
        max_number = n
    if n < min_number:
        min_number = n

print(f"The minimum number is {min_number}")
print(f"The maximum number is {max_number}")