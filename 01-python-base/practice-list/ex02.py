'''
Đề bài: Nhập một danh sách số. Tính giá trị trung bình.
'''

numbers = list(map(int, input("Input list of numbers: ").split()))

sum = 0
for n in numbers:
    sum += n
average = sum / len(numbers)
print(f"The average of {numbers} is {average}")