'''
Đề bài: Nhập một danh sách số. In ra danh sách theo thứ tự tăng dần và giảm dần.
'''

numbers = list(map(int, input("Enter list a number: ").split()))
print(f"Initial list: {numbers}")

numbers.sort()

print(f"Ascending: {numbers}")

numbers.sort(reverse=True)

print(f"Descending: {numbers}")

