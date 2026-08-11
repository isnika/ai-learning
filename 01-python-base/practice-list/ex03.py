'''
Đề bài: Nhập một danh sách. Xóa các phần tử bị trùng, chỉ giữ lại một lần xuất hiện.
'''

input = list(map(int, input("Enter list a number: ").split()))
result = []

for n in input:
    if n not in result:
        result.append(n)

print(result) #add n vao cuoi danh sach
