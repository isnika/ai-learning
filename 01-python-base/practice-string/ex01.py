'''
Đếm ký tự

Đề bài: Nhập một chuỗi. Đếm xem chuỗi có bao nhiêu ký tự.
'''

text = input("Enter text: ")
count = len(text)

print(f"Number of characters: {count}")

c = 0
for char in text: # char lan luot nhan tung ly tu
    c += 1
print(f"Number of characters: {c}")