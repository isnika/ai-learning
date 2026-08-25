'''
Cho:
y_true = [10, 20, 30, 40]
y_pred = [12, 18, 29, 45]
Viết Python tính:
MSE
'''

y_true = [ 10,20,30,40]
y_pred = [ 12,18,29,45]
print("y_true =", y_true)
print("y_pred =", y_pred)

n = len(y_true)
mse = sum((y_true[i]- y_pred[i]) ** 2 for i in range(n)) / n
print("MSE = ", mse)