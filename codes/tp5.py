import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
X = digits.data
y = digits.target

solution = np.array([
0.74,0.56,0.79,0.92,0.28,0.13,0.53,0.80,0.49,0.91,0.91,0.88,0.71,0.96,0.31,0.30,
0.01,0.14,0.36,0.42,0.53,0.99,0.73,0.53,0.84,0.10,0.34,0.63,0.02,0.29,0.46,0.30,
0.18,0.21,0.23,0.78,0.59,0.50,0.27,0.30,0.36,0.99,0.15,0.60,0.03,0.37,0.52,0.12,
0.32,0.69,0.48,0.91,0.45,0.57,0.46,0.62,0.68,0.48,0.27,0.94,0.47,0.70,0.12,0.35
])

SF = 25
alpha = 0.9

idx = np.argsort(solution)[::-1][:SF]

X = X[:, idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = KNeighborsClassifier(5)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)

f = alpha*(1-acc) + (1-alpha)*(SF/64)

print("Features:", idx)
print("Accuracy:", acc)
print("f(x):", f)