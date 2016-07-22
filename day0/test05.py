import numpy as np

m = 3
n = 2
a = np.zeros([m, n])

print a
print a.shape
print a.dtype.name

a = np.array([[2, 3], [5, 4]])
b = np.array([[1, 1], [1, 1]])

d = np.dot(a, b)  # MATRIX multiplication
print d

print a
print a.T

a = np.array([1, 2])
b = np.array([1, 1])
print np.dot(a, b)  # VECTOR dot (inner) product
print np.outer(a, b)  # VECTOR cross (outer) product

print np.eye(2)

