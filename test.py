import numpy as np

l = np.array([16, 45, 6, 1, 36, 43, 39, 23, 18, 12, 47, 34, 24, 3, 9, 10, 28, 35, 42, 13, 32,  5, 48, 46, 40, 49, 21, 30, 20, 44, 37, 38, 29, 19, 27,  8, 26,  4,  0, 15,  2, 11, 33,  7, 25, 14, 31, 22, 17, 41,])
i = 6

print(np.where(l == i)[0][0])