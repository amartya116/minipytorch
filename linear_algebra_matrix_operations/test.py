import numpy as np
a = np.array([[[1, 0],
              [0, 1]],[[1, 2],
              [6, 1]]])
b = np.array([[[4, 1],
              [2, 2]],[[5, 2],
              [9, 3]]])
c=np.linalg.matmul(a, b)
print(c)