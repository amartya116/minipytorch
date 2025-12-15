import numpy as np
cimport numpy as np
from libc.math cimport fabs

def det(np.ndarray[np.float64_t, ndim=2] A):
    cdef int n = A.shape[0]
    cdef double det = 1.0
    cdef int sign = 1
    cdef int i, j, k, p
    cdef double pivot, f

    A = A.copy()

    for i in range(n):
        p = i
        for j in range(i+1, n):
            if fabs(A[j,i]) > fabs(A[p,i]):
                p = j

        if fabs(A[p,i]) < 1e-12:
            return 0.0

        if p != i:
            A[[i,p]] = A[[p,i]]
            sign = -sign

        pivot = A[i,i]
        det *= pivot

        for j in range(i+1, n):
            f = A[j,i] / pivot
            for k in range(i+1, n):
                A[j,k] -= f * A[i,k]

    return det * sign
