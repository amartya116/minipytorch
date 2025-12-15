import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def reshape(int x, int y, list A):
    cdef int i
    cdef int size = x * y
    cdef list temp = []
    cdef list row
    
    if len(A) != size:
        raise ValueError("Unable to reshape: size mismatch")
    
    for i in range(0, size, y):
        row = A[i:i+y]
        temp.append(row)
    
    return temp