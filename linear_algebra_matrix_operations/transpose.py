def transpose(A):
    if len(A) == 0:
        return []
    if isinstance(A[0], (int, float)):
        raise TypeError("Must be a 2D array")
    
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]