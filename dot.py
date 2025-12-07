def dot(A, B):
    # ----------------------------
    # Case 1: 1D dot product
    # ----------------------------
    if isinstance(A[0], (int, float)) and isinstance(B[0], (int, float)):
        if len(A) != len(B):
            raise ValueError("Dimensions not equal for vector dot product")
        return sum(A[i] * B[i] for i in range(len(A)))

    # ----------------------------
    # Case 2: 2D matrix multiplication
    # ----------------------------
    if not isinstance(A[0], list) or not isinstance(B[0], list):
        raise ValueError("Both must be vectors or matrices")

    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        raise ValueError("Matrix dimensions not compatible for multiplication")

    # Result matrix: rowsA x colsB
    result = []

    for i in range(rowsA):
        row = []
        for j in range(colsB):
            s = 0
            for k in range(colsA):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)

    return result
