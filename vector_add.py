def vector_add(A, B):
    # Case 1: 1D vectors
    if isinstance(A[0], (int, float)):
        if len(A) != len(B):
            raise ValueError("Dimensions not equal")
        return [A[i] + B[i] for i in range(len(A))]

    # Case 2: 2D matrices
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrix dimensions not equal")

    result = []
    for rowA, rowB in zip(A, B):
        result.append([rowA[i] + rowB[i] for i in range(len(rowA))])
    return result