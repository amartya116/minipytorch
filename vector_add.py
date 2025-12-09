def vector_add(A, B):
    # Case 1: 1D vectors
    if isinstance(A[0], (int, float)):
        if len(A) != len(B):
            raise ValueError("Dimensions not equal")
        return [A[i] + B[i] for i in range(len(A))]


         # Case 1: scalar + list
    # ----------------------------
    if isinstance(A, (int, float)) and isinstance(B,list):
        if(isinstance(B[0],list)):
             for i in range(len(B)):
                for j in range(len(i)):
                    if(i==j):
                        B[i][j]=B[i][j]+A
    if isinstance(B, (int, float)) and isinstance(A,list):
        if(isinstance(A[0],list)):
            for i in range(len(A)):
                for j in range(len(i)):
                    if(i==j):
                        A[i][j]=A[i][j]+B
            
    # Case 2: 2D matrices
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrix dimensions not equal")
    
    result = []
    for rowA, rowB in zip(A, B):
        result.append([rowA[i] + rowB[i] for i in range(len(rowA))])
    return result