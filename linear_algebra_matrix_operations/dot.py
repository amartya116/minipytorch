import numpy as np
def dot(A, B):
    if isinstance(A[0], (int, float)) and isinstance(B[0], (int, float)):
        if len(A) != len(B):
            raise ValueError("Dimensions not equal for vector dot product")
        return sum(A[i] * B[i] for i in range(len(A)))
    


    if isinstance(A, (int, float)) and isinstance(B,list):
        if(isinstance(B[0],list)):
            for i in B:
                for j in i:
                    j=j*A
        else:
            for i in B:
                i=i*A
    if isinstance(B, (int, float)) and isinstance(A,list):
        if(isinstance(A[0],list)):
            for i in A:
                for j in i:
                    j=j*B
        else:
            for i in A:
                i=i*B
        
            
    if not isinstance(A[0], list) or not isinstance(B[0], list):
        raise ValueError("Both must be vectors or matrices")

    a=np.array(A)
    b=np.array(B)
    result=np.matmul(a,b)
    return result
