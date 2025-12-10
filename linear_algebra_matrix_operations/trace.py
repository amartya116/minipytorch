def trace(A):
    trace=0
    is2d=False
    for k in A:
        if(len(A)==len(k)):
            is2d=True
        else:
            is2d=False
    if(is2d==False):
        raise ValueError("Not a square matrix")
    else:
        for i in range(len(A)):
            for j in range(len(A[0])):
                if(i==j):
                     trace+=A[i][j]
    return trace

def diagonal(A):
    trace=[]
    is2d=False
    for k in A:
        if(len(A)==len(k)):
            is2d=True
        else:
            is2d=False
    if(is2d==False):
        raise ValueError("Not a square matrix")
    else:
        for i in range(len(A)):
            for j in range(len(A[0])):
                if(i==j):
                    trace.append(A[i][j])
    return trace


