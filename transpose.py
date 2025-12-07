def transpose(A):
    if(len(A)==0):
        return []
    if isinstance(A[0], (int, float)):
        raise TypeError("Must be a 2d array")
    else:
        t=[]
        colA=len(A)
        rowA=len(A[0])
        for i in range(colA):
            temp=[]
            for j in range(rowA):
                print(A[j][i])
                temp.append(A[j][i])
            t.append(temp)
    return t
    