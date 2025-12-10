def reshape(x, y, A):
    if len(A) != x * y:
        raise ValueError("Unable to reshape: size mismatch")
    
    temp = []
    for i in range(0, x * y, y):
        temp.append(A[i:i+y])
    return temp