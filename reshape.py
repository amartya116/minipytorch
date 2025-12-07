def reshape(x,y,A):
    temp=[]
    if(len(A)!=x*y):
        raise TypeError("unable to reshape")
    for i in range(0,x*y-1,x):
        temp.append(A[i:i+x])
    return temp
    