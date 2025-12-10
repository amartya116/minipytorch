def identity(n):
    temp=[]
    for i in range(len(n)):
        for j in range(len(n)):
            if(i==j):
                temp[i][j]=1
            else:
                temp[i][j]=0
    return temp