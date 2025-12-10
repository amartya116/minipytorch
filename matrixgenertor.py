import random as rd
def matrixcreate(n,m,k):
    temp=[]
    if(isinstance(k,(int,float))):
        for i in range(n):
            for j in range(m):
                temp[i][j]=k
    elif(k=="random"):
         for i in range(n):
            for j in range(m):
                temp[i][j]=rd.randint(1,1000)
    return temp
 