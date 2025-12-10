temp=[]
def flatten(A):
    for i in A:
        if(isinstance(i,list)):
            flatten(i)
        else:
            temp.append(i)
    return temp