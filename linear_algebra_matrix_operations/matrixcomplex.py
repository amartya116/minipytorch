#Complex opertions with time complexitx more than or equal to O(n3) best to do it in highly optmised manner using numpy
import numpy as np

def determinant(A):
    a = np.array(A)
    result=np.linalg.det(a)
    return result

def inverse(A):
    a=np.array(A)
    result=np.linalg.inv(a)
    return result

def eigenvalue(A):
    a=np.array(A)
    result=np.linalg.eig(a)
    return result

def SVD(A):
    a=np.array(A)
    result=np.svd(a)
    return result

def transpose(A):
    a=np.array(A)
    result=np.matrix_transpose(a)
    return result

def trace(A):
    a=np.array(A)
    result=np.linalg.trace(a)
    return result

def matmul(A,B):
    a = np.array(A)
    b = np.array(B)
    result=np.linalg.matmul(a, b)
    return result

