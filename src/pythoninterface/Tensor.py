from linear_algebra_matrix_operations.matrixcomplex import transpose, reshape, flatten, trace, matmul, SVD, eigenvalue, inverse, determinant
from matrix_ops_simple import add, sub, mul, div, sqrt, sqr, exponent, log, sum1, mul1, mean, max, min, generateI

class Tensor:
    def __init__(self, data):
        self.data = data

    def shape(self):
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))  # 2D
        return (len(self.data))                        # 1D

    def add(self, other):
        return Tensor(add(self.data, other.data))
    
    def sub(self, other):
        return Tensor(sub(self.data, other.data))
    
    def mul(self, other):
        return Tensor(mul(self.data, other.data))

    def div(self, other):
        return Tensor(div(self.data, other.data))
    
    def sqrt(self):
        return Tensor(sqrt(self.data))
    
    def sqr(self):
        return Tensor(sqr(self.data))
    
    def exponent(self):
        return Tensor(exponent(self.data))
    
    def log(self):
        return Tensor(log(self.data))
    
    def sum(self, axis=0):
        return Tensor(sum1(self.data, axis))
    
    def prod(self, axis=0):
        return Tensor(mul1(self.data, axis))
    
    def mean(self, axis=0):
        return Tensor(mean(self.data, axis))
    
    def max(self, axis=0):
        return Tensor(max(self.data, axis))
    
    def min(self, axis=0):
        return Tensor(min(self.data, axis))

    def numel(self):
        numel = flatten(self.data)
        return len(numel)

    @property
    def T(self):
        return Tensor(transpose(self.data))
    
    def reshape(self, shape, order='C'):
        return Tensor(reshape(self.data, shape, order))
    
    def flatten(self):
        return Tensor(flatten(self.data))
    
    def inverse(self):
        return Tensor(inverse(self.data))
    
    def eigenvalue(self):
        return Tensor(eigenvalue(self.data))
    
    def trace(self):
        return Tensor(trace(self.data))
    
    def matmul(self, other):
        return Tensor(matmul(self.data, other.data))
    
    def svd(self):
        return Tensor(SVD(self.data))
    
    def determinant(self):
        return Tensor(determinant(self.data))
    
    @staticmethod
    def identity(size):
        return Tensor(generateI(size))
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")
        return self.add(other)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")
        return self.sub(other)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")
        return self.matmul(other)



