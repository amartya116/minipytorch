from vector_add import vector_add
from dot import dot
from transpose import transpose
from reshape import reshape
from flatten import flatten

class Tensor:
    def __init__(self, data):
        self.data = data

    def shape(self):
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))  # 2D
        return (len(self.data))                        # 1D

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")
        return Tensor(vector_add(self.data, other.data))
    
    def numel(self):
        numel=flatten(self.data)
        return len(numel)

    @property
    def T(self):
        return Tensor(transpose(self.data))
    
    def reshape(x,y,self):
        return Tensor(reshape(x,y,self.data))
    
    
    def flatten(self):
        return Tensor(flatten(self.data))
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")
        return dot(self.data, other.data)



t1 = Tensor([1, 2, 3, 4])
t2 = Tensor([2, 3, 4, 5])

print((t1 + t2).data)  # [3, 5, 7, 9]
print(t1 * t2)         # 40
