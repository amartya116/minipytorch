import matrix_ops_simple

class VectorAdder:
    def __init__(self):
        pass

    def dot(self, i, j):
        return matrix_ops_simple.add(i, j)
    
    def flatten(self, i, j):
        return matrix_ops_simple.sub(i, j)
    
    def generatematrix(self,size):
        return matrix_ops_simple.generateI(size)


# Example usage
if __name__ == "__main__":
    adder = VectorAdder()
    result = adder.dot([[1,2,3],[3,4,5]],[[5,6,7],[8,9,0]])
    print(result)