import example

class VectorAdder:
    def __init__(self):
        pass

    def dot(self, i, j):
        return example.add(i, j)
    
    def flatten(self, i, j):
        return example.sub(i, j)
    
    def generatematrix(self,size):
        return example.darray(size)


# Example usage
if __name__ == "__main__":
    adder = VectorAdder()
    result = adder.generatematrix(5)
    print(result)