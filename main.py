class Nueron:
    def __init__(self, bias=0.0):
        self.bias = bias
        self.value = 0

class Connection:
    def __init__(self, source, destination, weight=0.0):
        self.source = source
        self.destination = destination
        self.weight = weight

class Layer:
    def __init__(self):
        self.nuerons = []

class NueralNetwork:
    def __init__(self):
        self.layers = []