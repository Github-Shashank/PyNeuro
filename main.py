from Activation import Activation
from enum import Enum, auto 


class NeuronType(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()


class LayerType(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()


class Neuron:
    def __init__(self, neuron_id: int, neuron_type: NeuronType, bias: float | None = None, activation: Activation | None = None):
        self.id = neuron_id
        self.type = neuron_type

        # Trainable parameter
        self.bias = bias

        # Forward-pass state
        self.z = 0.0
        self.value = 0.0

        # Backpropagation state
        self.gradient = 0.0
        self.bias_gradient = None if neuron_type == NeuronType.INPUT else 0.0

        # Activation function
        self.activation = activation 

        # Graph realationships
        self.incomming_connections = []
        self.outgoing_connections = []


class Connection:
    def __init__(self, source : int, destination : int, weight : float | None = 0.0):
        # Neurons
        self.source = source
        self.destination = destination

        # Trainable parameter
        self.weight = weight

        # Backpropagation state
        self.weight_gradient = 0.0


class Layer:
    def __init__(self):
        self.nuerons = []


class NueralNetwork:
    def __init__(self):
        self.layers = []


input_neuron = Neuron(
    0,
    NeuronType.INPUT
)

hidden_neuron = Neuron(
    1,
    NeuronType.HIDDEN,
    bias=0.1,
    activation=Activation.SIGMOID
)

output_neuron = Neuron(
    2,
    NeuronType.OUTPUT,
    bias=0.2,
    activation=Activation.SIGMOID
)

print(output_neuron.__dict__)