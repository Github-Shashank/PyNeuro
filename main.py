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
    def __init__(self, neuron_type: NeuronType, bias: float | None = None, activation: Activation | None = None):
        self.type = neuron_type
        self.layer = None

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
        self.incoming_connections = []
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
    def __init__(self, layer_type: LayerType):
        self.type = layer_type
        self.neurons = []

    def add_neuron(self, neuron: Neuron):
        neuron.layer = self 
        self.neurons.append(neuron)


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.neurons = {}
        self.connections = {}

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

        for neuron in layer.neurons:
            self.neurons[id(neuron)] = neuron

    def connect(self, source: Neuron, destination: Neuron, weight: float = 0.0):
        connection = Connection(source, destination, weight)
        self.connections[id(connection)] = connection

        source.outgoing_connections.append(connection)
        destination.incoming_connections.append(connection)

        return connection


network = NeuralNetwork()

# -----------------
# Input layer
# -----------------

input_layer = Layer(LayerType.INPUT)

input_1 = Neuron(NeuronType.INPUT)
input_2 = Neuron(NeuronType.INPUT)

input_layer.add_neuron(input_1)
input_layer.add_neuron(input_2)


# -----------------
# Hidden layer
# -----------------

hidden_layer = Layer(LayerType.HIDDEN)

hidden_1 = Neuron(
    NeuronType.HIDDEN,
    bias=0.1,
    activation=Activation.SIGMOID
)

hidden_2 = Neuron(
    NeuronType.HIDDEN,
    bias=0.2,
    activation=Activation.SIGMOID
)

hidden_3 = Neuron(
    NeuronType.HIDDEN,
    bias=0.3,
    activation=Activation.SIGMOID
)

hidden_layer.add_neuron(hidden_1)
hidden_layer.add_neuron(hidden_2)
hidden_layer.add_neuron(hidden_3)


# -----------------
# Output layer
# -----------------

output_layer = Layer(LayerType.OUTPUT)

output = Neuron(
    NeuronType.OUTPUT,
    bias=0.4,
    activation=Activation.SIGMOID
)

output_layer.add_neuron(output)


# -----------------
# Add layers
# -----------------

network.add_layer(input_layer)
network.add_layer(hidden_layer)
network.add_layer(output_layer)

# Input → Hidden

network.connect(input_1, hidden_1, 0.1)
network.connect(input_2, hidden_1, 0.2)

network.connect(input_1, hidden_2, 0.3)
network.connect(input_2, hidden_2, 0.4)

network.connect(input_1, hidden_3, 0.5)
network.connect(input_2, hidden_3, 0.6)


# Hidden → Output

network.connect(hidden_1, output, 0.7)
network.connect(hidden_2, output, 0.8)
network.connect(hidden_3, output, 0.9)

print("Number of layers:", len(network.layers))

for i, layer in enumerate(network.layers):
    print(
        "Layer", i,
        "type =", layer.type,
        "neurons =", len(layer.neurons)
    )

print(
    network.neurons[id(input_1)] is input_1
)

print(
    network.neurons[id(input_1)] is input_1
)

print("\nConnections:", len(network.connections))

for connection in network.connections.values():
    print(
        "source =", id(connection.source),
        "destination =", id(connection.destination),
        "weight =", connection.weight
    )

for connection in network.connections.values():
    print(
        "source =", id(connection.source),
        "destination =", id(connection.destination),
        "weight =", connection.weight
    )

print(
    "\nHidden 1 incoming:",
    len(hidden_1.incoming_connections)
)

print(
    "\nOutput incoming:",
    len(output.incoming_connections)
)

print(
    "Output outgoing:",
    len(output.outgoing_connections)
)

print(
    "\nInput 1 incoming:",
    len(input_1.incoming_connections)
)

print(
    "Input 1 outgoing:",
    len(input_1.outgoing_connections)
)

connection = input_1.outgoing_connections[0]

print(
    connection in hidden_1.incoming_connections
)

