from enum import Enum, auto


class Activation(Enum):
    LINEAR = auto()
    SIGMOID = auto()
    RELU = auto()
    TANH = auto()