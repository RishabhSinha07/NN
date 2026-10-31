import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.default_rng().random((output_size, input_size))
        self.biases = np.random.default_rng().random((output_size,)) # Adding bias help us move the axis away from the origin.

    def calculate_output(self, inputs : np.ndarray) -> np.ndarray:
        return np.dot(self.weights, inputs) + self.biases


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def calculate_output(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs
        for layer in self.layers:
            output = layer.calculate_output(output)
        return output
    
    def classify(self, inputs: np.ndarray) -> int:
        output = self.calculate_output(inputs)
        return np.argmax(output)