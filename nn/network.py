import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.default_rng().random((output_size, input_size))
        self.biases = np.random.default_rng().random((output_size,)) # Adding bias help us move the axis away from the origin.
        self.activation = lambda x: 1 / (1 + np.exp(-x))
        # Filled in by Network.learn, consumed by apply_gradients. Kept on the
        # layer rather than returned so step 12 can accumulate across a batch.
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.biases)

    def calculate_output(self, inputs : np.ndarray) -> np.ndarray:
        output = np.dot(self.weights, inputs) + self.biases
        if self.activation is not None:
            output = self.activation(output)
        return output

    # -- step 12 onwards: the analytic gradient ---------------------------
    # node_values are the error signal for THIS layer's output nodes, which is
    # d(cost)/d(a) * sigma'(z) -- not the raw error, and not this layer's
    # output activations. inputs are the activations that fed this layer.

    def weight_gradient(self, inputs: np.ndarray, node_values: np.ndarray) -> np.ndarray:
        return np.outer(node_values, inputs)

    def bias_gradient(self, node_values: np.ndarray) -> np.ndarray:
        return node_values          # d(z)/d(b) is 1, so the node value passes through

    def zero_gradients(self):
        self.weight_gradients.fill(0.0)
        self.bias_gradients.fill(0.0)

    def apply_gradients(self, learning_rate: float = 0.01):
        self.weights -= learning_rate * self.weight_gradients
        self.biases -= learning_rate * self.bias_gradients

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

    def cost(self, inputs: np.ndarray, expected: np.ndarray) -> float:
        predicted = self.calculate_output(inputs)
        return float(np.mean((predicted - expected) ** 2))

    def average_cost(self, X: np.ndarray, Y: np.ndarray) -> float:
        total_cost = 0.0
        for inputs, expected in zip(X, Y):
            total_cost += self.cost(inputs, expected)
        return total_cost / len(X)

    def learn(self, X: np.ndarray, Y: np.ndarray,
              learning_rate: float = 0.01, h: float = 1e-5) -> float:
        """One step of gradient descent, with gradients measured by nudging.

        Slow and almost impossible to get wrong: nudge a single parameter,
        see which way the cost moved, put it back. No calculus anywhere. That
        makes it the reference every faster method gets checked against.

        Costs 1 + num_parameters calls to average_cost, each of which runs the
        whole dataset through the network. Returns the cost before the step.
        """
        original = self.average_cost(X, Y)

        # Pass 1: measure. Every parameter is restored before the next one is
        # touched, so all gradients describe the slope at the SAME point.
        for layer in self.layers:
            for idx, _ in np.ndenumerate(layer.weights):
                layer.weights[idx] += h
                layer.weight_gradients[idx] = (self.average_cost(X, Y) - original) / h
                layer.weights[idx] -= h

            for idx, _ in np.ndenumerate(layer.biases):
                layer.biases[idx] += h
                layer.bias_gradients[idx] = (self.average_cost(X, Y) - original) / h
                layer.biases[idx] -= h

        # Pass 2: only now does the network actually move.
        for layer in self.layers:
            layer.apply_gradients(learning_rate)

        return original

    def num_parameters(self) -> int:
        return sum(l.weights.size + l.biases.size for l in self.layers)
