import random
from function import *

class Network:
    def __init__(self, layer_node_count):
        self.layers = []
        self.layer_node_count = layer_node_count

    def create(self, sigma, demo = 0):
        for i in range(len(self.layer_node_count)):
            layer = Layer(self.layer_node_count[i])
            if demo == 1: layer = sample_network(i)
            elif demo == 2: layer = sample_network_2(i)
            elif i + 1 == len(self.layer_node_count): layer.construct(0, 0)
            else: layer.construct(self.layer_node_count[i+1], sigma)
            self.layers.append(layer)

    def print(self):
        print("------------------------------------------")
        for ind, layer in enumerate(self.layers):
            print("Layer " + str(ind))
            print(layer)
        print("------------------------------------------")

    def output(self, layer_values, display = False):
        for ind, layer in enumerate(self.layers):
            layer.set_value(layer_values)
            if ind + 1 == len(self.layers):
                if display: self.print()
                return
            layer_values = sigmoid_list(np.matmul(layer.get_weights(), layer_values) + layer.get_biases())

    def final_output(self):
        return self.layers[-1].get_values()

    def get_number_of_layers(self):
        return len(self.layer_node_count)

    def get_layer(self, ind):
        return self.layers[ind]

class Layer:
    def __init__(self, num_of_nodes, weights = np.array([]), biases = np.array([])):
        self.n = num_of_nodes
        self.values = np.array([])
        self.weights = weights
        self.biases = biases

    def construct(self, next_layer_num_of_nodes, sigma = 1):
        self.weights = np.array([[random.normalvariate(0, sigma)
                                  for _ in range(self.n)]
                                 for _ in range(next_layer_num_of_nodes)])
        self.biases = np.array([random.normalvariate(0, sigma) for _ in range(next_layer_num_of_nodes)])

    def __str__(self):
        output = "Weights:\n"
        for w in self.weights:
            line = ""
            for i in w:
                line += str(i) + " "
            output += line + "\n"
        output += "Biases:\n"
        line = ""
        for i in self.biases:
            line += str(i) + " "
        output += line + "\n"
        output += "Values:\n"
        line = ""
        for i in self.values:
            line += str(i) + " "
        output += line + "\n"
        return output

    def set_value(self, values):
        self.values = values

    def get_values(self):
        return self.values

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def add_weight(self, y, x, add):
        self.weights[y][x] += add

    def add_bias(self, y, add):
        self.biases[y] += add

def sample_network(i):
    if i == 0: return Layer(2, np.array([[-1.46, -0.41], [0.77, -0.55]]),
                            np.array([-0.1, -0.55]))
    if i == 1: return Layer(2, np.array([[-0.1, 2.42]]),
                            np.array([-0.72]))
    if i == 2: return Layer(1)

def sample_network_2(i):
    if i == 0: return Layer(2, np.array([[20, 20], [-20, -20]]),
                            np.array([-10, 30]))
    if i == 1: return Layer(2, np.array([[20, 20]]),
                            np.array([-30]))
    if i == 2: return Layer(1)
