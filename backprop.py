from network import Network, Layer
from function import *

def get_partial_derivative(diff, z, x):
    return - diff * x * d_sigmoid(z)

def propagate_one_layer(layer: Layer, diff_list, scale):
    d_values = [0 for _ in range(len(layer.get_values()))]
    for diff_ind, diff in enumerate(diff_list):
        values = layer.get_values()
        weights = layer.get_weights()
        biases = layer.get_biases()
        z = np.matmul(weights[diff_ind], values) + biases[diff_ind]
        for input_ind in range(len(values)):
            d_weight = get_partial_derivative(diff, z, values[input_ind]) * scale
            layer.add_weight(diff_ind, input_ind, d_weight)
            d_values[input_ind] += get_partial_derivative(diff, z, weights[diff_ind][input_ind]) * scale
        d_bias = get_partial_derivative(diff, z, 1) * scale
        layer.add_bias(diff_ind, d_bias)
    return d_values

def back_propagation(n: Network, target, scale = 0.1):
    number_of_layers = n.get_number_of_layers()
    diff_list = []
    for ind, a in enumerate(n.get_layer(number_of_layers - 1).get_values()):
        diff_list.append(a-target[ind])
    for layer_ind in range(number_of_layers - 2, -1, -1):
        layer = n.get_layer(layer_ind)
        diff_list = propagate_one_layer(layer, diff_list, scale)