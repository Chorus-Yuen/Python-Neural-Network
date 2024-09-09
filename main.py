from fontTools.varLib.interpolatable import test_gen

from network import Network
import numpy as np
from backprop import back_propagation
from function import plot_graph, cost_function


def test_general(n, inputs, targets):
    count = 0
    ys = []
    n.output(np.array(inputs[0]))
    while abs(n.final_output()[0] - targets[0][0]) > 0.05 and count < 20000:
        back_propagation(n, targets[0], 1)
        ys.append(abs(n.final_output()[0] - targets[0][0]))
        count += 1
        n.output(np.array(inputs[0]))

    plot_graph(count, ys)

def test_xor(n):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    num = 500
    ys = []
    for i in range(num):
        for a in range(4):
            n.output(np.array(inputs[a]))
        back_propagation(n, targets[i%4], 0.5)
        ys.append(cost_function(n.final_output(), targets[i%4]))

    plot_graph(num, ys)

sigma = 3

network = Network([2, 2, 1])
network.create(sigma)

test_general(network, [[0, 1]], [[1]])
#test_xor(network)