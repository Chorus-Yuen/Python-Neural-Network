import numpy as np
import matplotlib.pyplot as plt

def sigmoid_list(l):
    output = []
    for i in range(len(l)):
        output.append(1 / (1 + np.exp(-l[i])))
    return output

def d_sigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2

def cost_function(c, a):
    total = 0
    for i in range(len(a)):
        total += (c[i] - a[i]) ** 2
    return total

def plot_graph(x_num, y_list):
    plt.ylim([-0.1, 1.1])
    plt.plot([i for i in range(x_num + 1)], [0 for _ in range(x_num + 1)])
    plt.plot([i for i in range(1, x_num + 1)], y_list)
    plt.show()