from random import random, seed, shuffle
from math import exp, log, sqrt
from decimal import Decimal

def initialize_weights(n_inputs, n_current):
    lower, upper = -(sqrt(6)/sqrt(n_inputs + n_current)), (sqrt(6)/sqrt(n_inputs + n_current))
    return lower, upper

def initialize_networks(n_inputs, n_hidden, n_output, n_layers=1):
    ''' the function initialize the neural network ready for training.
    it takes in the the number of neuron in the input, hidden layer and the output layer
    the function outputs a list of layers each represented by a list of dictionaries where each dictionary is a neuron with a weight and a bias. '''
    network = list()

    for i in range(n_layers):
        if i == 0:
            lower, upper = initialize_weights(n_inputs, n_hidden)
            hidden_layer = [{'weights': [lower + random() * (upper - lower) for i in range(n_inputs)], 'bias': lower + random() * (upper - lower)} for j in range(n_hidden)]
        else:
            lower, upper = initialize_weights(n_hidden, n_hidden)
            hidden_layer = [{'weights': [lower + random() * (upper - lower) for i in range(n_hidden)], 'bias': lower + random() * (upper - lower)} for j in range(n_hidden)]
        network.append(hidden_layer)
    lower, upper = initialize_weights(n_hidden, n_output)
    output_layer = [{'weights': [lower + random() * (upper - lower) for i in range(n_hidden)], 'bias': lower + random() * (upper - lower)} for j in range(n_output)]
    network.append(output_layer)
    return network


n_inputs = 3
n_outputs = 2
network = initialize_networks(n_inputs, 2, n_outputs, n_layers=1)
print(network)

lower, upper = initialize_weights(1, 2)
print(lower)