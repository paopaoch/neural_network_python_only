''' 
Date created: 20th April 2020
Source: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
info: This script is a very vanilla version of neural network. It uses Back Propagation and gradient decent to train the model
As of he date created, it only supports classifying data into two class 1 and 0.

'''


from random import random, seed
from math import exp, sqrt

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



def cal_weighted_input(inputs, weights, bias):
    ''' calculate the weighted input z of a neuron using the bias, weights and inputs '''
    weighted_input = bias
    for i in range(len(weights)):
        weighted_input += weights[i] * inputs[i]
    return weighted_input



def sigmoid(weighted_input):
    return 1.0/(1.0 + exp(-weighted_input))



def forward_propagate(network, inputs):
    ''' This function calculates all the activation in the network using the cal_weighted_input and sigmoid functions '''
    previous_activation = inputs
    for layer in network:
        new_inputs = list() # to store the new activations used in the next layer
        for neuron in layer:
            activation = sigmoid(cal_weighted_input(previous_activation, neuron['weights'], neuron['bias']))
            neuron['output'] = activation
            new_inputs.append(neuron['output'])
        previous_activation = new_inputs
    return previous_activation


# network = [[{'weights': [0.13436424411240122, 0.8474337369372327], 'bias': 0.763774618976614}],
# 		[{'weights': [0.2550690257394217], 'bias': 0.49543508709194095}, {'weights': [0.4494910647887381], 'bias': 0.651592972722763}]]

# print(forward_propagate(network, [2.5, 3]))
# print(network)
# print(sigmoid(0.13436424411240122 * 2.5 + 0.8474337369372327 * 3 + 0.763774618976614))
# print(sigmoid(sigmoid(0.13436424411240122 * 2.5 + 0.8474337369372327 * 3 + 0.763774618976614)*0.2550690257394217 + 0.49543508709194095))

def sigmoid_derivative(output):
    return output * (1 - output)



def backward_propagate_error(network, expected):
    ''' perform back-propagation from the output layer to calculate the delta value which is the "error" '''
    for i in reversed(range(len(network))):
        layer = network[i]
        if i == len(network) - 1: # output layer
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = (expected[j] - neuron['output']) * sigmoid_derivative(neuron['output'])
        else: # hidden layer
            for j in range(len(layer)):
                error = 0
                for neuron in network[i + 1]:
                    error += neuron['weights'][j] * neuron['delta'] * sigmoid_derivative(layer[j]['output'])
                layer[j]['delta'] = error


# network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327], 'bias': 0.763774618976614}],
# 		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217], 'bias': 0.49543508709194095}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381], 'bias': 0.651592972722763}]]

# expected = [0, 1]
# backward_propagate_error(network, expected)
# for layer in network:
#     print(layer)

# new_weight = old_weight + learning_rate * error * input

def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]] # set input as output of previous neuron
        else:
            inputs = row

        layer = network[i]
        for neuron in layer:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['bias'] += learning_rate * neuron['delta'] # bias is assume to have an input of 1


def train_network(network, x_train, y_train, learning_rate, epochs, n_outputs):
    for epoch in range(epochs):
        sum_error = 0
        for n in range(len(y_train)):
            x = x_train[n]
            y = y_train[n]
        
            outputs = forward_propagate(network, x)
            # print(outputs)
            if y == 1:
                expected = [0,1]
            elif y == 0:
                expected = [1,0]
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, x, learning_rate)
        print('epoch=',epoch+1, ', ', 'lrate=',learning_rate, ', ', 'error=', sum_error/(len(y_train)))

def make_predictions(network, data):
    prediction = forward_propagate(network, data)
    return prediction.index(max(prediction))

def make_predictions_list(network, test_dataset):
    predictions = list()
    for i in range(len(test_dataset)):
        row = test_dataset[i]
        prediction = make_predictions(network, row)
        predictions.append(prediction)
    return predictions

def accuracy_evaluation(network, test_dataset, test_y):
    accuracy = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
    for i in range(len(test_y)):
        row = test_dataset[i]
        prediction = make_predictions(network, row)
        if test_y[i] == prediction:
            if test_y[i] == 1:
                accuracy['TP'] += 1
            else:
                accuracy['TN'] += 1
        else:
            if test_y[i] == 1:
                accuracy['FP'] += 1
            else:
                accuracy['FP'] += 1
    
    accuracy['accuracy'] = (accuracy['TP'] + accuracy['TN'])/(accuracy['TP'] + accuracy['TN'] + accuracy['FP'] + accuracy['FN'])

    if (2*accuracy['TP'] + accuracy['FP'] + accuracy['FN']) == 0:
        accuracy['f1'] = None
    else:
        accuracy['f1'] = (2 * accuracy['TP'])/(2*accuracy['TP'] + accuracy['FP'] + accuracy['FN'])

    if (accuracy['TP'] + accuracy['FP']) == 0:
        accuracy['precision'] = None
    else:
        accuracy['precision'] = accuracy['TP']/(accuracy['TP'] + accuracy['FP'])

    if accuracy['TN'] + accuracy['FN'] == 0:
        accuracy['negative_precision'] = None
    else:
        accuracy['negative_precision'] = accuracy['TN']/(accuracy['TN'] + accuracy['FN'])

    if (accuracy['TP'] + accuracy['FN']) == 0:
        accuracy['sensitivity'] = None
    else:
        accuracy['sensitivity'] = accuracy['TP']/(accuracy['TP'] + accuracy['FN'])

    if (accuracy['TN'] + accuracy['FP']) == 0:
        accuracy['specificity'] = None
    else:
        accuracy['specificity'] = accuracy['TN']/(accuracy['TN'] + accuracy['FP'])

    return accuracy

if __name__ == "__main__":
    import time

    start = time.time()
    # defining a dataset
    dataset = [[2.7810836,2.550537003],
        [1.465489372,2.362125076],
        [3.396561688,4.400293529],
        [1.38807019,1.850220317],
        [3.06407232,3.005305973],
        [7.627531214,2.759262235],
        [5.332441248,2.088626775],
        [6.922596716,1.77106367],
        [8.675418651,-0.242068655],
        [7.673756466,3.508563011]]

    # defining expected values for the dataset
    y = [0,0,0,0,0,1,1,1,1,1]

    n_inputs = 2
    n_outputs = 2
    network = initialize_networks(n_inputs, 2, n_outputs, n_layers=2)
    train_network(network, dataset, y, 0.5, 100, n_outputs)

    accuracy = accuracy_evaluation(network, dataset, y)
    print(accuracy)

    print(make_predictions_list(network, dataset))
    print(make_predictions(network, [2.7810836,2.550537003]))
    end = time.time()
    print(end - start)