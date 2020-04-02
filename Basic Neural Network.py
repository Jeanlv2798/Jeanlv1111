# Materializing three-layer Basic Neural Network based on 'Deep Learning From Scratch(Hanbit Media, Inc.,2017)

import numpy as np


# Activation functions : sigmoid, identity_function, softmax

def sigmoid(x):
    """sigmoid = 1 / (1 + exp(-x))"""
    # return 1 / (1 + math.exp(-x))
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def softmax(x):
    max_x = np.max(x)
    y = np.exp(x - max_x) / np.sum(np.exp(x - max_x))
    return y


# Materializing three-layer Basic Neural Network

def basic_network():
    np.random.seed(118)

    # Dict type for saving weight/ bias
    network = dict()

    # x @ W1 + b1: 1x3 array
    # (1x2) @ (2x3) + (1x3)
    network['W1'] = np.random.random(size=(2, 3)).round(2)
    network['b1'] = np.random.random(3).round(2)

    # z1 @ W2 + b2: 1x2 array
    # (1x3) @ (3x2) + (1x2)
    network['W2'] = np.random.random((3, 2)).round(2)
    network['b2'] = np.random.random(2).round(2)

    # z2 @ W3 + b3: 1x2 array
    # (1x2) @ (2x2) + (1x2)
    network['W3'] = np.random.random((2, 2)).round(2)
    network['b3'] = np.random.random(2).round(2)

    return network


# forward propagation

def forward(network, x):

    # weight
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # Activation function at the first layer: Sigmoid
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)
    z2 = sigmoid(z1.dot(W2) + b2)

    # Output layer's value
    y = z2.dot(W3) + b3

    # Output layer's activation function will be different as per the purpose of ML issues
    # For instance, identity function -> regression | softmax -> classification
    # At this def we will use softmax, but also type identity function together.

    # return identity_function(y)  # identity_function
    return softmax(y)
