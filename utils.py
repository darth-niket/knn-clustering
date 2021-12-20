# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: Niket Nagaraj Malihalli -- nmalihal
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff
import math

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i]-x2[i])**2
    return math.sqrt(sum)

    raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    sum = 0
    for i in range(len(x1)):
        sum += abs(x1[i] - x2[i])
    return sum

    raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    print(x)
    if derivative == False:
        return x
    if derivative == True:
        return 1

    raise NotImplementedError('This function must be implemented by the student.')


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    func = 1 / (1 + np.exp(-x))
    if derivative == False:
        return f
    if derivative == True:
        return f * (1-f)


    raise NotImplementedError('This function must be implemented by the student.')


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    func = (2 / (1 + np.exp(-(2*x)))) - 1
    if derivative == False:
        return func
    if derivative == True:
        return 1 - (func**2)

    raise NotImplementedError('This function must be implemented by the student.')


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative == False:
        if x < 0:
            return 0
        if x >= 0:
            return x

    if derivative == True:
        if x < 0:
            return 0
        if x >= 0:
            return 1


    raise NotImplementedError('This function must be implemented by the student.')


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    return 0

    raise NotImplementedError('This function must be implemented by the student.')


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    map = {}
    for i in range(y.shape[0]):
        map[y[i]] = i

    one_hot_encode = []
    for item in y:
        arr = list(np.zeros(len(y), dtype=int))
        arr[map[item]] = 1
        one_hot_encode.append(arr)

    return one_hot_encode

    raise NotImplementedError('This function must be implemented by the student.')

def getNeighbours(X_test_row,X,y,k,distance_func):
    distances = []
    for i in range(X.shape[0]):
        dist = distance_func(X_test_row, X[i])
        distances.append((X_test_row, dist, y[i]))
    distances.sort(key=lambda x: x[1])
    neighbours = []
    for j in range(k):
        neighbours.append((distances[j][0], distances[j][2]))
    return neighbours
