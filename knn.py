import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import process_data
<<<<<<< HEAD
from sklearn.linear_model import LinearRegression


=======


# get the k minimum entries
>>>>>>> 1ffb4591fe66ba3106a70c6392a0ab0a817c8b4e
def get_min(x, k=5):
    minimums = []
    # index '1' contains the norms
    data = [j[1] for j in x]
    for i in range(k):
        min = np.min(data)
        # keep track of the corresponding data point, the corresponding minimum, and the index of minimum occurence
        minimums.append((x[data.index(min)], min, data.index(min)))
        # remove current min to get ride of 'duplicates'
        data.remove(min)
    return minimums


# l2 norm
def norm(x):
    return np.sqrt(np.sum(np.power(x, 2)))


def knn(X, Y, test_features, test_labels, k=5):
    estimates = []
    # go over the test set
    for t in range(len(test_features)):
        norms = []
        # get distance between the vectors in terms of l2 norms ( over the whole train set)
        for x in X:
            norms.append((x, norm(test_features[t] - x)))
        # get the k nearest neighbours
        k_neighbors = get_min(norms, k=k)
        total = 0
        for i in range(len(k_neighbors)):
            true_label = test_labels[t]
            estimate_label = Y[k_neighbors[i][2]]
            # take the k nearest neighbours' majority vote and normalize
            total += estimate_label
            prediction = total / k
            if prediction >= 0.5:
                estimates.append((1, true_label))
            elif prediction < 0.5:
                estimates.append((0, true_label))
    sq_error = 0
    # compute mse
    for e in estimates:
        sq_error += np.power(e[0] - e[1], 2)
    return sq_error / len(estimates)


<<<<<<< HEAD

trials = 10
lr_errors = []
sklr_errors = []
=======
# fetch and clean the data
>>>>>>> 1ffb4591fe66ba3106a70c6392a0ab0a817c8b4e
pre_data = process_data.get_data()
data = process_data.process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80])
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:])
test_labels = np.array(labels[80:]).reshape(20, 1)
# manually set the number of nearest neighbours to consider
K = 10
knn_pred = knn(train_features, train_labels,
                test_features, test_labels, k=K)
