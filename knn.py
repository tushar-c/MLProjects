import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import process_data


def get_min(x, k=5):
    minimums = []
    data = [j[1] for j in x]
    for i in range(k):
        min = np.min(data)
        minimums.append((x[data.index(min)], min, data.index(min)))
        data.remove(min)
    return minimums


def norm(x):
    return np.sqrt(np.sum(np.power(x, 2)))


def knn(X, Y, test_features, test_labels, k=5):
    estimates = []
    for t in range(len(test_features)):
        norms = []
        for x in X:
            norms.append((x, norm(test_features[t] - x)))
        k_neighbors = get_min(norms, k=k)
        total = 0
        for i in range(len(k_neighbors)):
            true_label = test_labels[t]
            estimate_label = Y[k_neighbors[i][2]]
            total += estimate_label
            prediction = total / k
            if prediction >= 0.5:
                estimates.append((1, true_label))
            elif prediction < 0.5:
                estimates.append((0, true_label))
    sq_error = 0
    for e in estimates:
        sq_error += np.power(e[0] - e[1], 2)
    return sq_error / len(estimates)


pre_data = process_data.get_data()
data = process_data.process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80])
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:])
test_labels = np.array(labels[80:]).reshape(20, 1)
K = 10
knn_pred = knn(train_features, train_labels,
                test_features, test_labels, k=K)
