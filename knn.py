import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression


def get_data():
    file = open('data.txt', 'r')
    file_data = file.read().split("\n")
    data_bunch = [i for i in file_data]
    data_bunch.remove(data_bunch[-1])
    random.shuffle(data_bunch)
    pre_processed_features = []
    pre_processed_labels = []
    for j in data_bunch:
        d = j.split(",")
        t_d = [i for i in d]
        t_d.remove(t_d[-1])
        pre_processed_features.append(t_d)
        pre_processed_labels.append(d[-1])
    processed_features = pre_processed_features
    processed_labels = []
    for label in pre_processed_labels:
        label = label.replace("Iris-versicolor", "versicolor")
        label = label.replace("Iris-setosa", "setosa")
        processed_labels.append(label)
    data = (processed_features, processed_labels)
    return data


def process_data(pre_data):
    train_data = []
    test_data = []
    for d in pre_data[0]:
        sub_data = []
        for entry in d:
            sub_data.append(float(entry))
        train_data.append(sub_data)
    for d in pre_data[1]:
        if d == "setosa":
            test_data.append(1)
        elif d == "versicolor":
            test_data.append(0)
    return (train_data, test_data)


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



trials = 10
lr_errors = []
sklr_errors = []
pre_data = get_data()
data = process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80])
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:])
test_labels = np.array(labels[80:]).reshape(20, 1)
K = 10
knn_pred = knn(train_features, train_labels,
                test_features, test_labels, k=K)
