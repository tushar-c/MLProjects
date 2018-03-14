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


def linear_regression(X, Y, X_, Y_):
    B = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
    predictions = []
    for entry in range(len(X_)):
        p = np.matmul(B.T, X_[entry])
        if p >= 0.5:
            pred = 1
        elif p < 0.5:
            pred = 0
        predictions.append(pred)
    predictions = np.array(predictions)
    error = np.sum(np.power(predictions - test_labels, 2)) / len(features)
    return (B, error, predictions)


pre_data = get_data()
data = process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80])
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:])
test_labels = np.array(labels[80:]).reshape(20, 1)
lr_pred = linear_regression(train_features, train_labels,
                            test_features, test_labels)
