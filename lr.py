import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import process_data
from sklearn.linear_model import LinearRegression


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


pre_data = process_data.get_data()
data = process_data.process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80])
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:])
test_labels = np.array(labels[80:]).reshape(20, 1)
lr_pred = linear_regression(train_features, train_labels,
                            test_features, test_labels)
