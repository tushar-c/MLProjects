import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import process_data
<<<<<<< HEAD
from sklearn.linear_model import LinearRegression

=======
>>>>>>> 1ffb4591fe66ba3106a70c6392a0ab0a817c8b4e


# OLS Linear Regression
def linear_regression(X, Y, X_, Y_):
    # estimate for the weight vector B, can be found by minimizing (y - XB).T * (y - XB)
    B = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
    predictions = []
    # iterate over the test set
    for entry in range(len(X_)):
        # get predicition
        p = np.matmul(B.T, X_[entry])
        # threshold for binary classification
        if p >= 0.5:
            pred = 1
        elif p < 0.5:
            pred = 0
        predictions.append(pred)
    predictions = np.array(predictions)
    # compute MSE 
    error = np.sum(np.power(predictions - test_labels, 2)) / len(features)
    return (B, error, predictions)


<<<<<<< HEAD
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
lr_pred = linear_regression(train_features, train_labels,
                            test_features, test_labels)
