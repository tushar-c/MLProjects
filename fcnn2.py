""" A dynamic feed-forward network, used for binary classificiation on the iris dataset for this particular example. """


import process_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


def normalize(X):
    return (X - np.mean(X))/np.std(X)


def sigmoid(x):
    return 1/(1 + np.power(np.e, -x))


def softmax(x, threshold=1.5e30):
    numerator = np.power(np.e, x)
    denominator = np.sum(np.power(np.e, x))
    for i in range(len(numerator)):
        if numerator[i] > threshold:
            numerator[i] = threshold
    if denominator > threshold:
        denominator = threshold
    return numerator/denominator


def predict(X, w):
    return np.matmul(X.T, w)


def mse(y, y_):
    return np.sum(np.power(y - y_, 2))/len(y)


def grad_mse(y, y_):
    return -np.sum((y - y_))


def gradient_descent_update(x, grad, eta):
    return x - (eta * sigmoid(x)*(1 - sigmoid(x)))


def norm(x):
    return np.power(np.sum(np.power(x, 2)), 1/2)

def relu(x):
    return max(0, x)


def max_proba_index(x):
    return x.index(max(x))


def accuracy(test_features, test_labels, weights, biases):
    correct = 0
    n = len(test_features)
    for d in range(n):
        predict = forward_prop(test_features[d], weights, biases)
        votes = [p[0] for p in predict[0]]
        predicted_class = max_proba_index(votes)
        true_class = max_proba_index(list(test_labels[d]))
        if predicted_class == true_class:
            correct += 1
    accuracy = (correct / n) * 100
    return accuracy


def backprop(y, y_, weights, biases, activations, eta):
    g = grad_mse(y, y_) * sigmoid(y_)
    for i in range(len(weights) - 1, -1, -1):
        grad_bias = g
        grad_weights = np.matmul(g, activations[i - 1].T)
        g = np.matmul(weights[i].T, g)
    return [grad_weights, grad_bias]


def forward_prop(X, weights, biases):
    activations = []
    inp = X.reshape(X.shape[0], 1)
    for w in range(len(weights) - 1):
        output = np.matmul(weights[w], inp) + biases[w]
        activation = sigmoid(output)
        activations.append(activation)
        inp = activation
    final_output = softmax(np.matmul(weights[-1], inp) + biases[-1])
    activations.append(final_output)
    return [final_output, activations]


def train_net(X, y, weights, biases, epochs=100, eta=0.002):
    accuracy_list = []
    hyperparam_tracker = []
    for e in range(1, epochs + 1):
        epoch_mse = 0
        print('train epoch = {}'.format(e), end = ", ")
        for t in range(len(X)):
            prediction = forward_prop(X[t], weights, biases)
            activations = prediction[1]
            epoch_mse += mse(y[t], prediction[0])
            gradients = backprop(y[t], prediction[0], weights, biases, activations, eta)
            for i in range(len(weights)):
                weights[i] = gradient_descent_update(weights[i], gradients[0], eta)
                biases[i] = gradient_descent_update(biases[i], gradients[1], eta)
        acc = accuracy(X, y, weights, biases)
        accuracy_list.append(acc)
        hyperparam_tracker.append((acc, weights, biases))
        print('train accuracy = {}%'.format(acc), end=", ")
        print('mse = {}'.format(epoch_mse))
    print('max train accuracy was {}%'.format(max(accuracy_list)))
    accs = accuracy_list.index(max(accuracy_list))
    highest_acc_params = hyperparam_tracker[accs]
    max_acc_w_b = [highest_acc_params[1], highest_acc_params[2]]
    return [weights, biases, max_acc_w_b]


if __name__ == '__main__':
    # process_data
    pre_data = process_data.get_data()
    data = process_data.process_data_2(pre_data)
    features = data[0]
    labels = data[1]
    train_features = np.array(features[:80]).reshape(80, 4)
    test_features = np.array(features[80:]).reshape(20, 4)
    train_labels = labels[:80]
    test_labels = labels[80:]
    train_features = normalize(train_features)
    test_features = normalize(test_features)


    # set up the architecture and hyperparameters
    epochs = 1000
    eta = 0.002
    input_dim = train_features.shape[1]
    output_dim = 2
    hidden_layers = 5
    hl_dimensions = [50, 40, 40, 30, 30]
    # init weights and biases
    W1 = np.array(np.random.normal(size=hl_dimensions[0] * input_dim), dtype=np.float64).reshape(hl_dimensions[0], input_dim)
    b1 = np.array(np.random.normal(size=hl_dimensions[0])).reshape(hl_dimensions[0], 1)
    weights = [W1]
    biases = [b1]
    for i in range(1, hidden_layers):
        curr_dim = hl_dimensions[i]
        prev_dim = hl_dimensions[i - 1]
        w_matrix = np.array(np.random.normal(size=curr_dim * prev_dim), dtype=np.float64).reshape(curr_dim, prev_dim)
        b_matrix = np.array(np.random.normal(size=curr_dim), dtype=np.float64).reshape(curr_dim, 1)
        biases.append(b_matrix)
        weights.append(w_matrix)

    output_weights = np.array(np.random.normal(size=hl_dimensions[-1] * output_dim), dtype=np.float64).reshape(output_dim, hl_dimensions[-1])
    output_biases = np.array(np.random.normal(size=output_dim), dtype=np.float64).reshape(output_dim, 1)
    weights.append(output_weights)
    biases.append(output_biases)

    # train
    summary = True
    if summary:
        print('training for {} epochs with learning rate {}'.format(epochs, eta))
        print('Network architecture: ')
        print('layers = {} (including input and output layers)'.format(hidden_layers + 2))
        print('-' * 50)
        print('input layer : shape = ({}, 1)'.format(input_dim))
        print('hidden layer {}, weights: {} , biases : {}'.format(1, (hl_dimensions[0], input_dim), hl_dimensions[0]))
        for i in range(1, hidden_layers):
            print('hidden layer {}, weights: {} , biases : {}'.format(i + 1, (hl_dimensions[i], hl_dimensions[i-1]), hl_dimensions[i]))
        print('output layer: weights: {} , biases : {}'.format((output_dim, hl_dimensions[-1]), hl_dimensions[-1]))
    print('-' * 50)
    train = train_net(train_features, train_labels, weights, biases, eta=eta, epochs=epochs)
    print()

    # display results
    show_coeffs = False
    if show_coeffs:
        print('learned weights.....')
        print(train[0])
        print('learned biases......')
        print(train[1])
    # test
    correct = 0
    n = len(test_labels)
    show_test_mse = True
    test_mse = 0
    for d in range(len(test_labels)):
        """ using max acc weights and biases. to use last found weights and biases,
            predict = forward_prop(test_features[d], train[0], train[1])
        """
        predict = forward_prop(test_features[d], train[2][0], train[2][1])
        test_mse += mse(test_labels[d], predict[0])
        votes = [p[0] for p in predict[0]]
        predicted_class = max_proba_index(votes)
        true_class = max_proba_index(list(test_labels[d]))
        if predicted_class == true_class:
            correct += 1

    accuracy = (correct / n) * 100
    if show_test_mse:
        print('test mse = {}'.format(test_mse))
    print('test accuracy = {}%'.format(accuracy))
    plt_error = False
    if plt_error:
        plt.plot(train[3])
        plt.xlabel('training error over epochs')
        plt.ylabel('mse over epochs')
        plt.show()
