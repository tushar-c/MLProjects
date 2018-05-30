""" A dynamic feed-forward network, used for multi class classificiation on the MNIST dataset for this particular example. """

import conv_utils
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist


def normalize(X):
    return (X - np.mean(X)) / np.std(X)


def sigmoid(x):
    return conv_utils.stable_sigmoid(x)


def grad_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    return np.power(np.e, x) / np.sum(np.power(np.e, x))


def grad_softmax(x):
    return softmax(x) * (1 - softmax(x))


def predict(X, w):
    return np.matmul(X.T, w)


def mse(y, y_):
    return np.sum(np.power(y - y_, 2)) / 2


def gradient_descent_update(x, grad, eta):
    return x - (eta * grad)


def norm(x):
    return np.power(np.sum(np.power(x, 2)), 1 / 2)


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


def backprop(y, y_, preactivated_output, weights, biases, activations, eta):
    y = y.reshape(y_.shape[0], 1)
    g = (y_ - y) * conv_utils.sigmoid_gradient(preactivated_output)
    grad_weight_cache, grad_bias_cache = [], []
    for i in range(len(weights) - 1, 0, -1):
        grad_bias = g
        grad_weights = np.matmul(g, activations[i].T)
        grad_weight_cache.append(grad_weights)
        grad_bias_cache.append(grad_bias)
        g = np.matmul(weights[i].T, g)
    grad_bias_cache.append(g)
    grad_weight_cache.append(np.matmul(g, activations[0].T))
    final_grad_weight_cache = [i for i in reversed(grad_weight_cache)]
    final_grad_bias_cache = [j for j in reversed(grad_bias_cache)]
    return [final_grad_weight_cache, final_grad_bias_cache]


def forward_prop(X, weights, biases):
    inp = X.reshape(X.shape[0] * X.shape[1], 1)
    activations = [inp]
    for w in range(len(weights) - 1):
        output = np.matmul(weights[w], inp) + biases[w]
        activation = conv_utils.stable_sigmoid(output)
        activations.append(activation)
        inp = activation
    preactivated_output = np.matmul(weights[-1], inp) + biases[-1]
    final_output = conv_utils.stable_sigmoid(preactivated_output)
    activations.append(final_output)
    return [final_output, activations, preactivated_output]


def train_net(X, y, X_test, y_test, weights, biases, epochs=12, eta=0.002):
    accuracy_list = []
    hyperparam_tracker = []
    mse_tracker = []
    for e in range(1, epochs + 1):
        epoch_mse = 0
        for t in range(len(X)):
            print('epoch {} / {} ; sample {} / {}'.format(e, epochs, t + 1, len(X)))
            prediction = forward_prop(X[t], weights, biases)
            activations = prediction[1]
            epoch_mse += mse(np.argmax(y[t]), np.argmax(prediction[0]))
            gradients = backprop(y[t], prediction[0], prediction[2], weights, biases, activations, eta)
            for i in range(len(weights)):
                weights[i] = gradient_descent_update(weights[i], gradients[0][i], eta)
            for b in range(len(biases)):
                biases[b] = gradient_descent_update(biases[b], gradients[1][b], eta)
        print('epoch {} / {} ; evaluating accuracy'.format(e, epochs))
        acc = accuracy(X, y, weights, biases)
        accuracy_list.append(acc)
        hyperparam_tracker.append((acc, weights, biases))
        print('train accuracy = {}%'.format(acc), end=", ")
        print('epoch mse = {}'.format(epoch_mse))
        mse_tracker.append(epoch_mse)
    print('max train accuracy was found to be {}%'.format(max(accuracy_list)))
    accs = accuracy_list.index(max(accuracy_list))
    highest_acc_params = hyperparam_tracker[accs]
    max_acc_w_b = [highest_acc_params[1], highest_acc_params[2]]
    return [weights, biases, max_acc_w_b, mse_tracker, accuracy_list]


if __name__ == '__main__':
    # process_data
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    train_features = x_train
    test_features = x_test
    train_labels = y_train
    test_labels = y_test
    train_features /= 255
    test_features /= 255
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)
    test_labels = test_labels.reshape(10000, 10)
    train_labels = train_labels.reshape(60000, 10)
    train_features = train_features[:10000]
    train_labels = train_labels[:10000]

    # set up the architecture and hyperparameters
    epochs = 30
    eta = 0.0275
    input_dim = 784
    output_dim = 10
    hidden_layers = 1
    hl_dimensions = [30]
    # init weights and biases
    W1 = np.array(np.random.normal(size=hl_dimensions[0] * input_dim), dtype=np.float64).reshape(hl_dimensions[0],
                                                                                                 input_dim)
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

    output_weights = np.array(np.random.normal(size=hl_dimensions[-1] * output_dim), dtype=np.float64).reshape(output_dim,
                                                                                                               hl_dimensions[
                                                                                                                   -1])
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
            print('hidden layer {}, weights: {} , biases : {}'.format(i + 1, (hl_dimensions[i], hl_dimensions[i - 1]),
                                                                      hl_dimensions[i]))
        print('output layer: weights: {} , biases : {}'.format((output_dim, hl_dimensions[-1]), hl_dimensions[-1]))
    print('-' * 50)
    train = train_net(train_features, train_labels, test_features, test_labels, weights, biases, eta=eta, epochs=epochs)
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
        """ 
        using max accuracy weights and biases. to use last found weights and biases, do:
        predict = forward_prop(test_features[d], train[0], train[1])
        for max weights and biases, do:
        predict = forward_prop(test_features[d], train[2][0], train[2][1])
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
    plt_error = True
    plt_accuracy = True
    if plt_error:
        plt.plot(train[3])
        plt.xlabel('epochs')
        plt.ylabel('mse over epochs')
        plt.show()
    if plt_accuracy:
        plt.plot(train[4])
        plt.xlabel('epochs')
        plt.ylabel('train accuracy over epochs')
        plt.show()
