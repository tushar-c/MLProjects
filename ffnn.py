""" A dynamic feed-forward network, used for binary classificiation on the iris dataset for this particular example. """


""" A dynamic feed-forward network, used for binary classificiation on the iris dataset for this particular example. """


import process_data
import numpy as np
import matplotlib.pyplot as plt
import time


def normalize(X):
    return (X - np.mean(X))/np.std(X)


def sigmoid(x):
    return 1/(1 + np.power(np.e, -x))


def sigmoid_derivative(x):
    return (1 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    return np.power(np.e, x)/np.sum(np.power(np.e, x))


def predict(X, w):
    return np.matmul(X.T, w)


def mse(y, y_):
    return np.sum(np.power(y - y_, 2))/len(y)


def grad_mse(X, Y, Y_):
    total = 0
    for pt in range(len(Y)):
        total += 2 * X[pt] * (Y_[pt] - Y[pt])
    return total


def gradient_descent_update(x, grad, eta):
    return x - (eta * sigmoid(x)*(1 - sigmoid(x)))


def gradient_descent_update_l2(x, grad, eta):
    return x - (eta * sigmoid(x))


def norm(x):
    return np.power(np.sum(np.power(x, 2)), 1/2)


def backprop(y, y_, weights, biases, activations, eta):
    g = mse(y, y_) * sigmoid(y_)
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
    for e in range(1, epochs + 1):
        print('train epoch = {}'.format(e))
        for t in range(len(X)):
            prediction = forward_prop(X[t], weights, biases)
            activations = prediction[1]
            gradients = backprop(y[t], prediction[0], weights, biases, activations, eta)
            for i in range(len(weights)):
                weights[i] = gradient_descent_update(weights[i], gradients[0], eta)
                biases[i] = gradient_descent_update(biases[i], gradients[1], eta)
    return [weights, biases]


# process_data
pre_data = process_data.get_data()
data = process_data.process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80]).reshape(80, 4)
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:]).reshape(20, 4)
test_labels = np.array(labels[80:]).reshape(20, 1)
train_features = normalize(train_features)
test_features = normalize(test_features)


# set up the architecture
input_dim = 4
output_dim = 2
hidden_layers = 5
hl_dimensions = [250, 200, 150, 100, 50]
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
epochs = 100
eta = 0.00001
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
# display results
show_coeffs = False
show_mse = False
if show_coeffs:
    print('learned weights.....')
    print(train[0])
    print('learned biases......')
    print(train[1])
if show_mse:
    print('total mse = {}'.format(train[2]))

# test
correct = 0
n = len(test_labels)

for d in range(len(test_labels)):
    predict = forward_prop(test_features[d], train[0], train[1])
    votes = [p[0] for p in predict[0]]
    predicted_class = votes.index(max(votes))
    if predicted_class == test_labels[d]:
        correct += 1

accuracy = (correct / n) * 100
print('test accuracy = {}%'.format(accuracy))
plt_error = False
if plt_error:
    plt.plot(train[3])
    plt.xlabel('training error over epochs')
    plt.ylabel('mse over epochs')
    plt.show()
