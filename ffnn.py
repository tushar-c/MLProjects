import process_data
import numpy as np
import matplotlib.pyplot as plt
import time


def normalize(X):
    return (X - np.mean(X))/np.std(X)


def sigmoid(x):
    return 1/(1 + np.power(np.e, -x))


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


def gradient_descent_update_l1(x, grad, eta):
    return x - (eta * sigmoid(x)*(1 - sigmoid(x)))


def gradient_descent_update_l2(x, grad, eta):
    return x - (eta * sigmoid(x))


def norm(x):
    return np.power(np.sum(np.power(x, 2)), 1/2)


def forward_prop(X, weights, biases):
    activations = []
    f_1 = np.matmul(weights[0], X) + biases[0]
    af_1 = sigmoid(f_1)
    activations.append(af_1)
    f_2 = np.matmul(weights[1], af_1) + biases[1]
    output = softmax(f_2)
    activations.append(output)
    return [output, activations]


def backward_prop(X, Y, Y_, activations, weights, biases, eta):
    g = grad_mse(X, Y, Y_)
    G = g * sigmoid(activations[1])
    w_2 = np.matmul(G, activations[0].T)
    weights[1] = gradient_descent_update_l2(weights[1], w_2, eta)
    g = np.matmul(weights[1].T, G)
    w_1 = np.matmul(g, X.T)
    weights[0] = gradient_descent_update_l1(weights[0], w_1, eta)
    biases[0] = gradient_descent_update_l1(biases[0], G, eta)
    biases[1] = gradient_descent_update_l2(biases[1], G, eta)
    weights = [weights[0], weights[1]]
    biases = [biases[0], biases[1]]
    return [weights, biases]


def train_net(features, labels, weights, biases, eta=0.02, epochs=500):
    track_error = []
    learned_weights, learned_biases = [], []
    for e in range(epochs):
        correct = 0
        error = 0
        print('train epoch {}'.format(e))
        for f in range(len(features)):
            votes = []
            data_point = features[f].ravel().reshape(features[f].shape[0], 1)
            predict = forward_prop(data_point, weights, biases)
            gradients = backward_prop(data_point, labels[f], predict[0], predict[1], weights, biases, eta)
            weights = gradients[0]
            biases = gradients[1]
            error += mse(predict[0], labels[f])
        track_error.append(mse(predict[0], labels[f]))
    learned_weights = weights
    learned_biases = biases
    return (learned_weights, learned_biases, error, track_error)


# process_data
pre_data = process_data.get_data()
data = process_data.process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80])
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:])
test_labels = np.array(labels[80:]).reshape(20, 1)
train_features = normalize(train_features)
test_features = normalize(test_features)


# set up the architecture
input_dim = 4
hl_1_dim = 10
output_dim = 2
# init weights and biases
W1 = np.array(np.random.normal(size=hl_1_dim * input_dim), dtype=np.float64).reshape(hl_1_dim, input_dim)
W2 = np.array(np.random.normal(size=output_dim * hl_1_dim), dtype=np.float64).reshape(output_dim, hl_1_dim)
b1 = np.array(np.random.normal(size=hl_1_dim), dtype=np.float64).reshape(hl_1_dim, 1)
b2 = np.array(np.random.normal(size=output_dim), dtype=np.float64).reshape(output_dim, 1)
weights = [W1, W2]
biases = [b1, b2]
# train
train = train_net(train_features, train_labels, weights, biases, eta= 0.00001, epochs=100)
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
