import process_data
import numpy as np
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


def gradient_descent_update_l1(x, grad, eta):
    return x - (eta * sigmoid(x)*(1 - sigmoid(x)))


def gradient_descent_update_l2(x, grad, eta):
    return x - (eta * sigmoid(x))


def forward_pass(X, weights, biases):
    f_1 = np.matmul(weights[0], X) + biases[0]
    af_1 = sigmoid(f_1)
    f_2 = np.matmul(weights[1], af_1) + biases[1]
    output = softmax(f_2)
    return output


def backward_pass(Y, weights, biases, eta):
    Y = sigmoid(Y)
    w_2 = np.matmul(weights[1].T, Y)
    w_1 = np.matmul(weights[0].T, w_2)
    weights[0] = gradient_descent_update_l1(weights[0], w_1, eta)
    weights[1] = gradient_descent_update_l2(weights[1], w_2, eta)
    biases[0] = gradient_descent_update_l1(biases[0], Y, eta)
    biases[1] = gradient_descent_update_l2(biases[1], Y, eta)
    weights = [weights[0], weights[1]]
    biases = [biases[0], biases[1]]
    return [weights, biases]


def train_net(features, labels, weights, biases, eta=0.02, epochs=500):
    error = 0
    learned_weights, learned_biases = [], []
    for e in range(epochs):
        correct = 0
        print('train epoch {}'.format(e))
        for f in range(len(features)):
            votes = []
            data_point = features[f].ravel().reshape(features[f].shape[0], 1)
            predict = forward_pass(data_point, weights, biases)
            gradients = backward_pass(predict, weights, biases, eta)
            weights = gradients[0]
            biases = gradients[1]
            error += mse(predict, labels[f])
    learned_weights = weights
    learned_biases = biases
    return (learned_weights, learned_biases, error)


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
train = train_net(train_features, train_labels, weights, biases, eta= 0.001, epochs=100)
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
    predict = forward_pass(test_features[d], train[0], train[1])
    votes = [p[0] for p in predict]
    predicted_class = votes.index(max(votes))
    if predicted_class == test_labels[d]:
        correct += 1

accuracy = (correct / n) * 100
print('test accuracy = {}%'.format(accuracy))
