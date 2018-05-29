import numpy as np
import conv_utils
import matplotlib.pyplot as plt

# get data
mnist_train_data = conv_utils.get_mnist_data()
train_features = mnist_train_data[0]
train_labels = mnist_train_data[1]
# hyperparameters
output_classes = 10
time_steps = 28
hidden_neurons = 100
eta = 0.095
features = 28
# init loss tracker
all_labels = [np.argmax(j) for j in train_labels]
loss_tracker = {num: [] for num in all_labels}

U = np.random.randn(hidden_neurons, features)
W = np.random.randn(hidden_neurons, hidden_neurons)
V = np.random.randn(output_classes, hidden_neurons)
b = np.zeros((hidden_neurons, 1))
c = np.zeros((output_classes, 1))
h = np.zeros((hidden_neurons, 1))

# initialize parameters for adagrad update 
mem_U, mem_W, mem_V = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
mem_b, mem_c = np.zeros_like(b), np.zeros_like(c)

delta = 1e-8

for d in range(len(train_features)):
    loss = 0
    # caches to use during backward pass
    h_cache, grad_loss_o_cache = [], []
    sequential_image = [train_features[d][:, j].reshape(features, 1) for j in range(features)]
    print('sample {} / {} ; true label = {} ;'.format(d + 1, len(train_features), np.argmax(train_labels[d])), end=' ')
    # forward pass
    for t in range(time_steps):
        a_t = b + np.matmul(W, h) + np.matmul(U, sequential_image[t])
        h = np.tanh(a_t)
        o_t = c + np.matmul(V, h)
        y_t = conv_utils.stable_softmax(o_t)
        h_cache.append(h)
        loss += -np.log(y_t[np.argmax(train_labels[d])] + delta)
        one_hot_vector = np.zeros((output_classes, 1))
        one_hot_vector[np.argmax(train_labels[d])] = 1
        error = y_t - one_hot_vector
        grad_loss_o_cache.append(error)

    # backward pass
    grad_loss_U, grad_loss_W, grad_loss_V = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    grad_loss_b, grad_loss_c = np.zeros_like(b), np.zeros_like(c)
    grad_h_final_time_step = np.matmul(V.T, grad_loss_o_cache[-1])
    prev_layer_grad_loss_h = grad_h_final_time_step
    for t in range(time_steps - 2, -1, -1):
        diagonal_matrix = np.zeros((hidden_neurons, hidden_neurons))
        for j in range(h_cache[t + 1].shape[0]):
            diagonal_matrix[j][j] = 1 - np.power(h_cache[t + 1][j, 0], 2)

        term1 = np.matmul(W.T, prev_layer_grad_loss_h).reshape(hidden_neurons, 1)
        term2 = np.matmul(term1.T, diagonal_matrix).T
        term3 = np.matmul(V.T, grad_loss_o_cache[t])
        grad_loss_h_t = term2 + term3
        prev_layer_grad_loss_h = grad_loss_h_t

        grad_loss_c += grad_loss_o_cache[t]
        grad_loss_b += np.matmul(diagonal_matrix, prev_layer_grad_loss_h)
        grad_loss_V += np.matmul(grad_loss_o_cache[t], h_cache[t].T)
        grad_loss_W += np.matmul(diagonal_matrix, np.matmul(prev_layer_grad_loss_h, h_cache[t - 1].T))
        grad_loss_U += np.matmul(diagonal_matrix, np.matmul(prev_layer_grad_loss_h, sequential_image[t].T))

        # gradient clipping to deal with exploding gradients
        for params in [grad_loss_U, grad_loss_W, grad_loss_b, grad_loss_V, grad_loss_c]:
            np.clip(params, -5, 5, out=params)

    # perform adagrad update
    for param, grad_param, mem in zip([U, W, V, b, c],
                                          [grad_loss_U, grad_loss_W, grad_loss_V, grad_loss_b, grad_loss_c],
                                          [mem_U, mem_W, mem_V, mem_b, mem_c]):
        mem += grad_param ** 2
        param += -eta * grad_param / np.sqrt(mem + delta)

    loss_tracker[np.argmax(train_labels[d])].append(loss[0])
    print('loss = {}'.format(loss[0]))


plot = False
if plot:
    for entry in loss_tracker:
        plt.plot(loss_tracker[entry])
        plt.xlabel('error for label: {}'.format(entry))
        plt.show()
    
