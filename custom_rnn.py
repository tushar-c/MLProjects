import numpy as np
import conv_utils
import matplotlib.pyplot as plt

mnist_train_data = conv_utils.get_mnist_data(sliced=1000)
train_features = mnist_train_data[0]
train_labels = mnist_train_data[1]
output_classes = 10
time_steps = 10
eta = 0.000001
init_x = train_features[0]
ts_shape = init_x.shape
ts_1 = init_x.shape[0]
ts_2 = init_x.shape[1]
ts = init_x.shape[0] * init_x.shape[1]
all_labels = [np.argmax(j) for j in train_labels]
loss_tracker = {num: [] for num in all_labels}


init_h = np.array([np.random.normal() for m in range(ts)]).reshape(ts, 1)
h = init_h
W = np.array([np.random.normal() for k in range(ts ** 2)]).reshape(ts, ts)
U = np.array([np.random.normal() for i in range(ts ** 2)]).reshape(ts, ts)
b = np.array([np.random.normal() for j in range(ts)]).reshape(ts, 1)
V_shape = (b + np.matmul(W, h) + np.matmul(U, init_x.ravel())).shape
V = np.array([np.random.normal() for l in range(output_classes * V_shape[1])]).reshape(output_classes, V_shape[1])
c_shape = np.matmul(V, h).shape
c = np.array([np.random.normal() for n in range(c_shape[0] * c_shape[1])]).reshape(c_shape[0], c_shape[1])
U_cache, W_cache, V_cache, h_cache, grad_loss_o_cache = [U], [W], [V], [h], []

mem_U, mem_W, mem_V = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
mem_b, mem_c = np.zeros_like(b), np.zeros_like(c)

delta = 1e-10

for d in range(len(train_features)):
    loss = 0
    x = train_features[d].reshape(train_features[d].ravel().shape[0], 1)
    print('sample {} / {} ; true label = {} ;'.format(d + 1, len(train_features), np.argmax(train_labels[d])), end=' ')
    # forward pass
    for t in range(time_steps):
        # print('sample {} / {}, time step {} / {}'.format(d + 1, len(train_features), t + 1, time_steps))
        a_t = b + np.matmul(W, h) + np.matmul(U, x)
        h = np.tanh(a_t)
        o_t = c + np.matmul(V, h)
        y_t = conv_utils.stable_softmax(o_t)
        h_cache.append(h)
        loss += -np.log(y_t[np.argmax(train_labels[d])] + delta)
        error = y_t - np.ones(output_classes).reshape(output_classes, 1)
        grad_loss_o_cache.append(error)
    # backward pass
    grad_loss_h_cache = []
    grad_h = np.matmul(V.T, grad_loss_o_cache[-1])
    grad_loss_h_cache.append(grad_h)
    for t in range(time_steps - 1, -1, -1):
        term1 = np.matmul(grad_h, np.diag(1 - np.power(h_cache[t], 2))).reshape(ts, 1)
        term2 = np.matmul(V.T, grad_loss_o_cache[t])
        grad_loss_h_t = np.matmul(W.T, term1) + term2
        grad_h = grad_loss_h_t
        grad_loss_h_cache.append(grad_h)

    for l in range(len(grad_loss_h_cache)):
        h_cache[t] = conv_utils.gradient_descent_update(h_cache[t], grad_loss_h_cache[t], eta)

    grad_loss_c = 0
    grad_loss_b = 0
    grad_loss_V = 0
    grad_loss_W = 0
    grad_loss_U = 0

    for g in grad_loss_o_cache:
        grad_loss_c += g

    for g in range(len(grad_loss_h_cache)):
        grad_loss_b_term = 1 - np.power(h_cache[g], 2)
        grad_loss_b_array = np.zeros((ts, ts))
        for i in range(len(grad_loss_b_term)):
            grad_loss_b_array[i][i] = grad_loss_b_term[i]
        grad_loss_b += np.matmul(grad_loss_b_array, grad_loss_h_cache[g])

    for g in range(len(grad_loss_o_cache)):
        grad_loss_V += np.matmul(grad_loss_o_cache[g], h_cache[g].T)

    for g in range(1, len(h_cache)):
        grad_loss_W_term = 1 - np.power(h_cache[g], 2)
        grad_loss_W_array = np.zeros((ts, ts))
        for i in range(len(grad_loss_W_term.ravel())):
            grad_loss_W_array[i][i] = grad_loss_W_term[i]
        grad_loss_W += np.matmul(grad_loss_W_array, np.matmul(grad_loss_h_cache[g], h_cache[g - 1].T))

    for g in range(len(h_cache)):
        grad_loss_U_term = 1 - np.power(h_cache[g], 2)
        grad_loss_U_array = np.zeros((ts, ts))
        for i in range(len(grad_loss_U_term.ravel())):
            grad_loss_U_array[i][i] = grad_loss_U_term[i]
        grad_loss_U += np.matmul(grad_loss_U_array, np.matmul(grad_loss_h_cache[g], x.T))

    for params in [grad_loss_U, grad_loss_W, grad_loss_b, grad_loss_V, grad_loss_c]:
        np.clip(params, -1, 1, out=params)

    h_cache = [init_h]
    grad_loss_o_cache = []

    for param, grad_param, mem in zip([U, W, V, b, c], [grad_loss_U, grad_loss_W, grad_loss_V, grad_loss_b, grad_loss_c],
                                      [mem_U, mem_W, mem_V, mem_b, mem_c]):
        mem += grad_param ** 2
        param += -eta * grad_param / np.sqrt(mem + 1e-8)

    loss_tracker[np.argmax(train_labels[d])].append(loss[0])
    print('loss = {}'.format(loss[0]))

correct = 0
for d in range(len(train_features)):
    x = train_features[d].reshape(train_features[d].ravel().shape[0], 1)
    a_t = b + np.matmul(W, h) + np.matmul(U, x)
    h = np.tanh(a_t)
    o_t = c + np.matmul(V, h)
    y_t = conv_utils.stable_softmax(o_t)
    if np.argmax(y_t) == np.argmax(train_labels[d]):
        correct += 1

# print('correct : {} / {}'.format(correct, len(train_labels)))
for entry in loss_tracker:
    plt.plot(loss_tracker[entry])
    plt.xlabel('error for label {}'.format(entry))
    plt.show()
