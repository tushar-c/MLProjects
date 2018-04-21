import numpy as np
import ffnn
import random
import matplotlib.pyplot as plt


def conv2d(img, kernel, bias, s, activation=False):
    x, y = img.shape[0], img.shape[1]
    k_x, k_y = kernel.shape[0], kernel.shape[1]
    if k_x > x or k_y > y or s > x or s > y:
        print('Warning! Kernel size or stride is greater than input size')
        print('kernel: {}, {}'.format(k_x, k_y))
        print('stride = {}'.format(s))
        print('input = {}, {}'.format(x, y))
        print('returning None')
        return None
    spat_dim = int(np.floor((x - k_x)/s) + 1)
    V = np.full((spat_dim, spat_dim), 0)
    x_spatial, y_spatial = 0, 0
    for x in range(spat_dim):
        for y in range(spat_dim):
            img_slice = img[x_spatial:k_x, y_spatial:k_y]
            conv_out = np.sum(img_slice * kernel)
            V[x, y] = conv_out + bias
            y_spatial += s
            k_y += s
        x_spatial = x_spatial + s
        k_x = k_x + s
        y_spatial, k_y = 0, kernel.shape[1]
    if activation:
        V = ffnn.sigmoid(V)
    return V


def avg_pooling(img, pool_window, s):
    x, y = img.shape[0], img.shape[1]
    p_x, p_y = pool_window[0], pool_window[1]
    if p_x > x or p_y > y or s > x or s > y:
        print('Warning! Pool window size or stride is greater than input size')
        print('pool window: {}, {}'.format(p_x, p_y))
        print('stride = {}'.format(s))
        print('input = {}, {}'.format(x, y))
        print('returning None')
        return None
    spat_dim = int(np.floor((x - p_x)/s) + 1)
    if spat_dim == 1:
        pooled = 0
    else:
        pooled = np.full((spat_dim, spat_dim), 0)
    x_spatial, y_spatial = 0, 0
    for x in range(spat_dim):
        for y in range(spat_dim):
            img_slice = img[x_spatial:p_x, y_spatial:p_y]
            flattened_pool = np.sum(img_slice.ravel())
            n = len(img_slice.ravel())
            pool_out = flattened_pool / n
            if spat_dim == 1:
                pooled = pool_out
            else:
                pooled[x, y] = pool_out
            y_spatial += s
            p_y += s
        x_spatial = x_spatial + s
        p_x = p_x + s
        y_spatial, p_y = 0, pool_window[1]
    return pooled


def relu(x):
    return max(0, x)


def conv_pass(inp, kernel, bias, strides, passes=1, show_flag=False):
    outputs = []
    for p in range(passes):
        o = int(np.floor((inp.shape[0] - kernel[p].shape[0]) / strides) + 1)
        conv_pass = conv2d(inp, kernel[p], bias[p], strides)
        o_pool = int(np.floor((o - kernel[p].shape[0]) / strides) + 1)
        if show_flag:
            print('after pooling pass {}, input shape : {}, output shape : {}'.format(p+1, inp.shape, (o_pool, o_pool)))
        pooled_conv = avg_pooling(conv_pass, (2, 2), strides)
        outputs.append(avg_pooling(conv_pass, (2, 2), strides))
        inp = pooled_conv
    return inp


def activation(x):
    return ffnn.sigmoid(x)


def grad_sigmoid(x):
    return activation(x) * (1 - activation(x))


def sigmoid(x):
    return ffnn.sigmoid(x)


def softmax(x):
    return ffnn.softmax(x)


def final_error(y, y_):
    return ffnn.grad_mse(y, y_)


def mse(y, y_):
    return np.sum(np.power(y - y_, 2)) / 2


def grad_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def grad_mse(y, y_):
    return -np.sum((y - y_))


def gradient_descent_update(x, grad, eta):
    return x - (eta * grad)


def stable_log(x):
    delta = 0.00000003
    if x == 0:
        return np.log(x + delta)
    return np.log(x)


def upscale(curr_array, desired_array):
    dupl_desired_array = np.zeros(desired_array.shape)
    for i in range(desired_array.shape[0]):
        for j in range(desired_array.shape[1]):
            r_assign, c_assign = int(np.ceil(i/2)), int(np.ceil(j/2))
            if r_assign >= curr_array.shape[0]:
                r_assign = curr_array.shape[0] - 1
            if r_assign >= curr_array.shape[1]:
                r_assign = curr_array.shape[1] - 1
            if c_assign >= curr_array.shape[0]:
                c_assign = curr_array.shape[0] - 1
            if c_assign >= curr_array.shape[1]:
                c_assign = curr_array.shape[1] - 1
            dupl_desired_array[i][j] = curr_array[r_assign, c_assign]
    return dupl_desired_array / 4


if __name__ == "__main__":
    # init stuff
    output_classes = 2
    eta = 0.1
    passes = 100
    x_input, y_input = 28, 28
    x_kernel, y_kernel = 5, 5
    pool_stride = 2
    pool_window = [2, 2]
    conv_stride = 1
    dataset_size = 200
    dummy_data = [np.array([np.random.normal() for i in range(x_input * y_input)]).reshape(x_input, y_input) for j in range(dataset_size)]
    dummy_answers = []
    for k in range(len(dummy_data)):
        chosen_index = random.choice((0, 1))
        if chosen_index == 1:
            d_lbl = np.array([0, 1]).reshape(2, 1)
        else:
            d_lbl = np.array([1, 0]).reshape(2, 1)
        dummy_answers.append(d_lbl)
    y = np.array([1, 0]).reshape(2, 1)
    k1 = np.array([np.random.normal() for i in range(x_kernel * y_kernel)]).reshape(x_kernel, y_kernel)
    k2 = np.array([np.random.normal() for i in range(x_kernel * y_kernel)]).reshape(x_kernel, y_kernel)
    bias_1, bias_2 = 0.5, 0.3
    output_layer_weights = np.array([np.random.normal() for i in range(output_classes * 16)]).reshape(output_classes, 16)
    output_layer_biases = np.array([0 for i in range(output_classes)]).reshape(output_classes, 1)
    for p in range(passes):
        pass_mse = 0
        pass_cross_entropy = 0
        pass_accuracy = 0
        for d in range(len(dummy_data)):
            # forward pass
            c1 = conv2d(dummy_data[d], k1, bias_1, 1)
            ac1 = sigmoid(c1)
            pool1 = avg_pooling(ac1, pool_window, pool_stride)
            c2 = conv2d(pool1, k2, bias_2, conv_stride)
            ac2 = sigmoid(c2)
            pool2 = avg_pooling(ac2, pool_window, pool_stride)
            # the 'layers' are actually weight matrices
            fc_layer = np.ravel(pool2).reshape(pool2.shape[0] * pool2.shape[1], 1)
            final_output = softmax(np.matmul(output_layer_weights, fc_layer) + output_layer_biases)
            error = (final_output - dummy_answers[d]) * sigmoid(final_output)
            prediction = list(final_output).index(max(final_output))
            if prediction == list(dummy_answers[d]).index(max(dummy_answers[d])):
                pass_accuracy += 1
            pass_mse += mse(prediction, dummy_answers[d])
            # backward pass
            delta_output_layer_weights = np.matmul(error, fc_layer.T)
            output_layer_weights = gradient_descent_update(output_layer_weights, delta_output_layer_weights, eta)
            output_layer_biases = gradient_descent_update(output_layer_biases, error, eta)
            pool2_error = np.matmul(output_layer_weights.T, error)
            matrix_pool2_error = pool2_error.reshape(pool2.shape[0], pool2.shape[1])
            # upscale
            delta_c2 = upscale(matrix_pool2_error, c2)
            activated_delta_c2 = sigmoid(delta_c2)
            delta_k2 = conv2d(np.rot90(pool1, 2), activated_delta_c2, bias_2, 1)
            # kernel2 update
            k2 = gradient_descent_update(k2, delta_k2, eta)
            # bias2 update
            bias_2 = np.sum(activated_delta_c2)
            pool1_error = conv2d(activated_delta_c2, np.rot90(k2, 2), bias_1, 1)
            delta_c1 = upscale(pool1_error, c1)
            activated_delta_c1 = sigmoid(delta_c1)
            delta_k1 = conv2d(np.rot90(dummy_data[d], 2), activated_delta_c1, bias_1, 1)
            # kernel1 update
            k1 = gradient_descent_update(k1, delta_k1, eta)
            bias_1 = np.sum(activated_delta_c1)
        print('pass = {}, pass_mse = {}, correct predictions = {}, pass_accuracy = {}'.format(p + 1, pass_mse, pass_accuracy, pass_accuracy / len(dummy_answers)))

