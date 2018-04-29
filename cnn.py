import numpy as np
import ffnn
import random


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
    V = np.full((spat_dim, spat_dim), 0, dtype=np.float)
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


def max_pooling(img, pool_window, s, threshold=1.5e30):
    x, y = img.shape[0], img.shape[1]
    p_x, p_y = pool_window[0], pool_window[1]
    if p_x > x or p_y > y or s > x or s > y:
        print('Warning! Pool window size or stride is greater than input size')
        print('pool window: {}, {}'.format(p_x, p_y))
        print('stride = {}'.format(s))
        print('input = {}, {}'.format(x, y))
        print('returning None')
        return None
    spat_dim = int(np.floor((x - p_x) / s) + 1)
    if spat_dim == 1:
        pooled = 0
    else:
        pooled = np.full((spat_dim, spat_dim), 0)
    x_spatial, y_spatial = 0, 0
    for x in range(spat_dim):
        for y in range(spat_dim):
            img_slice = img[x_spatial:p_x, y_spatial:p_y]
            pool_out = np.max(img_slice)
            if pool_out > threshold:
                pool_out = threshold
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


def avg_pooling(img, pool_window, s, threshold=1e15):
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
        pooled = np.full((spat_dim, spat_dim), 0, dtype=np.float)
    x_spatial, y_spatial = 0, 0
    for x in range(spat_dim):
        for y in range(spat_dim):
            img_slice = img[x_spatial:p_x, y_spatial:p_y]
            flattened_pool = np.sum(img_slice.ravel())
            if flattened_pool > threshold:
                flattened_pool = threshold
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


def grad_relu(x):
    orig_shape = (x.shape[0], x.shape[1])
    x = x.ravel()
    for i in range(len(x)):
        if x[i] == 0 or x[i] < 0:
            x[i] = 0
        else:
            x[1] = 1
    x = x.reshape(orig_shape[0], orig_shape[1])
    return x


def relu(x):
    orig_shape = (x.shape[0], x.shape[1])
    x = x.ravel()
    for i in range(len(x)):
        x[i] = max(0, x[i])
    x = x.reshape(orig_shape[0], orig_shape[1])
    return x


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


def infer_output_layer_shape(input_shape, conv_layers, kernels, poolings, kernel_strides=1, pool_stride=2):
    curr_input_shape = input_shape
    for c in range(conv_layers):
        conv_shape = int(np.floor((curr_input_shape - kernels[c].shape[0])/kernel_strides) + 1)
        pool_shape = int(np.floor((conv_shape - poolings[c][0])/pool_stride) + 1)
        curr_input_shape = pool_shape
    return curr_input_shape


def normalize(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])


def l2_norm(x):
    return np.power(np.sum(np.power(x, 2)), 1/2)


def forward_conv_pass(input_, conv_layers, kernels, biases, poolings, kernel_strides=1, pool_stride=2, threshold=1e20):
    curr_input = input_
    convolution_cache = []
    activation_cache = []
    pooling_cache = []
    for c in range(conv_layers):
        convolution = conv2d(curr_input, kernels[c], biases[c], kernel_strides)
        #activated_convolution = sigmoid(convolution)
        activated_convolution = relu(convolution)
        pooling = avg_pooling(activated_convolution, poolings[c], pool_stride)
        convolution_cache.append(convolution)
        activation_cache.append(activated_convolution)
        pooling_cache.append(pooling)
        curr_input = pooling
    return [convolution_cache, activation_cache, pooling_cache]


def stability_check(array, threshold=150):
    flat = array.ravel()
    for f in range(len(flat)):
        if flat[f] > threshold:
            flat[f] = threshold
    return flat.reshape(array.shape[0], array.shape[1])


if __name__ == "__main__":
    # init hyperparams
    global_threshold = 1.5e30
    conv_layers = 2
    output_classes = 2
    eta = 4.2
    passes = 100
    x_input, y_input = 12, 12
    x_kernel, y_kernel = 2, 2
    pool_stride = 2
    pool_window = [2, 2]
    conv_stride = 1
    kernel_stride = 1
    dataset_size = 500
    print('init data and labels')
    dummy_data = [np.array([random.randint(1,256) for i in range(x_input * y_input)]).reshape(x_input, y_input) for j in
                  range(dataset_size)]
    dummy_answers = []
    for i in range(len(dummy_data)):
        normalize(dummy_data[i])
    for k in range(len(dummy_data)):
        chosen_index = random.choice((0, 1))
        if chosen_index == 1:
            d_lbl = np.array([0, 1]).reshape(2, 1)
        else:
            d_lbl = np.array([1, 0]).reshape(2, 1)
        dummy_answers.append(d_lbl)
    print('data init complete, init kernels and biases')
    kernels = [np.array([np.random.uniform() for i in range(x_kernel * y_kernel)], dtype=np.float).reshape(x_kernel, y_kernel)\
               for k in range(conv_layers)]
    poolings = [pool_window for i in range(conv_layers)]
    biases = [np.random.uniform() for i in range(conv_layers)]
    print('init kernels and biases complete')
    final_output_shape = infer_output_layer_shape(x_input, conv_layers, kernels, poolings)
    output_layer_weights = np.array([np.random.uniform() for i in range(output_classes * (final_output_shape**2))]).reshape(output_classes, final_output_shape**2)
    output_layer_biases = np.array([np.random.uniform() for i in range(output_classes)]).reshape(output_classes, 1)
    print('init learning....')
    for p in range(passes):
        pass_mse = 0
        pass_cross_entropy = 0
        pass_accuracy = 0
        for d in range(len(dummy_data)):
            # forward pass
            forward_pass = forward_conv_pass(dummy_data[d], conv_layers, kernels, biases, poolings)
            convolution_cache, activation_cache, pooling_cache = forward_pass[0], forward_pass[1], forward_pass[2]
            fc_result = pooling_cache[-1]
            fc_layer = np.ravel(fc_result).reshape(fc_result.shape[0] * fc_result.shape[1], 1)
            final_output = softmax(np.matmul(output_layer_weights, fc_layer) + output_layer_biases)
            prediction = list(final_output).index(max(final_output))
            if prediction == list(dummy_answers[d]).index(max(dummy_answers[d])):
                pass_accuracy += 1
            pass_mse += mse(prediction, dummy_answers[d])
            # backward pass
            # here, the last conv backprop is manual
            #error = (final_output - dummy_answers[d]) * final_output * (1 - final_output)
            error = (final_output - dummy_answers[d]) * grad_relu(final_output)
            delta_output_layer_weights = np.matmul(error, fc_layer.T)
            delta_fully_connected = np.matmul(output_layer_weights.T, error)
            delta_final_pool = fc_layer.reshape(final_output_shape, final_output_shape)
            delta_final_conv = upscale(delta_final_pool, convolution_cache[-1])
            #delta_final_conv_sigma = delta_final_conv * convolution_cache[-1] * (1 - convolution_cache[-1])
            delta_final_conv_sigma = delta_final_conv * grad_relu(convolution_cache[-1])
            delta_final_kernel = conv2d(np.rot90(pooling_cache[-2], 2), delta_final_conv_sigma, 0.3, kernel_stride)
            delta_final_bias = np.sum(delta_final_conv_sigma)
            # automatic now
            backward_kernel_cache = []
            backward_bias_cache = []
            delta_conv_sigma_cache = []
            curr_delta_conv_sigma = delta_final_conv_sigma
            curr_delta_pool = delta_final_pool
            # we add the original input to the pooling to automate the last backward pass
            pooling_cache.append(dummy_data[d])
            for j in range(conv_layers - 2, -1, -1):
                delta_conv_sigma = curr_delta_conv_sigma
                delta_curr_pool = conv2d(delta_conv_sigma, np.rot90(kernels[j + 1], 2), biases[j + 1], kernel_stride)
                delta_curr_conv = upscale(delta_curr_pool, convolution_cache[j])
                # stability checks
                delta_curr_conv = stability_check(delta_curr_conv, global_threshold)
                convolution_cache[j] = stability_check(convolution_cache[j], global_threshold)
                #curr_delta_conv_sigma = delta_curr_conv * convolution_cache[j] * (1 - convolution_cache[j])
                curr_delta_conv_sigma = delta_curr_conv * grad_relu(convolution_cache[j])
                delta_curr_kernel = conv2d(np.rot90(pooling_cache[j - 1], 2), curr_delta_conv_sigma, 0, kernel_stride)
                delta_curr_bias = np.sum(curr_delta_conv_sigma)
                kernels[j] = gradient_descent_update(kernels[j], delta_curr_kernel, eta)
                biases[j] = gradient_descent_update(biases[j], delta_curr_bias, eta)
                backward_kernel_cache.append(delta_curr_kernel)
                backward_bias_cache.append(delta_curr_bias)
                delta_conv_sigma_cache.append(delta_conv_sigma)
            output_layer_weights = gradient_descent_update(output_layer_weights, delta_output_layer_weights, eta)
            output_layer_biases = error
        print('pass = {}, pass_mse = {}, correct predictions = {}, pass_accuracy = {}%'.\
              format(p + 1, pass_mse, pass_accuracy, (pass_accuracy * 100) / len(dummy_answers)))
