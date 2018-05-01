import numpy as np
from keras.datasets import mnist


def normalize(matrix, delta=0.00000001):
    for i in range(matrix.shape[1]):
        stddev = np.std(matrix[:, i])
        mean = np.mean(matrix[:, i])
        if stddev == 0:
            stddev += delta
        matrix[:, i] = (matrix[:, i] - mean) / stddev


def l2_norm(x):
    return np.power(np.sum(np.power(x, 2)), 1/2)


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


def mse(y, y_):
    return np.sum(np.power(y - y_, 2)) / 2


def grad_mse(y, y_):
    return np.sum((y - y_))


def gradient_descent_update(x, grad, eta):
    return (x + eta * grad)


def conv2d(img, kernel, bias, s):
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
            x[i] = 1
    x = x.reshape(orig_shape[0], orig_shape[1])
    return x


def relu(x):
    orig_shape = (x.shape[0], x.shape[1])
    x = x.ravel()
    for i in range(len(x)):
        x[i] = max(0, x[i])
    x = x.reshape(orig_shape[0], orig_shape[1])
    return x


def stable_sigmoid(x):
    if x.shape[1] > 1:
        squashed_x = np.zeros(x.shape[0] * x.shape[1]).reshape(x.shape[0], x.shape[1])
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                if x[j][k] >= 0:
                    z = np.power(np.e, -x[j][k])
                    squashed_x[j][k] = 1 / (1 + z)
                else:
                    z = np.power(np.e, x[j][k])
                    squashed_x[j][k] = z / (1 + z)
        return squashed_x
    else:
        squashed_x = np.zeros(x.shape[0]).reshape(x.shape[0], 1)
        for j in range(len(x)):
            if x[j] >= 0:
                z = np.power(np.e, -x[j])
                squashed_x[j] = 1 / (1 + z)
            else:
                z = np.power(np.e, x[j])
                squashed_x[j] = z / (1 + z)
        return squashed_x


def sigmoid_gradient(x):
    return stable_sigmoid(x) * (1 - stable_sigmoid(x))


def sigmoid(x, u_threshold=1.5e80, l_threshold=1.5e-80):
    squashed = 1/(1 + np.power(np.e, -x))
    for i in range(len(squashed)):
        if squashed[i] > u_threshold:
            squashed[i] = u_threshold
        if squashed[i] < l_threshold:
            squashed[i] = l_threshold
    return squashed


def softmax(x, threshold=1.5e80):
    try:
        numerator = np.power(np.e, x)
        denominator = np.sum(np.power(np.e, x))
    except RuntimeWarning:
        print(x, numerator, denominator)
    for i in range(len(numerator)):
        if numerator[i] > threshold:
            numerator[i] = threshold
    if denominator > threshold:
        denominator = threshold
    return numerator/denominator


def stable_softmax(x):
    y = (x - np.max(x))
    return np.power(np.e, y) / np.sum(np.power(np.e, y))


def output_layer_weights_biases(output_classes, final_output_shape):
    weights = np.random.rand(output_classes, final_output_shape ** 2)
    biases = np.array([np.random.uniform() for i in range(output_classes)]).reshape(output_classes, 1)
    return [weights, biases]


def get_mnist_data(sliced=60000, test_offset=1000, output_classes=10):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    train_features = x_train[:sliced]
    pre_train_labels = y_train[:sliced]
    train_labels = []
    test_features = x_test[sliced:sliced + test_offset]
    pre_test_labels = y_test[sliced:sliced + test_offset]
    test_labels = []
    for p in range(len(pre_train_labels)):
        answer = pre_train_labels[p]
        one_hot = np.zeros(output_classes).reshape(output_classes, 1)
        one_hot[answer - 1] = 1
        train_labels.append(one_hot)
    train_labels = np.array(train_labels)
    for p in range(len(pre_test_labels)):
        answer = pre_test_labels[p]
        one_hot = np.zeros(output_classes).reshape(output_classes, 1)
        one_hot[answer - 1] = 1
        test_labels.append(one_hot)
    test_labels = np.array(test_labels)
    return [train_features, train_labels, test_features, test_labels]


def get_mnist_data_test(sliced=60000, output_classes=10):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255
    test_features = x_test[:sliced]
    pre_test_labels = y_test[:sliced]
    test_labels = []
    for p in range(len(pre_test_labels)):
        answer = pre_test_labels[p]
        one_hot = np.zeros(output_classes).reshape(output_classes, 1)
        one_hot[answer - 1] = 1
        test_labels.append(one_hot)
    test_labels = np.array(test_labels)
    return [test_labels, test_labels]



def init_kernels(quantity, shape=2):
    kernels = []
    for k in range(quantity):
        kernel = np.array([np.random.uniform(-1/4, 1/4) for i in range(shape**2)], dtype=np.float).reshape(shape, shape)
        kernels.append(kernel)
    return kernels


def init_biases(quantity):
    biases = []
    for q in range(quantity):
        biases.append(np.random.normal())
    return biases


def init_poolings(quantity, shape=[2,2]):
    return [shape for i in range(quantity)]


def softmax_prediction(softmax_array):
    return list(softmax_array).index(np.max(softmax_array))


def check_prediction(softmax_output, true_label_vector):
    prediction = softmax_prediction(softmax_output)
    true_label = list(true_label_vector).index(max(true_label_vector))
    if prediction == true_label:
        return 0
    else:
        return 1


def stability_check(array, u_threshold=10e80, l_threshold=10e-80):
    flat = array.ravel()
    for f in range(len(flat)):
        if flat[f] >= u_threshold:
            flat[f] = u_threshold
        if flat[f] <= l_threshold:
            flat[f] = l_threshold
    return flat.reshape(array.shape[0], array.shape[1])


def update_params(kernels, biases, backward_pass, eta):
    kernel_gradients = backward_pass[0]
    biases_gradients = backward_pass[1]
    output_layer_weights_gradients = backward_pass[2]
    output_layer_biases_gradients = backward_pass[3]

    output_layer_weights = gradient_descent_update(output_layer_weights_gradients[0][0], \
                                                              output_layer_weights_gradients[0][1], eta)
    output_layer_biases = gradient_descent_update(output_layer_biases_gradients[0][0], \
                                                             output_layer_biases_gradients[0][1], eta)

    for k in range(len(kernel_gradients)):
        kernels[k] = gradient_descent_update(kernel_gradients[k][0], kernel_gradients[k][1], eta)
    for b in range(len(biases_gradients)):
        biases[k] = gradient_descent_update(biases_gradients[k][0], biases_gradients[k][1], eta)

    return kernels, biases, output_layer_weights, output_layer_biases


