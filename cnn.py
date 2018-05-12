import conv_utils
import numpy as np
import pdb


def forward_conv_pass(input_, conv_layers, kernels, biases, poolings, kernel_strides=1, pool_stride=2):
    curr_input = input_
    convolution_cache = []
    activation_cache = []
    pooling_cache = []
    for c in range(conv_layers):
        convolution = conv_utils.conv2d(curr_input, kernels[c], biases[c], kernel_strides)
        activated_convolution = conv_utils.relu(convolution)
        pooling = conv_utils.avg_pooling(activated_convolution, poolings[c], pool_stride)
        convolution_cache.append(convolution)
        activation_cache.append(activated_convolution)
        pooling_cache.append(pooling)
        curr_input = pooling
    return [convolution_cache, activation_cache, pooling_cache]


def backward_conv_pass(input_, preactivated_output, feature, label, conv_layers, kernels, weights, biases,
                       fc_layer,pooling_cache, convolution_cache, final_output_shape, observed_probs,
                       kernel_stride=1, delta=1e-20):

    kernel_gradients, biases_gradients, output_layer_weights_gradients, output_layer_biases_gradients = [], [], [], []
    output_layer_weights = weights
    output_layer_biases = biases
    error = (input_ - label) / len(input_) * conv_utils.grad_softmax(preactivated_output)
    delta_output_layer_weights = np.matmul(error, fc_layer.T)
    delta_output_layer_biases = error
    output_layer_weights_gradients.append((output_layer_weights, delta_output_layer_weights))
    output_layer_biases_gradients.append((output_layer_biases, delta_output_layer_biases))
    delta_fc_layer = np.matmul(delta_output_layer_weights.T, error)
    delta_final_pool = delta_fc_layer.reshape(final_output_shape, final_output_shape)
    delta_final_conv = conv_utils.upscale(delta_final_pool, convolution_cache[-1])
    delta_final_conv_sigma = delta_final_conv * conv_utils.grad_relu(convolution_cache[-1])
    if conv_layers > 1:
        delta_final_kernel = conv_utils.conv2d(np.rot90(pooling_cache[-2], 2), delta_final_conv_sigma,
                                               biases[-1], kernel_stride)
        delta_final_conv_sigma = conv_utils.stability_check(delta_final_conv_sigma)
        delta_final_kernel = conv_utils.stability_check(delta_final_kernel)
        delta_final_bias = np.sum(delta_final_conv_sigma)
        kernel_gradients.append((kernels[-1], delta_final_kernel))
        biases_gradients.append((biases[-1], delta_final_bias))
        curr_delta_conv_sigma = delta_final_conv_sigma
        pooling_cache.append(feature)
        for j in range(conv_layers - 2, -1, -1):
            delta_conv_sigma = curr_delta_conv_sigma
            delta_curr_pool = conv_utils.conv2d(delta_conv_sigma, np.rot90(kernels[j + 1], 2), biases[j + 1], kernel_stride)
            delta_curr_conv = conv_utils.upscale(delta_curr_pool, convolution_cache[j])
            delta_curr_conv = conv_utils.stability_check(delta_curr_conv)
            curr_delta_conv_sigma = delta_curr_conv * conv_utils.grad_relu(convolution_cache[j])
            delta_curr_kernel = conv_utils.conv2d(np.rot90(pooling_cache[j - 1], 2), curr_delta_conv_sigma, 0, kernel_stride)
            delta_curr_kernel = conv_utils.stability_check(delta_curr_kernel)
            delta_curr_bias = np.sum(curr_delta_conv_sigma)
            kernel_gradients.append((kernels[j], delta_curr_kernel))
            biases_gradients.append((biases[j], delta_curr_bias))
    else:
        delta_final_kernel = conv_utils.conv2d(np.rot90(feature, 2), delta_final_conv_sigma,
                                               biases[-1], kernel_stride)
        delta_final_conv_sigma = conv_utils.stability_check(delta_final_conv_sigma)
        delta_final_kernel = conv_utils.stability_check(delta_final_kernel)
        delta_final_bias = np.sum(delta_final_conv_sigma)
        kernel_gradients.append((kernels[-1], delta_final_kernel))
        biases_gradients.append((biases[-1], delta_final_bias))

    return kernel_gradients, biases_gradients, output_layer_weights_gradients, output_layer_biases_gradients


def train(features, labels, conv_layers, kernels, biases, poolings, eta, observed_probs,
          input_shape=28, output_classes=10, epochs=1000, sample_size=100):

    print('training cnn with {} layer(s) for {} epochs with learning rate {}'.format(conv_layers, epochs, eta))
    final_output_shape = conv_utils.infer_output_layer_shape(input_shape, conv_layers, kernels, poolings,
                                                            kernel_strides=1, pool_stride=2)
    output_layer_weights = conv_utils.output_layer_weights_biases(output_classes, final_output_shape)[0]
    output_layer_biases = conv_utils.output_layer_weights_biases(output_classes, final_output_shape)[1]
    for e in range(epochs):
        mse = 0
        correct = 0
        for d in range(len(features)):
            if d % 10000 == 0 and d != 0:
                print('epoch {} / {}, sample {} / {}'.format(e + 1, epochs, d, len(features)))
            forward_pass = forward_conv_pass(features[d], conv_layers, kernels, biases, poolings)
            convolution_cache, activation_cache, pooling_cache = forward_pass[0], forward_pass[1], forward_pass[2]
            final_pooling_layer = pooling_cache[-1]
            fc_layer = np.ravel(final_pooling_layer).reshape(final_pooling_layer.shape[0] * final_pooling_layer.shape[1], 1)
            preactivated_output = np.matmul(output_layer_weights, fc_layer) + output_layer_biases
            final_output = conv_utils.stable_softmax(preactivated_output)
            mse += conv_utils.mse(conv_utils.softmax_prediction(final_output), labels[d])
            if conv_utils.check_prediction(final_output, labels[d]) == 0:
                correct += 1
            backward_pass = backward_conv_pass(final_output, preactivated_output, features[d], labels[d],
                                               conv_layers, kernels, output_layer_weights, output_layer_biases,
                                                fc_layer, pooling_cache, convolution_cache, final_output_shape,
                                               observed_probs)

            kernels, biases, output_layer_weights, output_layer_biases = conv_utils.update_params(kernels, biases,
                                                                                                  backward_pass, eta)
        print('epoch {} / {}, epoch accuracy = {} %, correct predictions = {} out of {}'\
              .format(e + 1, epochs,(correct * 100) / len(features), correct, len(features)))
    return kernels, biases, output_layer_weights, output_layer_biases


mnist_train_data = conv_utils.get_mnist_data(sliced=100)
train_features = mnist_train_data[0]
train_labels = mnist_train_data[1]
output_classes = 10
conv_layers = 1
eta = 0.403
epochs = 200
observed_probs = conv_utils.get_empirical_probs(output_classes, train_labels)
kernels = conv_utils.init_kernels(conv_layers, shape=2)
biases = conv_utils.init_biases(conv_layers)
poolings = conv_utils.init_poolings(conv_layers)
train_cnn = train(train_features, train_labels, conv_layers, kernels, biases, poolings, eta, observed_probs,
                  output_classes=output_classes, epochs=epochs)
