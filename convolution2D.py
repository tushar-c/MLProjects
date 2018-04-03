import numpy as np


def conv2d(img, kernel, bias, s):
    x, y = img.shape[0], img.shape[1]
    k_x, k_y = kernel.shape[0], kernel.shape[1]
    if k_x > x or k_y > y or s > x or s > y:
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
    return V


def conv_pass(i, k, bias, s=2, passes=5):
    inp = i
    for i in range(passes):
        inp_shape = inp.shape
        o = int(np.floor((inp_shape[0] - x_kernel)/s) + 1)
        output_shape = (o, o)
        print('pass {}, input shape : {}, output shape : {}'.format(i+1, inp_shape, output_shape))
        conv_pass = conv2d(inp, kernel, bias, s)
        inp = conv_pass
    return inp


x_input, y_input = 100, 100
x_kernel, y_kernel = 2, 2
data = np.array([np.random.normal() for i in range(x_input * y_input)]).reshape(x_input, y_input)
kernel = np.array([np.random.normal() for i in range(x_kernel * y_kernel)]).reshape(x_kernel, y_kernel)
bias = np.random.normal()
x = np.full((x_input, y_input), data)
get = conv_pass(x, kernel, bias)
print(get)
