import numpy as np


def conv(img, kernel, bias, s):
    x, y, z = img.shape[0], img.shape[1], img.shape[2]
    k_x, k_y, k_z = kernel.shape[0], kernel.shape[1], kernel.shape[2]
    if k_x > x or k_y > y or k_z > z or s > x or s > y or s > z:
        return None
    spat_dim = int(np.floor((x - k_x)/s) + 1)
    V = np.full((spat_dim, spat_dim, z), 0)
    x_spatial, y_spatial = 0, 0
    for x in range(spat_dim):
        for y in range(spat_dim):
            img_slice = img[x_spatial:k_x, y_spatial:k_y, 0:k_z]
            conv_out = np.sum(img_slice * kernel)
            for k in range(z):
                V[x, y, k] = conv_out + bias[k]
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
        output_shape = (o, o, depth)
        print('pass {}, input shape:{}, output shape:{}'.format(i+1, inp_shape, output_shape))
        conv_pass = conv(inp, kernel, bias, s)
        inp = conv_pass
    return inp


depth = 3
x_input, y_input = 100, 100
x_kernel, y_kernel = 2, 2
data = np.array([np.random.normal() for i in range(x_input * y_input * depth)]).reshape(x_input, y_input, depth)
kernel = np.array([np.random.normal() for i in range(x_kernel * y_kernel * depth)]).reshape(x_kernel, y_kernel, depth)
bias = np.array([np.random.normal() for i in range(depth)]).reshape(depth, 1)
x = np.full((x_input, y_input, depth), data)
get = conv_pass(x, kernel, bias)
print(get)
