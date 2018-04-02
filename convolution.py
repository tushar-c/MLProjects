import numpy as np


def conv(img, kernel, s=1):
    x, y, z = img.shape[0], img.shape[1], img.shape[2]
    k_x, k_y, k_z = kernel.shape[0], kernel.shape[1], kernel.shape[2]
    if k_x > x or k_y > y or k_z > z or s > x or s > y:
        return None
    spatial_dim = int(np.floor((x - k_x)/s) + 1)
    V = np.full((spatial_dim, spatial_dim, k_z), 0)
    x_spatial, y_spatial = 0, 0
    for a in range(spatial_dim):
        for b in range(spatial_dim):
            img_slice = img[x_spatial:k_x, y_spatial:k_y, 0:k_z]
            conv_out = np.sum(img_slice * kernel)
            V[a, b, :] = conv_out
            k_x, k_y = x_spatial + s, y_spatial + s
    return V


def conv2d(img, kernel, s=1):
    x, y = img.shape[0], img.shape[1]
    k_x, k_y = kernel.shape[0], kernel.shape[1]
    if k_x > x or k_y > y or s > x or s > y:
        return None
    spat_dim = int(np.floor((x - k_x)/s) + 1)
    V = np.full((spat_dim, spat_dim), 0)
    x_spatial, y_spatial = 0, 0
    for a in range(spat_dim):
        for b in range(spat_dim):
            img_slice = img[x_spatial:k_x, y_spatial:k_y]
            conv_out = np.sum(img_slice * kernel)
            V[a, b] = conv_out
            k_x, k_y = x_spatial + s, y_spatial + s
    return V


data = np.array([i for i in range(1, 28)]).reshape(3,3,3)
x = np.full((3,3,3), data)
kernel = np.array([1,1,1,1,1,1,1,1,1,1,1,1]).reshape(2,2,3)
bi = np.full((2,2,3), 1)
data1 = np.array([i for i in range(1, 10)]).reshape(3, 3)
x1 = np.full((3, 3), data1)
kernel1 = np.array([1, 1, 1, 1]).reshape(2, 2)
get = conv(x, kernel)
get_2d = conv2d(x1, kernel1)
print(get)
print(get_2d)
